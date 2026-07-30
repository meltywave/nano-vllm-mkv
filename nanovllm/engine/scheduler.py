from collections import deque
import logging

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager

# 模块独立日志，避免循环导入
logger = logging.getLogger("scheduler")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


class Scheduler:

    def __init__(self, config: Config, model_runner=None):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(
            num_blocks=config.num_kvcache_blocks,
            block_size=config.kvcache_block_size,
            cfg=config
        )
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.model_runner = model_runner

    def set_model_runner(self, model_runner):
        self.model_runner = model_runner
        self.block_manager.set_model_runner(model_runner)
        logger.info("[Scheduler] ModelRunner已同步注入BlockManager，多级Swap模块就绪")

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill 阶段
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                num_new_blocks = seq.num_blocks - num_cached_blocks
                # 需要新块时触发多级驱逐
                if num_new_blocks > 0 and len(self.block_manager.free_block_ids) < num_new_blocks:
                    logger.warning(f"[Scheduler Prefill] 空间不足，需要{num_new_blocks}个新块，当前空闲{len(self.block_manager.free_block_ids)}")
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            
            if remaining < num_tokens and scheduled_seqs:
                break
            
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks, self.model_runner)
            
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode 阶段
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
                continue
            break

        if not scheduled_seqs:
            return [], False

        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)
        logger.info(f"[Scheduler Preempt] 序列被抢占，加入等待队列头部")

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        finished_count = 0
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
                finished_count += 1
        if finished_count > 0:
            logger.info(f"[Scheduler] 本轮结束{finished_count}条序列")