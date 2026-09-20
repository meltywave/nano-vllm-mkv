from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import CacheTier, Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.block_manager = BlockManager(
            config.num_kvcache_blocks,
            config.kvcache_block_size,
            config.num_cpu_kvcache_blocks,
            config.num_ssd_kvcache_blocks,
        )
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.swapped: deque[Sequence] = deque()

        self.num_swap_in_blocks = 0
        self.num_swap_out_blocks = 0
        self.num_gpu_to_cpu_blocks = 0
        self.num_cpu_to_ssd_blocks = 0
        self.num_gpu_to_ssd_blocks = 0
        self.num_cpu_to_gpu_blocks = 0
        self.num_ssd_to_gpu_blocks = 0

    def is_finished(self):
        return not self.waiting and not self.running and not self.swapped

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def _record_transfers(self, transfers):
        for transfer in transfers:
            src, dst = transfer.src_tier, transfer.dst_tier
            if src == CacheTier.GPU:
                self.num_swap_out_blocks += 1
            if dst == CacheTier.GPU:
                self.num_swap_in_blocks += 1
            if (src, dst) == (CacheTier.GPU, CacheTier.CPU):
                self.num_gpu_to_cpu_blocks += 1
            elif (src, dst) == (CacheTier.CPU, CacheTier.SSD):
                self.num_cpu_to_ssd_blocks += 1
            elif (src, dst) == (CacheTier.GPU, CacheTier.SSD):
                self.num_gpu_to_ssd_blocks += 1
            elif (src, dst) == (CacheTier.CPU, CacheTier.GPU):
                self.num_cpu_to_gpu_blocks += 1
            elif (src, dst) == (CacheTier.SSD, CacheTier.GPU):
                self.num_ssd_to_gpu_blocks += 1

    def _demote_cold_cpu_sequences(self, required_cpu_blocks: int):
        transfers = []
        block_manager = self.block_manager
        if block_manager.num_free_blocks(CacheTier.CPU) >= required_cpu_blocks:
            return transfers

        # The right side contains sequences that have waited the longest.
        for candidate in reversed(self.swapped):
            if candidate.block_table_tier != CacheTier.CPU:
                continue
            if not block_manager.can_swap(candidate, CacheTier.SSD):
                continue
            mappings = block_manager.swap(candidate, CacheTier.SSD)
            transfers.extend(mappings)
            self._record_transfers(mappings)
            if block_manager.num_free_blocks(CacheTier.CPU) >= required_cpu_blocks:
                break
        return transfers

    def _schedule_swapped(self):
        scheduled_seqs = []
        blocks_to_swap_in = []
        total_gpu_blocks = len(self.block_manager.blocks)

        while self.swapped and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.swapped[0]

            extra = int(len(seq) % self.block_size == 1)
            required = len(seq.block_table) + extra

            if required > total_gpu_blocks:
                raise RuntimeError(
                    f"Sequence {seq.seq_id} requires {required} GPU KV blocks, "
                    f"but only {total_gpu_blocks} blocks exist. "
                    "Whole-sequence swapping cannot serve this sequence."
                )

            if not self.block_manager.can_swap_in(seq, extra):
                break

            self.swapped.popleft()

            mappings = self.block_manager.swap_in(seq)
            blocks_to_swap_in.extend(mappings)
            self._record_transfers(mappings)

            seq.status = SequenceStatus.RUNNING
            seq.is_prefill = False
            seq.num_scheduled_tokens = 1
            self.block_manager.may_append(seq)

            self.running.append(seq)
            scheduled_seqs.append(seq)

        return scheduled_seqs, blocks_to_swap_in

    def schedule(self):
        scheduled_seqs = []
        blocks_to_swap_in = []
        blocks_to_swap_out = []
        num_batched_tokens = 0

        if not self.running and self.swapped:
            scheduled_seqs, blocks_to_swap_in = self._schedule_swapped()
            if scheduled_seqs:
                return scheduled_seqs, False, blocks_to_swap_in, blocks_to_swap_out

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    if seq.num_blocks > len(self.block_manager.blocks):
                        raise RuntimeError(
                            f"Sequence {seq.seq_id} requires {seq.num_blocks} GPU KV blocks, "
                            f"but only {len(self.block_manager.blocks)} blocks exist. "
                            "Whole-sequence swapping cannot serve this sequence."
                        )
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True, blocks_to_swap_in, blocks_to_swap_out

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            can_schedule = True

            while not self.block_manager.can_append(seq):
                if self.running:
                    victim = self.running.pop()
                else:
                    required = len(seq.block_table) + 1
                    total = len(self.block_manager.blocks)
                    if required > total:
                        raise RuntimeError(
                            f"Sequence {seq.seq_id} requires {required} GPU KV blocks, "
                            f"but only {total} blocks exist. "
                            "Whole-sequence swapping cannot serve this sequence."
                        )
                    victim = seq
                    can_schedule = False
                blocks_to_swap_out.extend(self.preempt(victim))

                if victim is seq:
                    break
            if not can_schedule:
                continue

            seq.num_scheduled_tokens = 1
            seq.is_prefill = False
            self.block_manager.may_append(seq)
            scheduled_seqs.append(seq)

        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False, blocks_to_swap_in, blocks_to_swap_out

    def preempt(self, seq: Sequence):
        block_manager = self.block_manager
        num_blocks = len(seq.block_table)
        mappings = []

        if num_blocks <= block_manager.capacity(CacheTier.CPU):
            mappings.extend(self._demote_cold_cpu_sequences(num_blocks))

        if block_manager.can_swap_out(seq, CacheTier.CPU):
            swap_out = block_manager.swap_out(seq, CacheTier.CPU)
        elif block_manager.can_swap_out(seq, CacheTier.SSD):
            swap_out = block_manager.swap_out(seq, CacheTier.SSD)
        else:
            swap_out = []

        if swap_out:
            mappings.extend(swap_out)
            self._record_transfers(swap_out)

            seq.status = SequenceStatus.SWAPPED
            self.swapped.appendleft(seq)
            return mappings

        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)
        return mappings

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
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
