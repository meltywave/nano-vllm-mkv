"""
三级缓存感知调度器
支持 GPU + CPU + DISK 三级缓存的换入换出调度

调度策略：
- 瀑布式换出：GPU 满 → 换出到 CPU；CPU 满 → 换出到 DISK
- 逐级换入：DISK → CPU → GPU
- 基于热度的换出决策（LRU）
"""

from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.multi_level_block_manager import (
    MultiLevelBlockManager,
    BlockLocation,
)


class MultiLevelScheduler:
    """
    三级缓存感知调度器

    队列结构：
    - waiting: 等待 prefill 的序列
    - running: 正在运行的序列（块主要在 GPU）
    - swapped_out: 已换出的序列（块在 CPU 或 DISK）

    调度流程：
    1. 尝试调度 waiting 队列的 prefill 请求
    2. 检查 GPU/CPU 使用率，触发瀑布式换出
    3. 尝试换入 swapped_out 队列中的热序列
    4. 调度 decode 请求（确保块在 GPU 中）
    """

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size

        # 多级缓存配置
        self.enable_multilevel = getattr(config, "enable_multilevel_kvcache", False)
        self.cpu_num_blocks = getattr(config, "cpu_num_kvcache_blocks", 0)
        self.disk_num_blocks = getattr(config, "disk_num_kvcache_blocks", 0)
        self.enable_disk = self.disk_num_blocks > 0
        self.swap_watermark_high = getattr(config, "swap_watermark_high", 0.9)
        self.swap_watermark_low = getattr(config, "swap_watermark_low", 0.7)
        self.replacement_policy = getattr(config, "replacement_policy", "lru")
        self.enable_prefetch = getattr(config, "enable_prefetch", True)
        self.prefetch_lookahead = getattr(config, "prefetch_lookahead", 2)

        # 块管理器
        if self.enable_multilevel and self.cpu_num_blocks > 0:
            self.block_manager = MultiLevelBlockManager(
                gpu_num_blocks=config.num_kvcache_blocks,
                cpu_num_blocks=self.cpu_num_blocks,
                block_size=config.kvcache_block_size,
                disk_num_blocks=self.disk_num_blocks,
                disk_cache_dir=getattr(config, "disk_cache_dir", "./kv_disk_cache"),
                replacement_policy=self.replacement_policy,
                swap_watermark_high=self.swap_watermark_high,
                swap_watermark_low=self.swap_watermark_low,
                enable_prefix_caching=True,
            )
            self.use_multilevel = True
        else:
            # 降级为原有的 BlockManager
            from nanovllm.engine.block_manager import BlockManager
            self.block_manager = BlockManager(
                config.num_kvcache_blocks,
                config.kvcache_block_size,
            )
            self.use_multilevel = False

        # 队列
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.swapped_out: deque[Sequence] = deque()  # 已换出的序列（CPU/DISK）

        # 统计
        self.stats = {
            "total_swap_out": 0,
            "total_swap_in": 0,
            "total_preempt": 0,
            "total_disk_swap_out": 0,
            "total_disk_swap_in": 0,
        }

    def is_finished(self):
        return not self.waiting and not self.running and not self.swapped_out

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """调度函数，返回 (调度的序列列表, 是否是 prefill 阶段)"""
        if not self.use_multilevel:
            return self._schedule_basic()
        return self._schedule_multilevel()

    def _schedule_basic(self) -> tuple[list[Sequence], bool]:
        """基础调度逻辑（与原 Scheduler 一致）"""
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:
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
            return scheduled_seqs, True

        # decode
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
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def _schedule_multilevel(self) -> tuple[list[Sequence], bool]:
        """三级缓存感知的调度逻辑"""
        scheduled_seqs = []
        num_batched_tokens = 0

        # Step 1: 尝试处理 waiting 队列中的 prefill 请求
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break

            # 检查是否能分配
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    # 空间不足，尝试换出一些序列腾出空间
                    if not self._try_swap_out_for_new_seq(seq):
                        break  # 换出也不够，等待
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens

            if remaining < num_tokens and scheduled_seqs:
                break

            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                self.block_manager.allocate(seq, num_cached_blocks)

            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens

            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)

            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # Step 2: 检查 GPU 使用率，触发换出（瀑布式）
        self._check_and_swap_out()

        # Step 3: 尝试换入 swapped_out 队列中的热序列
        self._try_swap_in()

        # Step 4: decode 阶段
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()

            # 确保所有需要的块都在 GPU 中
            self._ensure_seq_blocks_in_gpu(seq)

            # 检查是否能追加新块
            while not self.block_manager.can_append(seq):
                # 空间不足，尝试换出其他序列
                if self.running:
                    victim = self.running.pop()
                    self._swap_out_or_preempt(victim)
                else:
                    # 没有其他序列可换出，只能 preempt 当前序列
                    self._swap_out_or_preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)

        if not scheduled_seqs:
            # 如果 running 队列空了，尝试从 swapped_out 换入
            self._try_swap_in()
            # 再试一次
            while self.running and len(scheduled_seqs) < self.max_num_seqs:
                seq = self.running.popleft()
                if self.block_manager.can_append(seq):
                    seq.num_scheduled_tokens = 1
                    seq.is_prefill = False
                    self.block_manager.may_append(seq)
                    scheduled_seqs.append(seq)
                else:
                    self.running.appendleft(seq)
                    break

        assert scheduled_seqs, "No sequences to schedule"
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    # ==================== 换出逻辑 ====================

    def _try_swap_out_for_new_seq(self, new_seq: Sequence) -> bool:
        """尝试换出一些序列，为新序列腾出 GPU 空间"""
        if not self.running:
            return False

        needed_blocks = new_seq.num_blocks
        freed_blocks = 0

        # 按热度排序，最不活跃的先换出
        candidates = sorted(
            self.running,
            key=lambda s: (s.last_access_time, -len(s)),
        )

        for seq in candidates:
            if freed_blocks >= needed_blocks:
                break

            # 计算这个序列有多少 GPU 块
            gpu_blocks = sum(
                1 for loc in seq.block_locations if loc == BlockLocation.GPU
            )

            if gpu_blocks > 0:
                self._swap_out_sequence(seq)
                freed_blocks += gpu_blocks

        return freed_blocks >= needed_blocks

    def _check_and_swap_out(self):
        """检查 GPU/CPU 使用率，如果过高则触发瀑布式换出"""
        if not self.use_multilevel:
            return

        stats = self.block_manager.get_stats()

        # GPU 换出到 CPU
        if stats.gpu_utilization() >= self.swap_watermark_high:
            target_blocks = int(self.block_manager.gpu_num_blocks * self.swap_watermark_low)
            blocks_to_free = stats.gpu_used_blocks - target_blocks

            if blocks_to_free > 0:
                self._swap_out_gpu_to_cpu(blocks_to_free)

        # CPU 换出到 DISK
        if self.enable_disk and stats.cpu_utilization() >= self.swap_watermark_high:
            target_blocks = int(self.block_manager.cpu_num_blocks * self.swap_watermark_low)
            blocks_to_free = stats.cpu_used_blocks - target_blocks

            if blocks_to_free > 0:
                self._swap_out_cpu_to_disk(blocks_to_free)

    def _swap_out_gpu_to_cpu(self, num_blocks: int):
        """将 GPU 中的冷序列换出到 CPU"""
        candidates = sorted(
            self.running,
            key=lambda s: (s.last_access_time, -len(s)),
        )

        freed_blocks = 0
        for seq in candidates:
            if freed_blocks >= num_blocks:
                break

            gpu_blocks = sum(
                1 for loc in seq.block_locations if loc == BlockLocation.GPU
            )

            if gpu_blocks > 0:
                self._swap_out_sequence(seq)
                freed_blocks += gpu_blocks

    def _swap_out_cpu_to_disk(self, num_blocks: int):
        """将 CPU 中的冷序列换出到 DISK"""
        # 从 swapped_out 中找已经在 CPU 的序列，换出到 DISK
        candidates = sorted(
            self.swapped_out,
            key=lambda s: (s.last_access_time, -len(s)),
        )

        freed_blocks = 0
        for seq in candidates:
            if freed_blocks >= num_blocks:
                break

            cpu_blocks = sum(
                1 for loc in seq.block_locations if loc == BlockLocation.CPU
            )

            if cpu_blocks > 0:
                self._swap_out_sequence_to_disk(seq)
                freed_blocks += cpu_blocks

    def _swap_out_sequence(self, seq: Sequence):
        """将序列从 GPU 换出到 CPU"""
        if seq.status == SequenceStatus.SWAPPED_OUT:
            return

        # 从 running 队列中移除
        if seq in self.running:
            self.running.remove(seq)

        seq.status = SequenceStatus.SWAPPED_OUT
        self.swapped_out.append(seq)
        self.stats["total_swap_out"] += 1

        # 调用块管理器执行实际的块换出（元数据层面）
        if self.use_multilevel:
            gpu_ids, cpu_ids = self.block_manager.swap_out_sequence_to_cpu(seq)
            self.stats["total_swap_out_blocks"] = self.stats.get("total_swap_out_blocks", 0) + len(gpu_ids)
            # 注意：实际的数据拷贝由 ModelRunner 协调
            # 这里只更新元数据状态

    def _swap_out_sequence_to_disk(self, seq: Sequence):
        """将序列从 CPU 换出到 DISK"""
        if not self.enable_disk:
            return

        # 调用块管理器执行实际的块换出（元数据层面）
        cpu_ids, disk_ids = self.block_manager.swap_out_sequence_to_disk(seq)
        self.stats["total_disk_swap_out"] += len(cpu_ids)
        self.stats["total_swap_out_blocks"] = self.stats.get("total_swap_out_blocks", 0) + len(cpu_ids)
        # 注意：实际的数据拷贝由 ModelRunner 协调
        # 这里只更新元数据状态

    # ==================== 换入逻辑 ====================

    def _try_swap_in(self):
        """尝试将 swapped_out 队列中的热序列换入 GPU"""
        if not self.swapped_out:
            return

        stats = self.block_manager.get_stats()
        gpu_free = self.block_manager.gpu_num_blocks - stats.gpu_used_blocks

        # 预留一些空间给新请求和追加块
        reserve_blocks = 10
        available_blocks = gpu_free - reserve_blocks

        if available_blocks <= 0:
            return

        # 按优先级排序：访问次数多的、最近访问的优先
        candidates = sorted(
            self.swapped_out,
            key=lambda s: (-s.access_count, s.last_access_time),
        )

        swapped_in = []
        for seq in candidates:
            needed_blocks = len(seq.block_table)
            if needed_blocks <= available_blocks:
                self._swap_in_sequence(seq)
                available_blocks -= needed_blocks
                swapped_in.append(seq)
            else:
                continue

        # 从 swapped_out 中移除已换入的
        for seq in swapped_in:
            self.swapped_out.remove(seq)

    def _swap_in_sequence(self, seq: Sequence):
        """将序列从 swapped_out 状态换入到 running"""
        if seq.status != SequenceStatus.SWAPPED_OUT:
            return

        if seq in self.swapped_out:
            self.swapped_out.remove(seq)

        seq.status = SequenceStatus.RUNNING
        self.running.append(seq)
        self.stats["total_swap_in"] += 1

    # ==================== 其他方法 ====================

    def _swap_out_or_preempt(self, seq: Sequence):
        """
        优先换出，不行就 preempt（释放）

        换出的好处：保留 KV Cache，下次可以快速恢复
        preempt 的好处：释放更多空间
        """
        if self.use_multilevel and self.cpu_num_blocks > 0:
            stats = self.block_manager.get_stats()
            cpu_free = self.block_manager.cpu_num_blocks - stats.cpu_used_blocks

            seq_blocks = len(seq.block_table)
            if cpu_free >= seq_blocks or self.enable_disk:
                # 有空间（或可以换出到磁盘），换出
                self._swap_out_sequence(seq)
                return

        # 没有空间，只能 preempt
        self.preempt(seq)

    def _ensure_seq_blocks_in_gpu(self, seq: Sequence):
        """确保序列的所有块都在 GPU 中（用于 decode 前）"""
        if not self.use_multilevel:
            return

        has_non_gpu_blocks = any(
            loc != BlockLocation.GPU for loc in seq.block_locations
        )
        if not has_non_gpu_blocks:
            return

        # 确保所有块都在 GPU
        self.block_manager.ensure_blocks_in_gpu(seq, 0, len(seq.block_table))

    def preempt(self, seq: Sequence):
        """抢占序列（释放 KV Cache，重新排队）"""
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)
        self.stats["total_preempt"] += 1

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        """后处理：更新块哈希、更新缓存 token 数、检查是否结束"""
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0

            # 更新访问时间
            seq.access_count += 1
            seq.last_access_time = self._get_current_time()

            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue

            seq.append_token(token_id)

            if (
                not seq.ignore_eos and token_id == self.eos
            ) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                if seq in self.running:
                    self.running.remove(seq)
                elif seq in self.swapped_out:
                    self.swapped_out.remove(seq)

    def _get_current_time(self) -> float:
        """获取当前时间（用于 LRU）"""
        if not hasattr(self, "_time_counter"):
            self._time_counter = 0.0
        self._time_counter += 1.0
        return self._time_counter

    def get_stats(self) -> dict:
        """获取调度统计信息"""
        stats = self.stats.copy()
        stats["waiting"] = len(self.waiting)
        stats["running"] = len(self.running)
        stats["swapped_out"] = len(self.swapped_out)

        if self.use_multilevel:
            cache_stats = self.block_manager.get_stats()
            stats["gpu_utilization"] = cache_stats.gpu_utilization()
            stats["cpu_utilization"] = cache_stats.cpu_utilization()
            stats["disk_utilization"] = cache_stats.disk_utilization()
            stats["gpu_used_blocks"] = cache_stats.gpu_used_blocks
            stats["cpu_used_blocks"] = cache_stats.cpu_used_blocks
            stats["disk_used_blocks"] = cache_stats.disk_used_blocks
            stats["prefix_cache_hit_rate"] = cache_stats.prefix_cache_hit_rate()
            stats["total_swap_in_blocks"] = cache_stats.total_swap_in_blocks
            stats["total_swap_out_blocks"] = cache_stats.total_swap_out_blocks

        return stats
