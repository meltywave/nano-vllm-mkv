from collections import deque
from math import floor

from nanovllm.config import Config
from nanovllm.engine.sequence import CacheTier, Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.config = config
        self.cache_tiers = tuple(CacheTier(value) for value in config.kv_cache_tiers)
        self.block_manager = BlockManager(
            config.num_kvcache_blocks,
            config.kvcache_block_size,
            config.num_cpu_kvcache_blocks,
            config.num_ssd_kvcache_blocks,
            config.num_remote_kvcache_blocks,
            config.kv_enable_prefix_cache,
        )
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.max_transfer_blocks = config.kv_max_transfer_blocks
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.swapped: deque[Sequence] = deque()
        self.step_index = 0
        self.watermark_triggers = {tier.value: 0 for tier in self.cache_tiers}
        self.capacity_pressure = {tier.value: 0 for tier in self.cache_tiers}
        self.num_preemptions = 0
        self.num_recompute_preemptions = 0
        self.transfer_counts = {
            f"{src.value}_to_{dst.value}": 0
            for src in CacheTier
            for dst in CacheTier
            if src != dst
        }

        self.num_swap_in_blocks = 0
        self.num_swap_out_blocks = 0
        self.num_gpu_to_cpu_blocks = 0
        self.num_cpu_to_ssd_blocks = 0
        self.num_gpu_to_ssd_blocks = 0
        self.num_cpu_to_gpu_blocks = 0
        self.num_ssd_to_gpu_blocks = 0
        self.num_gpu_to_remote_blocks = 0
        self.num_cpu_to_remote_blocks = 0
        self.num_ssd_to_remote_blocks = 0
        self.num_remote_to_gpu_blocks = 0

    def is_finished(self):
        return not self.waiting and not self.running and not self.swapped

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def _record_transfers(self, transfers):
        for transfer in transfers:
            src, dst = transfer.src_tier, transfer.dst_tier
            self.transfer_counts[f"{src.value}_to_{dst.value}"] += 1
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
            elif (src, dst) == (CacheTier.GPU, CacheTier.REMOTE):
                self.num_gpu_to_remote_blocks += 1
            elif (src, dst) == (CacheTier.CPU, CacheTier.REMOTE):
                self.num_cpu_to_remote_blocks += 1
            elif (src, dst) == (CacheTier.SSD, CacheTier.REMOTE):
                self.num_ssd_to_remote_blocks += 1
            elif (src, dst) == (CacheTier.REMOTE, CacheTier.GPU):
                self.num_remote_to_gpu_blocks += 1

    def _lower_tiers(self, tier: CacheTier):
        index = self.cache_tiers.index(tier)
        return self.cache_tiers[index + 1:]

    def _ensure_free_blocks(self, tier: CacheTier, required_blocks: int):
        transfers = []
        block_manager = self.block_manager
        if block_manager.num_free_blocks(tier) >= required_blocks:
            return transfers

        candidates = sorted(
            (seq for seq in self.swapped if seq.block_table_tier == tier),
            key=lambda seq: (seq.last_used_step, seq.seq_id),
        )
        for candidate in candidates:
            candidate_blocks = len(candidate.block_table)

            for target_tier in self._lower_tiers(tier):
                if candidate_blocks > block_manager.capacity(target_tier):
                    continue
                nested = self._ensure_free_blocks(
                    target_tier, candidate_blocks
                )
                transfers.extend(nested)
                if not block_manager.can_swap(candidate, target_tier):
                    continue
                mappings = block_manager.swap(candidate, target_tier)
                candidate.num_swaps += 1
                transfers.extend(mappings)
                self._record_transfers(mappings)
                break

            if block_manager.num_free_blocks(tier) >= required_blocks:
                break
        return transfers

    def _transfer_budget_allows(self, current: int, additional: int):
        return (
            self.max_transfer_blocks == -1
            or current + additional <= self.max_transfer_blocks
        )

    def _demote_swapped_tier(self, tier: CacheTier):
        block_manager = self.block_manager
        capacity = block_manager.capacity(tier)
        if capacity == 0:
            return []
        high, low = self.config.watermarks(tier.value)
        if block_manager.utilization(tier) <= high:
            return []

        self.watermark_triggers[tier.value] += 1
        target_used = floor(capacity * low)
        transfers = []
        candidates = sorted(
            (seq for seq in self.swapped if seq.block_table_tier == tier),
            key=lambda seq: (seq.last_used_step, seq.seq_id),
        )
        for candidate in candidates:
            if block_manager.num_used_blocks(tier) <= target_used:
                break
            for target_tier in self._lower_tiers(tier):
                required = len(candidate.block_table)
                if not self._transfer_budget_allows(len(transfers), required):
                    continue
                nested = self._ensure_free_blocks(target_tier, required)
                transfers.extend(nested)
                if not block_manager.can_swap(candidate, target_tier):
                    continue
                mappings = block_manager.swap(candidate, target_tier)
                candidate.num_swaps += 1
                transfers.extend(mappings)
                self._record_transfers(mappings)
                break

        if block_manager.utilization(tier) > high:
            self.capacity_pressure[tier.value] += 1
        return transfers

    def _enforce_watermarks(self):
        transfers = []

        # Free lower tiers first so a later GPU demotion has somewhere to go.
        for tier in reversed(self.cache_tiers[1:-1]):
            transfers.extend(self._demote_swapped_tier(tier))

        gpu = CacheTier.GPU
        block_manager = self.block_manager
        capacity = block_manager.capacity(gpu)
        high, low = self.config.watermarks(gpu.value)
        if len(self.cache_tiers) > 1 and capacity and block_manager.utilization(gpu) > high:
            self.watermark_triggers[gpu.value] += 1
            target_used = floor(capacity * low)
            candidates = sorted(
                self.running,
                key=lambda seq: (seq.last_used_step, seq.seq_id),
            )
            for candidate in candidates:
                if block_manager.num_used_blocks(gpu) <= target_used:
                    break
                if not self._transfer_budget_allows(
                    len(transfers), len(candidate.block_table)
                ):
                    break
                self.running.remove(candidate)
                transfers.extend(self.preempt(candidate))
            if block_manager.utilization(gpu) > high:
                self.capacity_pressure[gpu.value] += 1

        terminal = self.cache_tiers[-1]
        if terminal != CacheTier.GPU:
            capacity = block_manager.capacity(terminal)
            high, _ = self.config.watermarks(terminal.value)
            if capacity and block_manager.utilization(terminal) > high:
                self.capacity_pressure[terminal.value] += 1
        return transfers

    def _schedule_swapped(self):
        scheduled_seqs = []
        blocks_to_swap_in = []
        total_gpu_blocks = self.block_manager.capacity(CacheTier.GPU)
        candidates = sorted(
            self.swapped,
            key=lambda seq: (seq.last_used_step, seq.seq_id),
        )

        for seq in candidates:
            if len(scheduled_seqs) >= self.max_num_seqs:
                break

            extra = int(len(seq) % self.block_size == 1)
            required = len(seq.block_table) + extra

            if required > total_gpu_blocks:
                raise RuntimeError(
                    f"Sequence {seq.seq_id} requires {required} GPU KV blocks, "
                    f"but only {total_gpu_blocks} blocks exist. "
                    "Whole-sequence swapping cannot serve a sequence larger "
                    "than the GPU cache."
                )

            if not self._transfer_budget_allows(
                len(blocks_to_swap_in), len(seq.block_table)
            ):
                break
            if not self.block_manager.can_swap_in(seq, extra):
                continue

            self.swapped.remove(seq)

            mappings = self.block_manager.swap_in(seq)
            seq.num_swaps += 1
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
        self.step_index += 1
        scheduled_seqs = []
        blocks_to_swap_in = []
        blocks_to_swap_out = self._enforce_watermarks()
        num_batched_tokens = 0

        if not self.running and self.swapped and not blocks_to_swap_out:
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
                    if seq.num_blocks > self.block_manager.capacity(CacheTier.GPU):
                        raise RuntimeError(
                            f"Sequence {seq.seq_id} requires {seq.num_blocks} GPU KV blocks, "
                            f"but only {self.block_manager.capacity(CacheTier.GPU)} blocks exist. "
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
                    victim = min(
                        self.running,
                        key=lambda item: (item.last_used_step, item.seq_id),
                    )
                    self.running.remove(victim)
                else:
                    required = len(seq.block_table) + 1
                    total = self.block_manager.capacity(CacheTier.GPU)
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
        self.num_preemptions += 1
        block_manager = self.block_manager
        num_blocks = len(seq.block_table)
        mappings = []

        swap_out = []
        for target_tier in self._lower_tiers(CacheTier.GPU):
            if num_blocks > block_manager.capacity(target_tier):
                continue
            demotions = self._ensure_free_blocks(target_tier, num_blocks)
            mappings.extend(demotions)
            if block_manager.can_swap_out(seq, target_tier):
                swap_out = block_manager.swap_out(seq, target_tier)
                seq.num_swaps += 1
                break

        if swap_out:
            mappings.extend(swap_out)
            self._record_transfers(swap_out)

            seq.status = SequenceStatus.SWAPPED
            self.swapped.append(seq)
            return mappings

        # Shared prefix blocks cannot be moved as a whole safely. Recomputing
        # preserves reference counts and correctness when no lower tier fits.
        self.num_recompute_preemptions += 1
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
            seq.last_used_step = self.step_index
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)

    def stats(self):
        tiers = {}
        for tier in self.cache_tiers:
            tiers[tier.value] = {
                "capacity_blocks": self.block_manager.capacity(tier),
                "used_blocks": self.block_manager.num_used_blocks(tier),
                "free_blocks": self.block_manager.num_free_blocks(tier),
                "utilization": self.block_manager.utilization(tier),
                "watermark_triggers": self.watermark_triggers[tier.value],
                "capacity_pressure": self.capacity_pressure[tier.value],
            }
        return {
            "tiers": tiers,
            "transfer_counts": dict(self.transfer_counts),
            "swap_in_blocks": self.num_swap_in_blocks,
            "swap_out_blocks": self.num_swap_out_blocks,
            "preemptions": self.num_preemptions,
            "recompute_preemptions": self.num_recompute_preemptions,
        }

    def reset_stats(self, clear_prefix_cache=False):
        if self.waiting or self.running or self.swapped:
            raise RuntimeError("Cannot reset cache statistics while requests are active")
        if clear_prefix_cache:
            self.block_manager.clear_prefix_cache()
        for key in self.transfer_counts:
            self.transfer_counts[key] = 0
        for tier in self.cache_tiers:
            self.watermark_triggers[tier.value] = 0
            self.capacity_pressure[tier.value] = 0
        self.num_swap_in_blocks = 0
        self.num_swap_out_blocks = 0
        self.num_preemptions = 0
        self.num_recompute_preemptions = 0
