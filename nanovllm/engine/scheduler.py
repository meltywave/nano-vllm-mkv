from collections import deque
from math import floor

from nanovllm.config import Config
from nanovllm.engine.sequence import CacheTier, Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config, transfer_blocks=None):
        self.config = config
        self.transfer_blocks = transfer_blocks if transfer_blocks is not None else (lambda transfers: None)
        self.cache_tiers = tuple(CacheTier(value) for value in config.kv_cache_tiers)
        self.block_hot = getattr(config, "kv_placement", "sequence") == "block-hot"
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
        self.step_transfer_count = 0
        self.watermark_triggers = {tier.value: 0 for tier in self.cache_tiers}
        self.capacity_pressure = {tier.value: 0 for tier in self.cache_tiers}
        self.watermark_demotions = {tier.value: 0 for tier in self.cache_tiers}
        self.peak_used_blocks = {tier.value: 0 for tier in self.cache_tiers}
        self.num_preemptions = 0
        self.num_recompute_preemptions = 0
        self.num_recompute_tokens = 0
        self.partial_swap_events = 0
        self.peak_mixed_sequences = 0
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
        if (
            self.max_transfer_blocks != -1
            and len(self.cache_tiers) > 1
            and seq.num_blocks > self.max_transfer_blocks
        ):
            raise ValueError(
                "kv_max_transfer_blocks is smaller than the request's initial KV blocks"
            )
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

    def _record_peak_occupancy(self):
        for tier in self.cache_tiers:
            self.peak_used_blocks[tier.value] = max(
                self.peak_used_blocks[tier.value],
                self.block_manager.num_used_blocks(tier),
            )

    def _lower_tiers(self, tier: CacheTier):
        index = self.cache_tiers.index(tier)
        return self.cache_tiers[index + 1:]

    def _move(self, seq: Sequence, target_tier: CacheTier):
        count = len(seq.block_table)
        if not self._transfer_budget_allows(count):
            return False
        plan = self.block_manager.prepare_swap(seq, target_tier)
        try:
            self.transfer_blocks(plan.transfers)
        except Exception:
            self.block_manager.abort_swap(plan)
            raise
        transfers = self.block_manager.commit_swap(plan)
        self.step_transfer_count += count
        seq.num_swaps += 1
        self._record_transfers(transfers)
        self._record_peak_occupancy()
        return True

    def _move_block(self, seq: Sequence, index: int, target_tier: CacheTier):
        if not self._transfer_budget_allows(1):
            return False
        plan = self.block_manager.prepare_block_move(seq, index, target_tier)
        try:
            self.transfer_blocks([plan.transfer])
        except Exception:
            self.block_manager.abort_block_move(plan)
            raise
        transfer = self.block_manager.commit_block_move(plan)
        self.step_transfer_count += 1
        seq.num_swaps += 1
        self._record_transfers([transfer])
        self._record_peak_occupancy()
        self._record_mixed_residency()
        return True

    def _restore_blocks(self, seq: Sequence):
        manager = self.block_manager
        missing = [
            index for index, tier in enumerate(seq.block_tiers)
            if tier != CacheTier.GPU
        ]
        plans = []
        try:
            for index in missing:
                plans.append(manager.prepare_block_move(seq, index, CacheTier.GPU))
            self.transfer_blocks([plan.transfer for plan in plans])
        except Exception:
            for plan in reversed(plans):
                manager.abort_block_move(plan)
            raise
        for plan in plans:
            self._record_transfers([manager.commit_block_move(plan)])
        self.step_transfer_count += len(plans)
        seq.num_swaps += bool(plans)
        self._record_peak_occupancy()
        self._record_mixed_residency()

    def _record_mixed_residency(self):
        if not self.block_hot:
            return
        mixed = sum(
            len(set(seq.block_tiers)) > 1
            for seq in (*self.waiting, *self.running, *self.swapped)
        )
        if mixed:
            self.partial_swap_events += 1
        self.peak_mixed_sequences = max(self.peak_mixed_sequences, mixed)

    def _cold_blocks(self, tier: CacheTier, exclude_seq: Sequence | None = None):
        candidates = []
        for seq in self.swapped:
            if seq is exclude_seq:
                continue
            for index, location in enumerate(seq.block_tiers):
                if location != tier:
                    continue
                block = self.block_manager._blocks_for_tier(tier)[seq.block_table[index]]
                if block.ref_count > 0:
                    # Readmission serves the oldest swapped request first. A
                    # newer request is expected to wait longer, while moving a
                    # shared GPU block does not free its physical slot.
                    candidates.append((block.ref_count > 1,
                                       -seq.last_used_step, block.reuse_hits,
                                       block.last_used_step, index, seq.seq_id,
                                       seq, index))
        candidates.sort(key=lambda item: item[:6])
        return [(item[-2], item[-1]) for item in candidates]

    def _demote_block(self, seq: Sequence, index: int, tier: CacheTier):
        for lower in self._lower_tiers(tier):
            if not self.block_manager.num_free_blocks(lower):
                self._ensure_free_blocks(lower, 1)
            if self.block_manager.can_move_block(seq, index, lower):
                return self._move_block(seq, index, lower)
        return False

    def _would_stall_on_budget(self):
        return self.max_transfer_blocks != -1 and any(
            (sum(tier != CacheTier.GPU for tier in seq.block_tiers)
             if self.block_hot else len(seq.block_table))
            > self.max_transfer_blocks for seq in self.swapped
        )

    def _blocked_by_gpu_capacity(self):
        if not self.swapped or self.running or self.waiting:
            return False
        total = self.block_manager.capacity(CacheTier.GPU)
        if self.block_hot:
            return all(
                self.block_manager.num_free_blocks(CacheTier.GPU)
                < sum(tier != CacheTier.GPU for tier in seq.block_tiers)
                + int(len(seq) % self.block_size == 1)
                for seq in self.swapped
            )
        return all(
            not self.block_manager.can_swap_in(
                seq, int(len(seq) % self.block_size == 1)
            )
            for seq in self.swapped
        ) and all(
            len(seq.block_table) + int(len(seq) % self.block_size == 1) <= total
            for seq in self.swapped
        )


    def _ensure_free_blocks(
        self, tier: CacheTier, required_blocks: int,
        exclude_seq: Sequence | None = None,
    ):
        block_manager = self.block_manager
        if block_manager.num_free_blocks(tier) >= required_blocks:
            return []

        if self.block_hot:
            for seq, index in self._cold_blocks(tier, exclude_seq):
                if block_manager.num_free_blocks(tier) >= required_blocks:
                    break
                self._demote_block(seq, index, tier)
            return []

        candidates = sorted(
            (seq for seq in self.swapped if seq.block_table_tier == tier),
            key=lambda seq: (seq.last_used_step, seq.seq_id),
        )
        for candidate in candidates:
            candidate_blocks = len(candidate.block_table)

            for target_tier in self._lower_tiers(tier):
                if candidate_blocks > block_manager.capacity(target_tier):
                    continue
                self._ensure_free_blocks(target_tier, candidate_blocks)
                if not block_manager.can_swap(candidate, target_tier):
                    continue
                if self._move(candidate, target_tier):
                    break

            if block_manager.num_free_blocks(tier) >= required_blocks:
                break
        return []

    def _transfer_budget_allows(self, additional: int):
        return (
            self.max_transfer_blocks == -1
            or self.step_transfer_count + additional <= self.max_transfer_blocks
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
        if self.block_hot:
            for seq, index in self._cold_blocks(tier):
                if block_manager.num_used_blocks(tier) <= target_used:
                    break
                if self._demote_block(seq, index, tier):
                    self.watermark_demotions[tier.value] += 1
            if block_manager.utilization(tier) > high:
                self.capacity_pressure[tier.value] += 1
            return []
        candidates = sorted(
            (seq for seq in self.swapped if seq.block_table_tier == tier),
            key=lambda seq: (seq.last_used_step, seq.seq_id),
        )
        for candidate in candidates:
            if block_manager.num_used_blocks(tier) <= target_used:
                break
            for target_tier in self._lower_tiers(tier):
                required = len(candidate.block_table)
                if not self._transfer_budget_allows(required):
                    continue
                self._ensure_free_blocks(target_tier, required)
                if not block_manager.can_swap(candidate, target_tier):
                    continue
                if self._move(candidate, target_tier):
                    self.watermark_demotions[tier.value] += 1
                    break

        if block_manager.utilization(tier) > high:
            self.capacity_pressure[tier.value] += 1
        return []

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
            if self.block_hot:
                while block_manager.num_used_blocks(gpu) > target_used:
                    moved = False
                    for seq, index in self._cold_blocks(gpu):
                        if self._demote_block(seq, index, gpu):
                            moved = True
                            break
                    if not moved and len(self.running) > 1:
                        candidate = min(self.running,
                                        key=lambda seq: (seq.last_used_step, seq.seq_id))
                        self.running.remove(candidate)
                        try:
                            self.preempt(candidate, allow_recompute=False)
                        except Exception:
                            self.running.appendleft(candidate)
                            raise
                        if candidate.status == SequenceStatus.RUNNING:
                            self.running.appendleft(candidate)
                        else:
                            moved = True
                    if not moved:
                        break
                    self.watermark_demotions[gpu.value] += 1
                if block_manager.utilization(gpu) > high:
                    self.capacity_pressure[gpu.value] += 1
            else:
                candidates = sorted(
                    self.running,
                    key=lambda seq: (seq.last_used_step, seq.seq_id),
                )
                for candidate in candidates:
                    if block_manager.num_used_blocks(gpu) <= target_used:
                        break
                    if len(self.running) <= 1:
                        break
                    if not self._transfer_budget_allows(len(candidate.block_table)):
                        continue
                    self.running.remove(candidate)
                    before = candidate.num_swaps
                    try:
                        self.preempt(candidate, allow_recompute=False)
                    except Exception:
                        self.running.appendleft(candidate)
                        raise
                    if candidate.status == SequenceStatus.RUNNING:
                        self.running.appendleft(candidate)
                    if candidate.num_swaps > before:
                        self.watermark_demotions[gpu.value] += 1
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
        num_batched_tokens = 0
        total_gpu_blocks = self.block_manager.capacity(CacheTier.GPU)
        candidates = sorted(
            self.swapped,
            key=lambda seq: (seq.last_used_step, seq.seq_id),
        )

        for seq in candidates:
            if (
                len(self.running) >= self.max_num_seqs
                or num_batched_tokens >= self.max_num_batched_tokens
            ):
                break

            extra = int(len(seq) % self.block_size == 1)
            missing = sum(tier != CacheTier.GPU for tier in seq.block_tiers)
            required = (missing if self.block_hot else len(seq.block_table)) + extra

            if len(seq.block_table) + extra > total_gpu_blocks:
                raise RuntimeError(
                    f"Sequence {seq.seq_id} requires {required} GPU KV blocks, "
                    f"but only {total_gpu_blocks} blocks exist. "
                    "Whole-sequence swapping cannot serve a sequence larger "
                    "than the GPU cache."
                )

            if (self.block_hot and not self.running
                    and self.block_manager.num_free_blocks(CacheTier.GPU) < required):
                self._ensure_free_blocks(CacheTier.GPU, required, exclude_seq=seq)

            if not self._transfer_budget_allows(missing if self.block_hot else len(seq.block_table)):
                break
            high, _ = self.config.watermarks(CacheTier.GPU.value)
            projected = self.block_manager.num_used_blocks(CacheTier.GPU) + required
            if self.running and projected / total_gpu_blocks > high:
                continue
            if (self.block_hot and self.block_manager.num_free_blocks(CacheTier.GPU) < required
                    or not self.block_hot and not self.block_manager.can_swap_in(seq, extra)):
                continue

            if self.block_hot:
                self._restore_blocks(seq)
                if any(tier != CacheTier.GPU for tier in seq.block_tiers):
                    raise RuntimeError("Request was readmitted with offloaded KV blocks")
            elif not self._move(seq, CacheTier.GPU):
                break
            self.swapped.remove(seq)

            seq.status = SequenceStatus.RUNNING
            seq.is_prefill = False
            seq.num_scheduled_tokens = 1
            self.block_manager.may_append(seq)
            self._record_peak_occupancy()

            self.running.append(seq)
            scheduled_seqs.append(seq)
            num_batched_tokens += 1

        return scheduled_seqs, blocks_to_swap_in

    def schedule(self):
        self.step_index += 1
        self.step_transfer_count = 0
        self._record_peak_occupancy()
        scheduled_seqs = []
        blocks_to_swap_in = []
        blocks_to_swap_out = self._enforce_watermarks()
        num_batched_tokens = 0

        if self.swapped and self.step_transfer_count == 0 and (
            not self.waiting or self.step_index % 2 == 0
        ):
            scheduled_seqs, blocks_to_swap_in = self._schedule_swapped()
            if scheduled_seqs:
                return scheduled_seqs, False, blocks_to_swap_in, blocks_to_swap_out
            if not self.waiting and self._would_stall_on_budget():
                raise RuntimeError(
                    "Swapped requests cannot make progress: check GPU capacity, "
                    "transfer budget, and lower-tier block availability."
                )
            if (not self.waiting and self.step_transfer_count == 0
                    and self._blocked_by_gpu_capacity()):
                raise RuntimeError(
                    "Swapped requests cannot fit in available GPU blocks; "
                    "check pinned GPU prefix blocks and cache capacities."
                )

        # prefill
        active_before_prefill = len(self.running)
        while self.waiting and active_before_prefill + len(scheduled_seqs) < self.max_num_seqs:
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
                self._record_peak_occupancy()
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
                try:
                    blocks_to_swap_out.extend(self.preempt(victim))
                except Exception:
                    self.running.appendleft(victim)
                    if victim is not seq:
                        self.running.appendleft(seq)
                    raise

                if victim is seq:
                    break
            if not can_schedule:
                continue

            seq.num_scheduled_tokens = 1
            seq.is_prefill = False
            self.block_manager.may_append(seq)
            self._record_peak_occupancy()
            scheduled_seqs.append(seq)

        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False, blocks_to_swap_in, blocks_to_swap_out

    def preempt(self, seq: Sequence, allow_recompute=True):
        block_manager = self.block_manager
        num_blocks = len(seq.block_table)
        mappings = []

        if self.block_hot:
            candidates = []
            for index, tier in enumerate(seq.block_tiers):
                if tier != CacheTier.GPU:
                    continue
                block = block_manager.blocks[seq.block_table[index]]
                if block.ref_count == 1:
                    candidates.append((block.reuse_hits, block.last_used_step, index))
            for _, _, index in sorted(candidates):
                if self._demote_block(seq, index, CacheTier.GPU):
                    self.num_preemptions += 1
                    seq.status = SequenceStatus.SWAPPED
                    self.swapped.append(seq)
                    self._record_mixed_residency()
                    return mappings

        swap_out = False
        if not self.block_hot:
            for target_tier in self._lower_tiers(CacheTier.GPU):
                if num_blocks > block_manager.capacity(target_tier):
                    continue
                demotions = self._ensure_free_blocks(target_tier, num_blocks)
                mappings.extend(demotions)
                if block_manager.can_swap_out(seq, target_tier):
                    if self._move(seq, target_tier):
                        swap_out = True
                        break

        if swap_out:
            self.num_preemptions += 1
            seq.status = SequenceStatus.SWAPPED
            self.swapped.append(seq)
            return mappings

        if not allow_recompute:
            return mappings

        # Shared prefix blocks cannot be moved as a whole safely. Recomputing
        # preserves reference counts and correctness when no lower tier fits.
        self.num_preemptions += 1
        self.num_recompute_preemptions += 1
        self.num_recompute_tokens += seq.num_cached_tokens
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)
        return mappings


    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            self.block_manager.touch(seq, self.step_index)
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
                "watermark_demotions": self.watermark_demotions[tier.value],
                "capacity_pressure": self.capacity_pressure[tier.value],
                "peak_used_blocks": self.peak_used_blocks[tier.value],
            }
        return {
            "tiers": tiers,
            "transfer_counts": dict(self.transfer_counts),
            "swap_in_blocks": self.num_swap_in_blocks,
            "swap_out_blocks": self.num_swap_out_blocks,
            "preemptions": self.num_preemptions,
            "recompute_preemptions": self.num_recompute_preemptions,
            "recompute_tokens": self.num_recompute_tokens,
            "partial_swap_events": self.partial_swap_events,
            "peak_mixed_sequences": self.peak_mixed_sequences,
        }

    def reset_stats(self, clear_prefix_cache=False):
        if self.waiting or self.running or self.swapped:
            raise RuntimeError("Cannot reset cache statistics while requests are active")
        if clear_prefix_cache:
            self.block_manager.clear_prefix_cache()
        self.step_index = 0
        self.step_transfer_count = 0
        for key in self.transfer_counts:
            self.transfer_counts[key] = 0
        for tier in self.cache_tiers:
            self.watermark_triggers[tier.value] = 0
            self.watermark_demotions[tier.value] = 0
            self.capacity_pressure[tier.value] = 0
            self.peak_used_blocks[tier.value] = self.block_manager.num_used_blocks(tier)
        self.num_swap_in_blocks = 0
        self.num_swap_out_blocks = 0
        self.num_preemptions = 0
        self.num_recompute_preemptions = 0
        self.num_recompute_tokens = 0
        self.partial_swap_events = 0
        self.peak_mixed_sequences = 0
