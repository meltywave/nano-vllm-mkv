from collections import deque
from dataclasses import dataclass
import hashlib

from nanovllm.engine.sequence import CacheTier, Sequence


@dataclass(frozen=True, slots=True)
class BlockTransfer:
    src_tier: CacheTier
    src_block_id: int
    dst_tier: CacheTier
    dst_block_id: int


@dataclass(slots=True)
class SwapPlan:
    seq: Sequence
    source_tier: CacheTier
    target_tier: CacheTier
    source_table: list[int]
    target_table: list[int]
    transfers: list[BlockTransfer]


@dataclass(slots=True)
class BlockMovePlan:
    seq: Sequence
    index: int
    source_tier: CacheTier
    source_id: int
    target_tier: CacheTier
    target_id: int
    transfer: BlockTransfer


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []
        self.last_used_step = -1
        self.reuse_hits = 0

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []
        self.last_used_step = -1
        self.reuse_hits = 0


class BlockManager:

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_cpu_blocks: int = 0,
        num_ssd_blocks: int = 0,
        num_remote_blocks: int = 0,
        enable_prefix_cache: bool = True,
    ):
        self.block_size = block_size
        self.enable_prefix_cache = enable_prefix_cache
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        self.cpu_blocks = [Block(i) for i in range(num_cpu_blocks)]
        self.free_cpu_block_ids = deque(range(num_cpu_blocks))
        self.ssd_blocks = [Block(i) for i in range(num_ssd_blocks)]
        self.free_ssd_block_ids = deque(range(num_ssd_blocks))
        self.remote_blocks = [Block(i) for i in range(num_remote_blocks)]
        self.free_remote_block_ids = deque(range(num_remote_blocks))

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = hashlib.blake2b(digest_size=8)
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        for token_id in token_ids:
            h.update(int(token_id).to_bytes(8, "little", signed=False))
        return int.from_bytes(h.digest(), "little")

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        if not self.enable_prefix_cache:
            return 0 if len(self.free_block_ids) >= seq.num_blocks else -1
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        assert seq.block_table_tier == CacheTier.GPU
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            block.reuse_hits += 1
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
            seq.block_tiers.append(CacheTier.GPU)
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
            seq.block_tiers.append(CacheTier.GPU)
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        assert len(seq.block_tiers) == len(seq.block_table)
        for tier, block_id in reversed(list(zip(seq.block_tiers, seq.block_table))):
            if tier == CacheTier.GPU:
                block = self.blocks[block_id]
                block.ref_count -= 1
                if block.ref_count == 0:
                    self._deallocate_block(block_id)
            else:
                self._release_offload_block(tier, block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()
        seq.block_tiers.clear()
        seq.block_table_tier = CacheTier.GPU

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())
            seq.block_tiers.append(CacheTier.GPU)

    def touch(self, seq: Sequence, step: int):
        for tier, block_id in zip(seq.block_tiers, seq.block_table):
            self._blocks_for_tier(tier)[block_id].last_used_step = step

    def hash_blocks(self, seq: Sequence):
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            if self.enable_prefix_cache:
                self.hash_to_block_id[h] = block.block_id

    def clear_prefix_cache(self):
        if self.used_block_ids:
            raise RuntimeError("Cannot clear prefix cache while GPU blocks are in use")
        self.hash_to_block_id.clear()
        for block in self.blocks:
            block.hash = -1
            block.token_ids = []

    def _blocks_for_tier(self, tier: CacheTier) -> list[Block]:
        if tier == CacheTier.GPU:
            return self.blocks
        if tier == CacheTier.CPU:
            return self.cpu_blocks
        if tier == CacheTier.SSD:
            return self.ssd_blocks
        if tier == CacheTier.REMOTE:
            return self.remote_blocks
        raise ValueError(f"Unknown cache tier: {tier!r}")

    def _free_ids_for_tier(self, tier: CacheTier) -> deque[int]:
        if tier == CacheTier.GPU:
            return self.free_block_ids
        if tier == CacheTier.CPU:
            return self.free_cpu_block_ids
        if tier == CacheTier.SSD:
            return self.free_ssd_block_ids
        if tier == CacheTier.REMOTE:
            return self.free_remote_block_ids
        raise ValueError(f"Unknown cache tier: {tier!r}")

    def capacity(self, tier: CacheTier) -> int:
        return len(self._blocks_for_tier(tier))

    def num_free_blocks(self, tier: CacheTier) -> int:
        return len(self._free_ids_for_tier(tier))

    def num_used_blocks(self, tier: CacheTier) -> int:
        return self.capacity(tier) - self.num_free_blocks(tier)

    def utilization(self, tier: CacheTier) -> float:
        capacity = self.capacity(tier)
        return self.num_used_blocks(tier) / capacity if capacity else 0.0

    def is_exclusively_owned(self, seq: Sequence) -> bool:
        blocks = self._blocks_for_tier(seq.block_table_tier)
        return all(blocks[block_id].ref_count == 1 for block_id in seq.block_table)

    def can_swap(
        self,
        seq: Sequence,
        target_tier: CacheTier,
        extra_blocks: int = 0,
    ) -> bool:
        if seq.block_table_tier == target_tier:
            return False
        if not self.is_exclusively_owned(seq):
            return False
        required = len(seq.block_table) + extra_blocks
        return self.num_free_blocks(target_tier) >= required

    def can_swap_out(
        self,
        seq: Sequence,
        target_tier: CacheTier = CacheTier.CPU,
    ) -> bool:
        return self.can_swap(seq, target_tier)

    def can_swap_in(self, seq: Sequence, extra_gpu_blocks: int = 0) -> bool:
        return self.can_swap(seq, CacheTier.GPU, extra_gpu_blocks)

    def _allocate_offload_block(self, tier: CacheTier, src: Block) -> int:
        block_id = self._free_ids_for_tier(tier).popleft()
        block = self._blocks_for_tier(tier)[block_id]
        assert block.ref_count == 0
        block.ref_count = 1
        block.update(src.hash, list(src.token_ids))
        block.last_used_step = src.last_used_step
        block.reuse_hits = src.reuse_hits
        return block_id

    def _release_offload_block(self, tier: CacheTier, block_id: int):
        block = self._blocks_for_tier(tier)[block_id]
        assert block.ref_count == 1
        block.ref_count = 0
        block.hash = -1
        block.token_ids = []
        self._free_ids_for_tier(tier).append(block_id)

    def prepare_swap(
        self,
        seq: Sequence,
        target_tier: CacheTier,
    ) -> SwapPlan:
        source_tier = seq.block_table_tier
        if not self.can_swap(seq, target_tier):
            raise RuntimeError(
                f"Cannot swap {len(seq.block_table)} blocks from "
                f"{source_tier.value} to {target_tier.value}."
            )

        source_blocks = self._blocks_for_tier(source_tier)
        source_table = list(seq.block_table)
        target_table = []
        transfers = []
        try:
            for source_id in source_table:
                source_block = source_blocks[source_id]
                if target_tier == CacheTier.GPU:
                    target_id = self._allocate_block()
                    self.blocks[target_id].update(
                        source_block.hash, list(source_block.token_ids)
                    )
                    self.blocks[target_id].last_used_step = source_block.last_used_step
                    self.blocks[target_id].reuse_hits = source_block.reuse_hits
                else:
                    target_id = self._allocate_offload_block(
                        target_tier, source_block
                    )
                target_table.append(target_id)
                transfers.append(
                    BlockTransfer(source_tier, source_id, target_tier, target_id)
                )
        except Exception:
            self.abort_swap(
                SwapPlan(
                    seq, source_tier, target_tier, source_table, target_table, transfers
                )
            )
            raise
        return SwapPlan(
            seq, source_tier, target_tier, source_table, target_table, transfers
        )

    def abort_swap(self, plan: SwapPlan):
        if plan.target_tier == CacheTier.GPU:
            for target_id in reversed(plan.target_table):
                block = self.blocks[target_id]
                block.ref_count = 0
                self._deallocate_block(target_id)
                block.hash = -1
                block.token_ids = []
        else:
            for target_id in plan.target_table:
                self._release_offload_block(plan.target_tier, target_id)

    def commit_swap(self, plan: SwapPlan):
        seq = plan.seq
        if seq.block_table_tier != plan.source_tier or seq.block_table != plan.source_table:
            raise RuntimeError("Sequence changed while KV transfer was in progress")
        if plan.source_tier == CacheTier.GPU:
            for source_id in reversed(plan.source_table):
                block = self.blocks[source_id]
                block.ref_count -= 1
                if block.ref_count == 0:
                    self._deallocate_block(source_id)
        else:
            for source_id in plan.source_table:
                self._release_offload_block(plan.source_tier, source_id)

        seq.block_table = plan.target_table
        seq.block_table_tier = plan.target_tier
        seq.block_tiers = [plan.target_tier] * len(plan.target_table)
        if plan.target_tier == CacheTier.GPU:
            for target_id in plan.target_table:
                block = self.blocks[target_id]
                if self.enable_prefix_cache and block.hash != -1:
                    self.hash_to_block_id[block.hash] = target_id
        return plan.transfers

    def swap(
        self,
        seq: Sequence,
        target_tier: CacheTier,
    ) -> list[BlockTransfer]:
        """Metadata-only swap for scheduler tests; production uses prepare/commit."""
        return self.commit_swap(self.prepare_swap(seq, target_tier))

    def swap_out(
        self,
        seq: Sequence,
        target_tier: CacheTier = CacheTier.CPU,
    ) -> list[BlockTransfer]:
        assert seq.block_table_tier == CacheTier.GPU
        return self.swap(seq, target_tier)

    def swap_in(self, seq: Sequence) -> list[BlockTransfer]:
        assert seq.block_table_tier != CacheTier.GPU
        return self.swap(seq, CacheTier.GPU)

    def can_move_block(self, seq: Sequence, index: int, target_tier: CacheTier) -> bool:
        source_tier = seq.block_tiers[index]
        source_id = seq.block_table[index]
        return (
            source_tier != target_tier
            and self._blocks_for_tier(source_tier)[source_id].ref_count > 0
            and self.num_free_blocks(target_tier) > 0
        )

    def prepare_block_move(
        self, seq: Sequence, index: int, target_tier: CacheTier
    ) -> BlockMovePlan:
        if not self.can_move_block(seq, index, target_tier):
            raise RuntimeError(f"Cannot move logical block {index} to {target_tier.value}")
        source_tier = seq.block_tiers[index]
        source_id = seq.block_table[index]
        source = self._blocks_for_tier(source_tier)[source_id]
        if target_tier == CacheTier.GPU:
            target_id = self._allocate_block()
            target = self.blocks[target_id]
            target.update(source.hash, list(source.token_ids))
            target.last_used_step = source.last_used_step
            target.reuse_hits = source.reuse_hits
        else:
            target_id = self._allocate_offload_block(target_tier, source)
        return BlockMovePlan(
            seq, index, source_tier, source_id, target_tier, target_id,
            BlockTransfer(source_tier, source_id, target_tier, target_id),
        )

    def abort_block_move(self, plan: BlockMovePlan):
        if plan.target_tier == CacheTier.GPU:
            target = self.blocks[plan.target_id]
            target.ref_count = 0
            self._deallocate_block(plan.target_id)
            target.hash = -1
            target.token_ids = []
        else:
            self._release_offload_block(plan.target_tier, plan.target_id)

    def commit_block_move(self, plan: BlockMovePlan) -> BlockTransfer:
        seq = plan.seq
        if (seq.block_tiers[plan.index] != plan.source_tier
                or seq.block_table[plan.index] != plan.source_id):
            raise RuntimeError("Logical block changed during KV transfer")
        source = self._blocks_for_tier(plan.source_tier)[plan.source_id]
        if plan.source_tier == CacheTier.GPU:
            source.ref_count -= 1
            if source.ref_count == 0:
                if self.hash_to_block_id.get(source.hash) == plan.source_id:
                    del self.hash_to_block_id[source.hash]
                self._deallocate_block(plan.source_id)
                source.hash = -1
                source.token_ids = []
        else:
            self._release_offload_block(plan.source_tier, plan.source_id)
        seq.block_table[plan.index] = plan.target_id
        seq.block_tiers[plan.index] = plan.target_tier
        seq.block_table_tier = (
            plan.target_tier if all(tier == plan.target_tier for tier in seq.block_tiers)
            else None
        )
        if plan.target_tier == CacheTier.GPU:
            target = self.blocks[plan.target_id]
            if self.enable_prefix_cache and target.hash != -1:
                self.hash_to_block_id[target.hash] = plan.target_id
        return plan.transfer
