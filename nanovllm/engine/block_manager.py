from collections import deque
from dataclasses import dataclass
import xxhash
import numpy as np

from nanovllm.engine.sequence import CacheTier, Sequence


@dataclass(frozen=True, slots=True)
class BlockTransfer:
    src_tier: CacheTier
    src_block_id: int
    dst_tier: CacheTier
    dst_block_id: int


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_cpu_blocks: int = 0,
        num_ssd_blocks: int = 0,
        num_remote_blocks: int = 0,
    ):
        self.block_size = block_size
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
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

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
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        assert seq.block_table_tier == CacheTier.GPU
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

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
            self.hash_to_block_id[h] = block.block_id

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

    def can_swap(
        self,
        seq: Sequence,
        target_tier: CacheTier,
        extra_blocks: int = 0,
    ) -> bool:
        if seq.block_table_tier == target_tier:
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
        return block_id

    def _release_offload_block(self, tier: CacheTier, block_id: int):
        block = self._blocks_for_tier(tier)[block_id]
        assert block.ref_count == 1
        block.ref_count = 0
        block.hash = -1
        block.token_ids = []
        self._free_ids_for_tier(tier).append(block_id)

    def swap(
        self,
        seq: Sequence,
        target_tier: CacheTier,
    ) -> list[BlockTransfer]:
        source_tier = seq.block_table_tier
        if not self.can_swap(seq, target_tier):
            raise RuntimeError(
                f"Cannot swap {len(seq.block_table)} blocks from "
                f"{source_tier.value} to {target_tier.value}."
            )

        source_blocks = self._blocks_for_tier(source_tier)
        target_table = []
        transfers = []
        for source_id in seq.block_table:
            source_block = source_blocks[source_id]
            if target_tier == CacheTier.GPU:
                target_id = self._allocate_block()
                target_block = self.blocks[target_id]
                target_block.update(source_block.hash, list(source_block.token_ids))
                if target_block.hash != -1:
                    self.hash_to_block_id[target_block.hash] = target_id
            else:
                target_id = self._allocate_offload_block(
                    target_tier, source_block
                )
            target_table.append(target_id)
            transfers.append(
                BlockTransfer(source_tier, source_id, target_tier, target_id)
            )

        if source_tier == CacheTier.GPU:
            for source_id in reversed(seq.block_table):
                block = self.blocks[source_id]
                block.ref_count -= 1
                if block.ref_count == 0:
                    self._deallocate_block(source_id)
        else:
            for source_id in seq.block_table:
                self._release_offload_block(source_tier, source_id)

        seq.block_table = target_table
        seq.block_table_tier = target_tier
        return transfers

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
