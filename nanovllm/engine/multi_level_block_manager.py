"""
三级 KV Cache 块管理器
支持 GPU + CPU + SSD 三级缓存，动态换入换出

层级结构：
    GPU HBM (最快, 最小)  ←→  CPU DRAM (中等)  ←→  SSD/NVMe (最慢, 最大)
       热数据                   温数据                冷数据
"""

import os
import time
from collections import deque
from enum import Enum, auto
from typing import Optional
import xxhash
import numpy as np
import torch

from nanovllm.engine.sequence import Sequence


class BlockLocation(Enum):
    """块位置枚举"""
    GPU = auto()
    CPU = auto()
    DISK = auto()
    NONE = auto()


class Block:
    """物理块数据结构"""

    def __init__(self, block_id: int, location: BlockLocation = BlockLocation.GPU):
        self.block_id = block_id
        self.location = location
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

        # 热度统计
        self.last_access_time = 0.0
        self.access_count = 0

        # 拥有该块的序列（用于热度统计）
        self.owner_seqs = set()

    def update(self, hash_val: int, token_ids: list[int]):
        self.hash = hash_val
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []
        self.last_access_time = 0.0
        self.access_count = 0
        self.owner_seqs.clear()

    def access(self, current_time: float):
        """记录访问"""
        self.last_access_time = current_time
        self.access_count += 1


class MultiLevelCacheStats:
    """多级缓存统计信息"""

    def __init__(self):
        self.gpu_total_blocks = 0
        self.gpu_used_blocks = 0
        self.cpu_total_blocks = 0
        self.cpu_used_blocks = 0
        self.disk_total_blocks = 0
        self.disk_used_blocks = 0

        self.total_swap_in = 0
        self.total_swap_out = 0
        self.total_swap_in_blocks = 0
        self.total_swap_out_blocks = 0

        # GPU ↔ CPU
        self.gpu_cpu_swap_in = 0
        self.gpu_cpu_swap_out = 0
        # CPU ↔ DISK
        self.cpu_disk_swap_in = 0
        self.cpu_disk_swap_out = 0

        self.prefix_cache_hits = 0
        self.prefix_cache_misses = 0

        # I/O 统计
        self.disk_read_bytes = 0
        self.disk_write_bytes = 0
        self.disk_read_time = 0.0
        self.disk_write_time = 0.0

    def gpu_utilization(self) -> float:
        if self.gpu_total_blocks == 0:
            return 0.0
        return self.gpu_used_blocks / self.gpu_total_blocks

    def cpu_utilization(self) -> float:
        if self.cpu_total_blocks == 0:
            return 0.0
        return self.cpu_used_blocks / self.cpu_total_blocks

    def disk_utilization(self) -> float:
        if self.disk_total_blocks == 0:
            return 0.0
        return self.disk_used_blocks / self.disk_total_blocks

    def prefix_cache_hit_rate(self) -> float:
        total = self.prefix_cache_hits + self.prefix_cache_misses
        if total == 0:
            return 0.0
        return self.prefix_cache_hits / total


class DiskBlockStore:
    """
    SSD 磁盘块存储

    使用文件系统存储 KV Cache 块：
    - 每个块保存为独立的 .pt 文件
    - 支持异步读写（可选）
    """

    def __init__(self, cache_dir: str, num_blocks: int, block_shape: tuple):
        """
        Args:
            cache_dir: 缓存目录路径
            num_blocks: 最大块数
            block_shape: 单个块的 shape (num_layers, block_size, num_kv_heads, head_dim)
                         注意：K 和 V 各一份，所以实际是 2 * ...
        """
        self.cache_dir = cache_dir
        self.num_blocks = num_blocks
        self.block_shape = block_shape  # (2, num_layers, block_size, num_kv_heads, head_dim)
        self.dtype = torch.float16  # 默认，实际由外部设置

        os.makedirs(cache_dir, exist_ok=True)

        # 空闲块 ID（逻辑 ID，从 0 开始）
        self.free_block_ids = deque(range(num_blocks))
        self.used_block_ids = set()

        # 块文件映射：逻辑块 ID → 文件名
        self._block_file = lambda bid: os.path.join(cache_dir, f"block_{bid:08d}.pt")

    def allocate(self) -> int:
        """分配一个磁盘块，返回逻辑块 ID"""
        if not self.free_block_ids:
            raise RuntimeError("Out of disk cache space")
        block_id = self.free_block_ids.popleft()
        self.used_block_ids.add(block_id)
        return block_id

    def deallocate(self, block_id: int):
        """释放磁盘块"""
        self.used_block_ids.discard(block_id)
        self.free_block_ids.append(block_id)
        # 删除文件
        fpath = self._block_file(block_id)
        if os.path.exists(fpath):
            try:
                os.remove(fpath)
            except OSError:
                pass

    def write_block(self, block_id: int, kv_data: torch.Tensor) -> int:
        """
        写入块到磁盘

        Args:
            block_id: 逻辑块 ID
            kv_data: shape (2, num_layers, block_size, num_kv_heads, head_dim)

        Returns:
            写入的字节数
        """
        fpath = self._block_file(block_id)
        # 移到 CPU 再保存
        kv_cpu = kv_data.cpu()
        torch.save(kv_cpu, fpath)
        return os.path.getsize(fpath)

    def read_block(self, block_id: int) -> torch.Tensor:
        """
        从磁盘读取块

        Returns:
            kv_data: shape (2, num_layers, block_size, num_kv_heads, head_dim)
        """
        fpath = self._block_file(block_id)
        kv_data = torch.load(fpath, map_location="cpu", weights_only=True)
        return kv_data

    def get_used_count(self) -> int:
        return len(self.used_block_ids)

    def get_free_count(self) -> int:
        return len(self.free_block_ids)

    def clear(self):
        """清空所有磁盘块"""
        for block_id in list(self.used_block_ids):
            self.deallocate(block_id)
        # 清理残留文件
        for fname in os.listdir(self.cache_dir):
            if fname.startswith("block_") and fname.endswith(".pt"):
                try:
                    os.remove(os.path.join(self.cache_dir, fname))
                except OSError:
                    pass


class MultiLevelBlockManager:
    """
    三级 KV Cache 块管理器

    支持 GPU + CPU + SSD 三级缓存：
    - GPU 层：高速，小容量，存放热数据
    - CPU 层：中速，中等容量，存放温数据
    - DISK 层：低速，大容量，存放冷数据
    - 动态换入换出：基于热度的调度策略（瀑布式）
    - 前缀缓存：三级都支持前缀缓存
    """

    def __init__(
        self,
        gpu_num_blocks: int,
        cpu_num_blocks: int,
        block_size: int,
        disk_num_blocks: int = 0,
        disk_cache_dir: str = "./kv_disk_cache",
        replacement_policy: str = "lru",
        swap_watermark_high: float = 0.9,
        swap_watermark_low: float = 0.7,
        enable_prefix_caching: bool = True,
    ):
        self.block_size = block_size
        self.gpu_num_blocks = gpu_num_blocks
        self.cpu_num_blocks = cpu_num_blocks
        self.disk_num_blocks = disk_num_blocks
        self.enable_disk = disk_num_blocks > 0
        self.enable_prefix_caching = enable_prefix_caching

        # 水位线
        self.swap_watermark_high = swap_watermark_high
        self.swap_watermark_low = swap_watermark_low

        # 替换策略
        self.replacement_policy = replacement_policy

        # GPU 块池（block_id: 0 ~ gpu_num_blocks-1）
        self.gpu_blocks: list[Block] = [
            Block(i, BlockLocation.GPU) for i in range(gpu_num_blocks)
        ]
        self.gpu_free_block_ids: deque[int] = deque(range(gpu_num_blocks))
        self.gpu_used_block_ids: set[int] = set()

        # CPU 块池（block_id: gpu_num_blocks ~ gpu_num_blocks + cpu_num_blocks - 1）
        self.cpu_blocks: list[Block] = [
            Block(i + gpu_num_blocks, BlockLocation.CPU) for i in range(cpu_num_blocks)
        ]
        self.cpu_free_block_ids: deque[int] = deque(
            range(gpu_num_blocks, gpu_num_blocks + cpu_num_blocks)
        )
        self.cpu_used_block_ids: set[int] = set()

        # DISK 块池（逻辑 ID，从 0 开始；全局 block_id 用偏移区分）
        self.disk_base_id = gpu_num_blocks + cpu_num_blocks
        self.disk_store: Optional[DiskBlockStore] = None
        self.disk_blocks: dict[int, Block] = {}  # 全局 block_id → Block
        if self.enable_disk:
            # DiskBlockStore 在外部初始化时设置（需要知道 shape）
            pass

        # 前缀缓存哈希表（三级分别维护）
        self.gpu_hash_to_block_id: dict[int, int] = dict()
        self.cpu_hash_to_block_id: dict[int, int] = dict()
        self.disk_hash_to_block_id: dict[int, int] = dict()

        # 统计信息
        self.stats = MultiLevelCacheStats()
        self.stats.gpu_total_blocks = gpu_num_blocks
        self.stats.cpu_total_blocks = cpu_num_blocks
        self.stats.disk_total_blocks = disk_num_blocks

        # 时间计数器（用于 LRU）
        self._time_counter = 0.0

        # 序列映射（seq_id -> Sequence），用于换出时更新序列的 block_table
        self.seq_map: dict[int, Sequence] = {}

    def init_disk_store(self, cache_dir: str, block_shape: tuple, dtype):
        """初始化磁盘存储（需要知道块的 shape 和 dtype）"""
        if not self.enable_disk:
            return
        self.disk_store = DiskBlockStore(
            cache_dir=cache_dir,
            num_blocks=self.disk_num_blocks,
            block_shape=block_shape,
        )
        self.disk_store.dtype = dtype

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1) -> int:
        """计算块的内容哈希（用于前缀缓存）"""
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _tick(self) -> float:
        """时间前进，用于 LRU 统计"""
        self._time_counter += 1.0
        return self._time_counter

    def _get_block(self, block_id: int) -> Block:
        """根据块 ID 获取块对象"""
        if block_id < self.gpu_num_blocks:
            return self.gpu_blocks[block_id]
        elif block_id < self.disk_base_id:
            return self.cpu_blocks[block_id - self.gpu_num_blocks]
        else:
            return self.disk_blocks[block_id]

    # ==================== GPU 块分配/释放 ====================

    def _allocate_gpu_block(self) -> int:
        """分配一个 GPU 块（不足时触发瀑布式换出）"""
        if not self.gpu_free_block_ids:
            # GPU 不足，触发换出到 CPU
            self._trigger_gpu_swap_out_if_needed()

        if not self.gpu_free_block_ids:
            raise RuntimeError("Out of GPU memory")

        block_id = self.gpu_free_block_ids.popleft()
        block = self.gpu_blocks[block_id]
        assert block.ref_count == 0

        # 从前缀缓存中移除
        if block.hash != -1 and self.gpu_hash_to_block_id.get(block.hash) == block_id:
            del self.gpu_hash_to_block_id[block.hash]

        block.reset()
        block.location = BlockLocation.GPU
        self.gpu_used_block_ids.add(block_id)
        self.stats.gpu_used_blocks += 1
        return block_id

    def _deallocate_gpu_block(self, block_id: int):
        """释放一个 GPU 块"""
        block = self.gpu_blocks[block_id]
        assert block.ref_count == 0
        assert block.location == BlockLocation.GPU

        # 注意：释放时不从前缀缓存中删除
        # 因为块的内容还在，可以被后续请求复用（从 free 列表移回 used 列表）
        # 只有在分配块时（内容会被覆盖）才会从前缀缓存中删除

        self.gpu_used_block_ids.remove(block_id)
        self.gpu_free_block_ids.append(block_id)
        self.stats.gpu_used_blocks -= 1

    # ==================== CPU 块分配/释放 ====================

    def _allocate_cpu_block(self) -> int:
        """分配一个 CPU 块"""
        if not self.cpu_free_block_ids:
            # CPU 不足，触发换出到 DISK
            if self.enable_disk:
                self._trigger_cpu_swap_out_if_needed()

        if not self.cpu_free_block_ids:
            raise RuntimeError("Out of CPU memory")

        block_id = self.cpu_free_block_ids.popleft()
        block = self.cpu_blocks[block_id - self.gpu_num_blocks]
        assert block.ref_count == 0

        # 从前缀缓存中移除
        if block.hash != -1 and self.cpu_hash_to_block_id.get(block.hash) == block_id:
            del self.cpu_hash_to_block_id[block.hash]

        block.reset()
        block.location = BlockLocation.CPU
        self.cpu_used_block_ids.add(block_id)
        self.stats.cpu_used_blocks += 1
        return block_id

    def _deallocate_cpu_block(self, block_id: int):
        """释放一个 CPU 块"""
        block = self.cpu_blocks[block_id - self.gpu_num_blocks]
        assert block.ref_count == 0
        assert block.location == BlockLocation.CPU

        # 注意：释放时不从前缀缓存中删除
        # 因为块的内容还在，可以被后续请求复用（从 free 列表移回 used 列表）
        # 只有在分配块时（内容会被覆盖）才会从前缀缓存中删除

        self.cpu_used_block_ids.remove(block_id)
        self.cpu_free_block_ids.append(block_id)
        self.stats.cpu_used_blocks -= 1

    # ==================== DISK 块分配/释放 ====================

    def _allocate_disk_block(self) -> int:
        """分配一个 DISK 块，返回全局 block_id"""
        if not self.enable_disk or self.disk_store is None:
            raise RuntimeError("Disk cache not enabled")

        logical_id = self.disk_store.allocate()
        global_id = self.disk_base_id + logical_id

        block = Block(global_id, BlockLocation.DISK)
        self.disk_blocks[global_id] = block
        self.stats.disk_used_blocks += 1
        return global_id

    def _deallocate_disk_block(self, block_id: int):
        """释放一个 DISK 块"""
        if self.disk_store is None:
            return

        # 注意：释放时不从前缀缓存中删除
        # 因为块的内容还在，可以被后续请求复用
        # 只有在分配块时（内容会被覆盖）才会从前缀缓存中删除

        logical_id = block_id - self.disk_base_id
        self.disk_store.deallocate(logical_id)
        del self.disk_blocks[block_id]
        self.stats.disk_used_blocks -= 1

    # ==================== 兼容原接口的方法 ====================

    def can_allocate(self, seq: Sequence) -> int:
        """
        检查是否能为序列分配块，返回可复用的前缀缓存块数
        """
        if not self.enable_prefix_caching:
            total_free = (
                len(self.gpu_free_block_ids)
                + len(self.cpu_free_block_ids)
                + (self.disk_store.get_free_count() if self.enable_disk and self.disk_store else 0)
            )
            if total_free < seq.num_blocks:
                return -1
            return 0

        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks

        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)

            # 先查 GPU
            block_id = self.gpu_hash_to_block_id.get(h, -1)
            if block_id != -1:
                block = self.gpu_blocks[block_id]
                if block.token_ids == token_ids:
                    num_cached_blocks += 1
                    num_new_blocks -= 1  # 所有命中的块都可以复用，不管是 used 还是 free
                    continue

            # 再查 CPU
            block_id = self.cpu_hash_to_block_id.get(h, -1)
            if block_id != -1:
                block = self.cpu_blocks[block_id - self.gpu_num_blocks]
                if block.token_ids == token_ids:
                    num_cached_blocks += 1
                    num_new_blocks -= 1  # 所有命中的块都可以复用，不管是 used 还是 free
                    continue

            # 再查 DISK
            if self.enable_disk:
                block_id = self.disk_hash_to_block_id.get(h, -1)
                if block_id != -1 and block_id in self.disk_blocks:
                    block = self.disk_blocks[block_id]
                    if block.token_ids == token_ids:
                        num_cached_blocks += 1
                        num_new_blocks -= 1
                        continue

            # 都没命中，停止
            break

        # 检查总空间
        total_free = (
            len(self.gpu_free_block_ids)
            + len(self.cpu_free_block_ids)
            + (self.disk_store.get_free_count() if self.enable_disk and self.disk_store else 0)
        )
        if total_free < num_new_blocks:
            return -1

        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        """
        为序列分配块（优先 GPU，其次 CPU，最后 DISK）
        """
        assert not seq.block_table
        current_time = self._tick()

        # 注册序列到 seq_map
        self.seq_map[seq.seq_id] = seq

        if self.enable_prefix_caching and num_cached_blocks > 0:
            h = -1
            for i in range(num_cached_blocks):
                token_ids = seq.block(i)
                h = self.compute_hash(token_ids, h)

                # 先查 GPU
                block_id = self.gpu_hash_to_block_id.get(h, -1)
                if block_id != -1:
                    block = self.gpu_blocks[block_id]
                    if block_id in self.gpu_used_block_ids:
                        block.ref_count += 1
                    else:
                        block.ref_count = 1
                        self.gpu_free_block_ids.remove(block_id)
                        self.gpu_used_block_ids.add(block_id)
                        self.stats.gpu_used_blocks += 1
                    block.access(current_time)
                    block.owner_seqs.add(seq.seq_id)
                    seq.block_table.append(block_id)
                    seq.block_locations.append(BlockLocation.GPU)
                    self.stats.prefix_cache_hits += 1
                    continue

                # 再查 CPU
                block_id = self.cpu_hash_to_block_id.get(h, -1)
                if block_id != -1:
                    block = self.cpu_blocks[block_id - self.gpu_num_blocks]
                    if block_id in self.cpu_used_block_ids:
                        block.ref_count += 1
                    else:
                        block.ref_count = 1
                        self.cpu_free_block_ids.remove(block_id)
                        self.cpu_used_block_ids.add(block_id)
                        self.stats.cpu_used_blocks += 1
                    block.access(current_time)
                    block.owner_seqs.add(seq.seq_id)
                    seq.block_table.append(block_id)
                    seq.block_locations.append(BlockLocation.CPU)
                    self.stats.prefix_cache_hits += 1
                    continue

                # 再查 DISK
                if self.enable_disk:
                    block_id = self.disk_hash_to_block_id.get(h, -1)
                    if block_id != -1 and block_id in self.disk_blocks:
                        block = self.disk_blocks[block_id]
                        block.ref_count += 1
                        block.access(current_time)
                        block.owner_seqs.add(seq.seq_id)
                        seq.block_table.append(block_id)
                        seq.block_locations.append(BlockLocation.DISK)
                        self.stats.prefix_cache_hits += 1
                        continue

                self.stats.prefix_cache_misses += 1

        # 分配新块（优先 GPU → CPU → DISK）
        num_new_blocks = seq.num_blocks - num_cached_blocks
        for _ in range(num_new_blocks):
            try:
                block_id = self._allocate_gpu_block()
                seq.block_locations.append(BlockLocation.GPU)
            except RuntimeError:
                try:
                    block_id = self._allocate_cpu_block()
                    seq.block_locations.append(BlockLocation.CPU)
                except RuntimeError:
                    if self.enable_disk:
                        block_id = self._allocate_disk_block()
                        seq.block_locations.append(BlockLocation.DISK)
                    else:
                        raise RuntimeError("Out of memory at all levels")

            block = self._get_block(block_id)
            block.access(current_time)
            block.owner_seqs.add(seq.seq_id)
            seq.block_table.append(block_id)

        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        """释放序列的所有块"""
        for block_id in reversed(seq.block_table):
            block = self._get_block(block_id)
            block.ref_count -= 1
            block.owner_seqs.discard(seq.seq_id)

            if block.ref_count == 0:
                if block.location == BlockLocation.GPU:
                    self._deallocate_gpu_block(block_id)
                elif block.location == BlockLocation.CPU:
                    self._deallocate_cpu_block(block_id)
                elif block.location == BlockLocation.DISK:
                    self._deallocate_disk_block(block_id)

        seq.num_cached_tokens = 0
        seq.block_table.clear()
        seq.block_locations.clear()

        # 从 seq_map 中移除
        if seq.seq_id in self.seq_map:
            del self.seq_map[seq.seq_id]

    def can_append(self, seq: Sequence) -> bool:
        """检查是否可以追加新块"""
        if len(seq) % self.block_size != 1:
            return True

        total_free = (
            len(self.gpu_free_block_ids)
            + len(self.cpu_free_block_ids)
            + (self.disk_store.get_free_count() if self.enable_disk and self.disk_store else 0)
        )
        return total_free >= 1

    def may_append(self, seq: Sequence):
        """如果需要，追加一个新块"""
        if len(seq) % self.block_size == 1:
            current_time = self._tick()
            try:
                block_id = self._allocate_gpu_block()
                seq.block_locations.append(BlockLocation.GPU)
            except RuntimeError:
                try:
                    block_id = self._allocate_cpu_block()
                    seq.block_locations.append(BlockLocation.CPU)
                except RuntimeError:
                    if self.enable_disk:
                        block_id = self._allocate_disk_block()
                        seq.block_locations.append(BlockLocation.DISK)
                    else:
                        raise

            block = self._get_block(block_id)
            block.access(current_time)
            block.owner_seqs.add(seq.seq_id)
            seq.block_table.append(block_id)

    def hash_blocks(self, seq: Sequence):
        """计算并更新块的哈希值（用于前缀缓存）"""
        if not self.enable_prefix_caching:
            return

        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end:
            return

        current_time = self._tick()

        # 找到起始的前缀哈希
        if start > 0:
            start_block_id = seq.block_table[start - 1]
            start_block = self._get_block(start_block_id)
            h = start_block.hash
        else:
            h = -1

        for i in range(start, end):
            block_id = seq.block_table[i]
            block = self._get_block(block_id)
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            block.access(current_time)

            # 加入对应层级的前缀缓存
            if block.location == BlockLocation.GPU:
                self.gpu_hash_to_block_id[h] = block.block_id
            elif block.location == BlockLocation.CPU:
                self.cpu_hash_to_block_id[h] = block.block_id
            elif block.location == BlockLocation.DISK:
                self.disk_hash_to_block_id[h] = block.block_id

    # ==================== 换出触发逻辑 ====================

    def _trigger_gpu_swap_out_if_needed(self):
        """GPU 超过高水位线时，换出到 CPU"""
        gpu_util = self.stats.gpu_utilization()
        if gpu_util < self.swap_watermark_high:
            return

        target_blocks = int(self.gpu_num_blocks * self.swap_watermark_low)
        blocks_to_swap = self.stats.gpu_used_blocks - target_blocks

        if blocks_to_swap <= 0:
            return

        self._swap_out_gpu_to_cpu(blocks_to_swap)

    def _trigger_cpu_swap_out_if_needed(self):
        """CPU 超过高水位线时，换出到 DISK"""
        if not self.enable_disk:
            return

        cpu_util = self.stats.cpu_utilization()
        if cpu_util < self.swap_watermark_high:
            return

        target_blocks = int(self.cpu_num_blocks * self.swap_watermark_low)
        blocks_to_swap = self.stats.cpu_used_blocks - target_blocks

        if blocks_to_swap <= 0:
            return

        self._swap_out_cpu_to_disk(blocks_to_swap)

    # ==================== GPU → CPU 换出 ====================

    def _swap_out_gpu_to_cpu(self, num_blocks: int):
        """
        将 GPU 中的冷块换出到 CPU

        选择策略：
        1. 只换出引用计数为 1 的块（共享块不换出）
        2. 按 LRU 选择最久未访问的
        """
        candidates = []
        for block_id in self.gpu_used_block_ids:
            block = self.gpu_blocks[block_id]
            if block.ref_count == 1:
                score = block.last_access_time  # LRU
                candidates.append((score, block_id))

        if not candidates:
            return

        candidates.sort()  # 升序，最久的先换出

        swapped = 0
        for score, gpu_block_id in candidates:
            if swapped >= num_blocks:
                break

            if not self.cpu_free_block_ids:
                # CPU 也满了，尝试换出到磁盘腾出空间
                if self.enable_disk:
                    self._swap_out_cpu_to_disk(max(1, num_blocks - swapped))
                    if not self.cpu_free_block_ids:
                        break
                else:
                    break

            gpu_block = self.gpu_blocks[gpu_block_id]

            # 分配 CPU 块
            try:
                cpu_block_id = self._allocate_cpu_block()
            except RuntimeError:
                break

            cpu_block = self.cpu_blocks[cpu_block_id - self.gpu_num_blocks]

            # 复制块元数据（实际数据拷贝由 ModelRunner 完成）
            cpu_block.hash = gpu_block.hash
            cpu_block.token_ids = gpu_block.token_ids.copy()
            cpu_block.ref_count = gpu_block.ref_count
            cpu_block.last_access_time = gpu_block.last_access_time
            cpu_block.access_count = gpu_block.access_count
            cpu_block.owner_seqs = gpu_block.owner_seqs.copy()

            # 更新前缀缓存
            if gpu_block.hash != -1:
                if self.gpu_hash_to_block_id.get(gpu_block.hash) == gpu_block_id:
                    del self.gpu_hash_to_block_id[gpu_block.hash]
                self.cpu_hash_to_block_id[gpu_block.hash] = cpu_block_id

            # 更新所有引用该块的序列
            for seq_id in gpu_block.owner_seqs:
                if seq_id in self.seq_map:
                    seq = self.seq_map[seq_id]
                    for i, bid in enumerate(seq.block_table):
                        if bid == gpu_block_id:
                            seq.block_table[i] = cpu_block_id
                            seq.block_locations[i] = BlockLocation.CPU
                            break

            # 释放 GPU 块
            gpu_block.ref_count = 0
            self._deallocate_gpu_block(gpu_block_id)

            swapped += 1
            self.stats.total_swap_out += 1
            self.stats.total_swap_out_blocks += 1
            self.stats.gpu_cpu_swap_out += 1

    # ==================== CPU → DISK 换出 ====================

    def _swap_out_cpu_to_disk(self, num_blocks: int):
        """
        将 CPU 中的冷块换出到 DISK

        注意：这只是元数据层面的操作，实际数据写入由 ModelRunner 协调
        """
        if not self.enable_disk or self.disk_store is None:
            return

        candidates = []
        for block_id in self.cpu_used_block_ids:
            block = self.cpu_blocks[block_id - self.gpu_num_blocks]
            if block.ref_count == 1:
                score = block.last_access_time
                candidates.append((score, block_id))

        if not candidates:
            return

        candidates.sort()

        swapped = 0
        for score, cpu_block_id in candidates:
            if swapped >= num_blocks:
                break

            # 分配磁盘块
            try:
                disk_block_id = self._allocate_disk_block()
            except RuntimeError:
                break

            cpu_block = self.cpu_blocks[cpu_block_id - self.gpu_num_blocks]
            disk_block = self.disk_blocks[disk_block_id]

            # 复制元数据
            disk_block.hash = cpu_block.hash
            disk_block.token_ids = cpu_block.token_ids.copy()
            disk_block.ref_count = cpu_block.ref_count
            disk_block.last_access_time = cpu_block.last_access_time
            disk_block.access_count = cpu_block.access_count
            disk_block.owner_seqs = cpu_block.owner_seqs.copy()

            # 更新前缀缓存
            if cpu_block.hash != -1:
                if self.cpu_hash_to_block_id.get(cpu_block.hash) == cpu_block_id:
                    del self.cpu_hash_to_block_id[cpu_block.hash]
                self.disk_hash_to_block_id[cpu_block.hash] = disk_block_id

            # 更新所有引用该块的序列
            for seq_id in cpu_block.owner_seqs:
                if seq_id in self.seq_map:
                    seq = self.seq_map[seq_id]
                    for i, bid in enumerate(seq.block_table):
                        if bid == cpu_block_id:
                            seq.block_table[i] = disk_block_id
                            seq.block_locations[i] = BlockLocation.DISK
                            break

            # 释放 CPU 块
            cpu_block.ref_count = 0
            self._deallocate_cpu_block(cpu_block_id)

            swapped += 1
            self.stats.total_swap_out += 1
            self.stats.total_swap_out_blocks += 1
            self.stats.cpu_disk_swap_out += 1

    # ==================== 序列级换出 ====================

    def swap_out_sequence_to_cpu(self, seq: "Sequence") -> tuple[list[int], list[int]]:
        """
        将序列的所有 GPU 块换出到 CPU（序列级换出）

        Args:
            seq: 要换出的序列

        Returns:
            (gpu_block_ids, cpu_block_ids): 需要传输的块 ID 列表，供 ModelRunner 执行数据传输
        """
        gpu_block_ids = []
        cpu_block_ids = []

        for i, block_id in enumerate(seq.block_table):
            if seq.block_locations[i] != BlockLocation.GPU:
                continue

            gpu_block = self.gpu_blocks[block_id]

            # 共享块不换出（ref_count > 1）
            if gpu_block.ref_count > 1:
                continue

            # 确保 CPU 有空间
            if not self.cpu_free_block_ids:
                if self.enable_disk:
                    self._swap_out_cpu_to_disk(1)
                    if not self.cpu_free_block_ids:
                        break
                else:
                    break

            # 分配 CPU 块
            try:
                cpu_block_id = self._allocate_cpu_block()
            except RuntimeError:
                break

            cpu_block = self.cpu_blocks[cpu_block_id - self.gpu_num_blocks]

            # 复制块元数据
            cpu_block.hash = gpu_block.hash
            cpu_block.token_ids = gpu_block.token_ids.copy()
            cpu_block.ref_count = gpu_block.ref_count
            cpu_block.last_access_time = gpu_block.last_access_time
            cpu_block.access_count = gpu_block.access_count
            cpu_block.owner_seqs = gpu_block.owner_seqs.copy()

            # 更新前缀缓存
            if gpu_block.hash != -1:
                if self.gpu_hash_to_block_id.get(gpu_block.hash) == block_id:
                    del self.gpu_hash_to_block_id[gpu_block.hash]
                self.cpu_hash_to_block_id[gpu_block.hash] = cpu_block_id

            # 更新所有引用该块的序列
            for seq_id in gpu_block.owner_seqs:
                if seq_id in self.seq_map:
                    s = self.seq_map[seq_id]
                    for j, bid in enumerate(s.block_table):
                        if bid == block_id:
                            s.block_table[j] = cpu_block_id
                            s.block_locations[j] = BlockLocation.CPU
                            break

            # 释放 GPU 块
            gpu_block.ref_count = 0
            self._deallocate_gpu_block(block_id)

            gpu_block_ids.append(block_id)
            cpu_block_ids.append(cpu_block_id)

            self.stats.total_swap_out += 1
            self.stats.total_swap_out_blocks += 1
            self.stats.gpu_cpu_swap_out += 1

        return gpu_block_ids, cpu_block_ids

    def swap_out_sequence_to_disk(self, seq: "Sequence") -> tuple[list[int], list[int]]:
        """
        将序列的所有 CPU 块换出到 DISK（序列级换出）

        Args:
            seq: 要换出的序列

        Returns:
            (cpu_block_ids, disk_logical_ids): 需要传输的块 ID 列表，供 ModelRunner 执行数据传输
        """
        if not self.enable_disk:
            return [], []

        cpu_block_ids = []
        disk_logical_ids = []

        for i, block_id in enumerate(seq.block_table):
            if seq.block_locations[i] != BlockLocation.CPU:
                continue

            cpu_block = self.cpu_blocks[block_id - self.gpu_num_blocks]

            # 共享块不换出
            if cpu_block.ref_count > 1:
                continue

            # 分配 DISK 块
            try:
                disk_block_id = self._allocate_disk_block()
            except RuntimeError:
                break

            disk_logical_id = disk_block_id - self.disk_base_id

            # 复制块元数据到 DISK 块
            disk_block = self.disk_blocks[disk_logical_id]
            disk_block.hash = cpu_block.hash
            disk_block.token_ids = cpu_block.token_ids.copy()
            disk_block.ref_count = cpu_block.ref_count
            disk_block.last_access_time = cpu_block.last_access_time
            disk_block.access_count = cpu_block.access_count
            disk_block.owner_seqs = cpu_block.owner_seqs.copy()

            # 更新前缀缓存
            if cpu_block.hash != -1:
                if self.cpu_hash_to_block_id.get(cpu_block.hash) == block_id:
                    del self.cpu_hash_to_block_id[cpu_block.hash]
                self.disk_hash_to_block_id[cpu_block.hash] = disk_block_id

            # 更新所有引用该块的序列
            for seq_id in cpu_block.owner_seqs:
                if seq_id in self.seq_map:
                    s = self.seq_map[seq_id]
                    for j, bid in enumerate(s.block_table):
                        if bid == block_id:
                            s.block_table[j] = disk_block_id
                            s.block_locations[j] = BlockLocation.DISK
                            break

            # 释放 CPU 块
            cpu_block.ref_count = 0
            self._deallocate_cpu_block(block_id)

            cpu_block_ids.append(block_id)
            disk_logical_ids.append(disk_logical_id)

            self.stats.total_swap_out += 1
            self.stats.total_swap_out_blocks += 1
            self.stats.cpu_disk_swap_out += 1

        return cpu_block_ids, disk_logical_ids

    # ==================== DISK → CPU 换入 ====================

    def swap_in_disk_to_cpu(self, disk_block_id: int) -> int:
        """
        将 DISK 中的块换入到 CPU
        返回新的 CPU 块 ID
        """
        assert disk_block_id >= self.disk_base_id
        assert self.enable_disk and self.disk_store is not None

        disk_block = self.disk_blocks[disk_block_id]
        assert disk_block.location == BlockLocation.DISK

        # 分配 CPU 块（可能触发 CPU→DISK 换出）
        cpu_block_id = self._allocate_cpu_block()
        cpu_block = self.cpu_blocks[cpu_block_id - self.gpu_num_blocks]

        # 复制元数据
        cpu_block.hash = disk_block.hash
        cpu_block.token_ids = disk_block.token_ids.copy()
        cpu_block.ref_count = disk_block.ref_count
        cpu_block.last_access_time = disk_block.last_access_time
        cpu_block.access_count = disk_block.access_count
        cpu_block.owner_seqs = disk_block.owner_seqs.copy()

        # 更新前缀缓存
        if disk_block.hash != -1:
            if self.disk_hash_to_block_id.get(disk_block.hash) == disk_block_id:
                del self.disk_hash_to_block_id[disk_block.hash]
            self.cpu_hash_to_block_id[disk_block.hash] = cpu_block_id

        # 更新所有引用该块的序列的 block_table 和 block_locations
        for seq_id in disk_block.owner_seqs:
            if seq_id in self.seq_map:
                s = self.seq_map[seq_id]
                for j, bid in enumerate(s.block_table):
                    if bid == disk_block_id:
                        s.block_table[j] = cpu_block_id
                        s.block_locations[j] = BlockLocation.CPU
                        break

        # 释放 DISK 块
        disk_block.ref_count = 0
        self._deallocate_disk_block(disk_block_id)

        self.stats.total_swap_in += 1
        self.stats.total_swap_in_blocks += 1
        self.stats.cpu_disk_swap_in += 1

        return cpu_block_id

    # ==================== CPU → GPU 换入 ====================

    def swap_in_cpu_to_gpu(self, cpu_block_id: int) -> int:
        """
        将 CPU 中的块换入到 GPU
        返回新的 GPU 块 ID
        """
        assert self.gpu_num_blocks <= cpu_block_id < self.disk_base_id

        cpu_block = self.cpu_blocks[cpu_block_id - self.gpu_num_blocks]
        assert cpu_block.location == BlockLocation.CPU

        # 分配 GPU 块
        gpu_block_id = self._allocate_gpu_block()
        gpu_block = self.gpu_blocks[gpu_block_id]

        # 复制元数据
        gpu_block.hash = cpu_block.hash
        gpu_block.token_ids = cpu_block.token_ids.copy()
        gpu_block.ref_count = cpu_block.ref_count
        gpu_block.last_access_time = cpu_block.last_access_time
        gpu_block.access_count = cpu_block.access_count
        gpu_block.owner_seqs = cpu_block.owner_seqs.copy()

        # 更新前缀缓存
        if cpu_block.hash != -1:
            if self.cpu_hash_to_block_id.get(cpu_block.hash) == cpu_block_id:
                del self.cpu_hash_to_block_id[cpu_block.hash]
            self.gpu_hash_to_block_id[cpu_block.hash] = gpu_block_id

        # 更新所有引用该块的序列的 block_table 和 block_locations
        for seq_id in cpu_block.owner_seqs:
            if seq_id in self.seq_map:
                s = self.seq_map[seq_id]
                for j, bid in enumerate(s.block_table):
                    if bid == cpu_block_id:
                        s.block_table[j] = gpu_block_id
                        s.block_locations[j] = BlockLocation.GPU
                        break

        # 释放 CPU 块
        cpu_block.ref_count = 0
        self._deallocate_cpu_block(cpu_block_id)

        self.stats.total_swap_in += 1
        self.stats.total_swap_in_blocks += 1
        self.stats.gpu_cpu_swap_in += 1

        return gpu_block_id

    # ==================== 序列级别的换入换出 ====================

    def ensure_blocks_in_gpu(self, seq: Sequence, start_block: int, end_block: int) -> bool:
        """
        确保序列的指定块范围都在 GPU 中
        DISK → CPU → GPU 逐级换入
        """
        current_time = self._tick()
        success = True

        for i in range(start_block, end_block):
            if i >= len(seq.block_table):
                break

            block_id = seq.block_table[i]
            location = seq.block_locations[i]

            if location == BlockLocation.DISK:
                # DISK → CPU
                try:
                    cpu_block_id = self.swap_in_disk_to_cpu(block_id)
                    seq.block_table[i] = cpu_block_id
                    seq.block_locations[i] = BlockLocation.CPU
                    block_id = cpu_block_id
                    location = BlockLocation.CPU
                except RuntimeError:
                    success = False
                    break

            if location == BlockLocation.CPU:
                # CPU → GPU
                try:
                    gpu_block_id = self.swap_in_cpu_to_gpu(block_id)
                    seq.block_table[i] = gpu_block_id
                    seq.block_locations[i] = BlockLocation.GPU

                    new_block = self.gpu_blocks[gpu_block_id]
                    new_block.access(current_time)
                except RuntimeError:
                    success = False
                    break
            else:
                # 已经在 GPU
                block = self.gpu_blocks[block_id]
                block.access(current_time)

        return success

    def get_block_heat_map(self) -> dict[int, float]:
        """
        获取所有块的热度映射

        热度计算公式：结合 LRU 和 LFU
        - LRU 部分：最近访问的块热度高（1 / (idle_time + 1)）
        - LFU 部分：访问次数多的块热度高（log(access_count + 1)）
        - 两者相加，综合考虑最近访问和访问频率
        """
        import math
        heat_map = {}
        current_time = self._time_counter

        for block_id in self.gpu_used_block_ids:
            block = self.gpu_blocks[block_id]
            idle_time = current_time - block.last_access_time
            # 结合 LRU 和 LFU 的热度公式
            lru_score = 1.0 / (idle_time + 1)
            lfu_score = math.log(block.access_count + 1)
            heat = lru_score + lfu_score
            heat_map[block_id] = heat

        for block_id in self.cpu_used_block_ids:
            block = self.cpu_blocks[block_id - self.gpu_num_blocks]
            idle_time = current_time - block.last_access_time
            lru_score = 1.0 / (idle_time + 1)
            lfu_score = math.log(block.access_count + 1)
            heat = lru_score + lfu_score
            heat_map[block_id] = heat

        for block_id in self.disk_blocks:
            block = self.disk_blocks[block_id]
            idle_time = current_time - block.last_access_time
            lru_score = 1.0 / (idle_time + 1)
            lfu_score = math.log(block.access_count + 1)
            heat = lru_score + lfu_score
            heat_map[block_id] = heat

        return heat_map

    def record_disk_read(self, num_bytes: int, elapsed_time: float):
        """记录磁盘读取统计（由 ModelRunner 在实际读取后调用）"""
        self.stats.disk_read_bytes += num_bytes
        self.stats.disk_read_time += elapsed_time

    def record_disk_write(self, num_bytes: int, elapsed_time: float):
        """记录磁盘写入统计（由 ModelRunner 在实际写入后调用）"""
        self.stats.disk_write_bytes += num_bytes
        self.stats.disk_write_time += elapsed_time

    def get_stats(self) -> MultiLevelCacheStats:
        """获取统计信息"""
        return self.stats

    def get_block_location(self, block_id: int) -> BlockLocation:
        """获取块的位置"""
        block = self._get_block(block_id)
        return block.location
