"""
三级 KV Cache ModelRunner 扩展
支持 GPU ↔ CPU ↔ DISK 三级的数据传输

功能：
- GPU ↔ CPU：通过 PCIe 直接拷贝（pin_memory + cuda stream）
- CPU ↔ DISK：通过文件系统序列化/反序列化（torch.save/load）
- 预取：提前将即将访问的块换入 GPU
- 异步传输：使用独立 CUDA stream 与计算重叠
"""

import os
import time
from typing import Optional
import torch

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.multi_level_block_manager import (
    MultiLevelBlockManager,
    BlockLocation,
)


class MultiLevelModelRunnerMixin:
    """
    ModelRunner 的多级缓存 Mixin

    提供 GPU ↔ CPU ↔ DISK 三级 KV Cache 的数据传输能力。
    需要与 ModelRunner 配合使用，在 allocate_kv_cache 之后调用 init_multilevel_kvcache。
    """

    def init_multilevel_kvcache(self, config: Config):
        """
        初始化多级 KV Cache

        在 allocate_kv_cache 之后调用。
        """
        self.enable_multilevel = getattr(config, "enable_multilevel_kvcache", False)
        if not self.enable_multilevel:
            return

        self.cpu_num_blocks = getattr(config, "cpu_num_kvcache_blocks", 0)
        self.disk_num_blocks = getattr(config, "disk_num_kvcache_blocks", 0)
        self.enable_disk = self.disk_num_blocks > 0
        self.disk_cache_dir = getattr(config, "disk_cache_dir", "./kv_disk_cache")
        self.enable_prefetch = getattr(config, "enable_prefetch", True)
        self.prefetch_lookahead = getattr(config, "prefetch_lookahead", 2)

        if self.cpu_num_blocks <= 0:
            return

        hf_config = config.hf_config
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(
            hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads
        )

        # CPU KV Cache：pin_memory，方便快速 DMA 传输
        self.cpu_kv_cache = torch.empty(
            2,
            hf_config.num_hidden_layers,
            self.cpu_num_blocks,
            self.block_size,
            num_kv_heads,
            head_dim,
            dtype=hf_config.dtype,
            pin_memory=True,
        )
        print(
            f"[MultiLevel] CPU KV Cache allocated: "
            f"{self.cpu_kv_cache.numel() * self.cpu_kv_cache.element_size() / 1024**3:.2f} GB"
        )

        # 异步传输用的 CUDA stream
        self.swap_stream = torch.cuda.Stream()

        # 统计
        self.swap_stats = {
            "gpu_to_cpu_bytes": 0,
            "cpu_to_gpu_bytes": 0,
            "cpu_to_disk_bytes": 0,
            "disk_to_cpu_bytes": 0,
            "gpu_to_cpu_time": 0.0,
            "cpu_to_gpu_time": 0.0,
            "cpu_to_disk_time": 0.0,
            "disk_to_cpu_time": 0.0,
        }

        # 初始化块管理器的磁盘存储（如果启用了磁盘缓存）
        if self.enable_disk:
            self._init_disk_cache(config)

    def _init_disk_cache(self, config: Config):
        """初始化磁盘缓存"""
        hf_config = config.hf_config
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(
            hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads
        )

        block_shape = (
            2,
            hf_config.num_hidden_layers,
            self.block_size,
            num_kv_heads,
            head_dim,
        )

        # 创建磁盘缓存目录
        os.makedirs(self.disk_cache_dir, exist_ok=True)

        print(
            f"[MultiLevel] Disk cache initialized: "
            f"{self.disk_num_blocks} blocks at {self.disk_cache_dir}"
        )

    # ==================== GPU ↔ CPU 传输 ====================

    def swap_out_blocks(self, gpu_block_ids: list[int], cpu_block_ids: list[int]):
        """
        将 GPU 块换出到 CPU

        Args:
            gpu_block_ids: GPU 中的块 ID 列表
            cpu_block_ids: CPU 中的块 ID 列表（与 gpu_block_ids 一一对应）
        """
        if not gpu_block_ids:
            return

        t0 = time.perf_counter()

        with torch.cuda.stream(self.swap_stream):
            for gpu_bid, cpu_bid in zip(gpu_block_ids, cpu_block_ids):
                # K 和 V 各一层
                for kv_idx in range(2):
                    src = self.kv_cache[kv_idx, :, gpu_bid]
                    dst = self.cpu_kv_cache[kv_idx, :, cpu_bid - self.config.num_kvcache_blocks]
                    dst.copy_(src, non_blocking=True)

        self.swap_stream.synchronize()

        elapsed = time.perf_counter() - t0
        bytes_transferred = (
            len(gpu_block_ids)
            * 2
            * self.config.hf_config.num_hidden_layers
            * self.block_size
            * (self.config.hf_config.num_key_value_heads // self.world_size)
            * getattr(
                self.config.hf_config,
                "head_dim",
                self.config.hf_config.hidden_size // self.config.hf_config.num_attention_heads,
            )
            * self.kv_cache.element_size()
        )

        self.swap_stats["gpu_to_cpu_bytes"] += bytes_transferred
        self.swap_stats["gpu_to_cpu_time"] += elapsed

    def swap_in_blocks(self, cpu_block_ids: list[int], gpu_block_ids: list[int]):
        """
        将 CPU 块换入到 GPU

        Args:
            cpu_block_ids: CPU 中的块 ID 列表
            gpu_block_ids: GPU 中的块 ID 列表（与 cpu_block_ids 一一对应）
        """
        if not cpu_block_ids:
            return

        t0 = time.perf_counter()

        with torch.cuda.stream(self.swap_stream):
            for cpu_bid, gpu_bid in zip(cpu_block_ids, gpu_block_ids):
                for kv_idx in range(2):
                    src = self.cpu_kv_cache[kv_idx, :, cpu_bid - self.config.num_kvcache_blocks]
                    dst = self.kv_cache[kv_idx, :, gpu_bid]
                    dst.copy_(src, non_blocking=True)

        self.swap_stream.synchronize()

        elapsed = time.perf_counter() - t0
        bytes_transferred = (
            len(cpu_block_ids)
            * 2
            * self.config.hf_config.num_hidden_layers
            * self.block_size
            * (self.config.hf_config.num_key_value_heads // self.world_size)
            * getattr(
                self.config.hf_config,
                "head_dim",
                self.config.hf_config.hidden_size // self.config.hf_config.num_attention_heads,
            )
            * self.kv_cache.element_size()
        )

        self.swap_stats["cpu_to_gpu_bytes"] += bytes_transferred
        self.swap_stats["cpu_to_gpu_time"] += elapsed

    # ==================== CPU ↔ DISK 传输 ====================

    def swap_out_cpu_to_disk(self, cpu_block_ids: list[int], disk_logical_ids: list[int]):
        """
        将 CPU 块换出到 DISK

        Args:
            cpu_block_ids: CPU 中的块 ID 列表
            disk_logical_ids: DISK 中的逻辑块 ID 列表
        """
        if not cpu_block_ids or not self.enable_disk:
            return

        t0 = time.perf_counter()
        total_bytes = 0

        for cpu_bid, disk_lid in zip(cpu_block_ids, disk_logical_ids):
            # 从 CPU KV Cache 中取出块数据
            cpu_idx = cpu_bid - self.config.num_kvcache_blocks
            block_data = self.cpu_kv_cache[:, :, cpu_idx].clone()

            # 保存到磁盘
            fpath = os.path.join(self.disk_cache_dir, f"block_{disk_lid:08d}.pt")
            torch.save(block_data, fpath)
            total_bytes += os.path.getsize(fpath)

        elapsed = time.perf_counter() - t0
        self.swap_stats["cpu_to_disk_bytes"] += total_bytes
        self.swap_stats["cpu_to_disk_time"] += elapsed

    def swap_in_disk_to_cpu(self, disk_logical_ids: list[int], cpu_block_ids: list[int]):
        """
        将 DISK 块换入到 CPU

        Args:
            disk_logical_ids: DISK 中的逻辑块 ID 列表
            cpu_block_ids: CPU 中的块 ID 列表
        """
        if not disk_logical_ids or not self.enable_disk:
            return

        t0 = time.perf_counter()
        total_bytes = 0

        for disk_lid, cpu_bid in zip(disk_logical_ids, cpu_block_ids):
            # 从磁盘加载
            fpath = os.path.join(self.disk_cache_dir, f"block_{disk_lid:08d}.pt")
            block_data = torch.load(fpath, map_location="cpu", weights_only=True)

            # 写入 CPU KV Cache
            cpu_idx = cpu_bid - self.config.num_kvcache_blocks
            self.cpu_kv_cache[:, :, cpu_idx].copy_(block_data)
            total_bytes += block_data.numel() * block_data.element_size()

        elapsed = time.perf_counter() - t0
        self.swap_stats["disk_to_cpu_bytes"] += total_bytes
        self.swap_stats["disk_to_cpu_time"] += elapsed

    # ==================== 序列级别的换入换出 ====================

    def swap_out_sequence(self, seq: Sequence, block_manager: MultiLevelBlockManager):
        """
        将序列的 GPU 块换出到 CPU

        正确的流程：
        1. 调用块管理器的 swap_out_sequence_to_cpu，更新元数据
        2. 执行实际的数据传输（GPU→CPU）
        """
        if not self.enable_multilevel:
            return

        # 调用块管理器执行元数据层面的换出
        gpu_block_ids, cpu_block_ids = block_manager.swap_out_sequence_to_cpu(seq)

        # 执行实际的数据传输
        if gpu_block_ids:
            self.swap_out_blocks(gpu_block_ids, cpu_block_ids)

    def swap_in_sequence(self, seq: Sequence, block_manager: MultiLevelBlockManager):
        """
        将序列的 CPU 块换入到 GPU

        正确的流程：
        1. 调用块管理器的 swap_in 方法，更新元数据
        2. 执行实际的数据传输（CPU→GPU）

        注意：这里只处理 CPU→GPU 的换入
        DISK→CPU 的换入由 ensure_blocks_in_gpu 统一处理
        """
        if not self.enable_multilevel:
            return

        cpu_block_ids = []
        gpu_block_ids = []

        for i, block_id in enumerate(seq.block_table):
            if seq.block_locations[i] == BlockLocation.CPU:
                try:
                    gpu_block_id = block_manager.swap_in_cpu_to_gpu(block_id)
                    # 注意：swap_in_cpu_to_gpu 已经更新了所有引用序列的 block_table
                    cpu_block_ids.append(block_id)
                    gpu_block_ids.append(gpu_block_id)
                except RuntimeError:
                    # GPU 满了，先触发换出腾出空间
                    block_manager._trigger_gpu_swap_out_if_needed()
                    try:
                        gpu_block_id = block_manager.swap_in_cpu_to_gpu(block_id)
                        cpu_block_ids.append(block_id)
                        gpu_block_ids.append(gpu_block_id)
                    except RuntimeError:
                        continue

        if cpu_block_ids:
            self.swap_in_blocks(cpu_block_ids, gpu_block_ids)

    def ensure_blocks_in_gpu(self, seq: Sequence, block_manager: MultiLevelBlockManager):
        """
        确保序列的所有块都在 GPU 中
        DISK → CPU → GPU 逐级换入
        """
        if not self.enable_multilevel:
            return

        # 先检查有没有 DISK 块
        disk_blocks = [
            (i, bid)
            for i, bid in enumerate(seq.block_table)
            if seq.block_locations[i] == BlockLocation.DISK
        ]

        if disk_blocks and self.enable_disk:
            # DISK → CPU
            disk_logical_ids = []
            cpu_block_ids = []
            indices = []

            for i, disk_bid in disk_blocks:
                try:
                    cpu_block_id = block_manager.swap_in_disk_to_cpu(disk_bid)
                except RuntimeError:
                    # CPU 满了，先换出一些
                    block_manager._trigger_cpu_swap_out_if_needed()
                    try:
                        cpu_block_id = block_manager.swap_in_disk_to_cpu(disk_bid)
                    except RuntimeError:
                        continue

                disk_logical_ids.append(disk_bid - block_manager.disk_base_id)
                cpu_block_ids.append(cpu_block_id)
                indices.append(i)

            # 执行数据传输
            if disk_logical_ids:
                self.swap_in_disk_to_cpu(disk_logical_ids, cpu_block_ids)

                # 更新序列的块位置
                for idx, cpu_bid in zip(indices, cpu_block_ids):
                    seq.block_table[idx] = cpu_bid
                    seq.block_locations[idx] = BlockLocation.CPU

        # 再检查 CPU 块，换入到 GPU
        cpu_blocks = [
            (i, bid)
            for i, bid in enumerate(seq.block_table)
            if seq.block_locations[i] == BlockLocation.CPU
        ]

        if cpu_blocks:
            cpu_block_ids = []
            gpu_block_ids = []
            indices = []

            for i, cpu_bid in cpu_blocks:
                try:
                    gpu_block_id = block_manager.swap_in_cpu_to_gpu(cpu_bid)
                except RuntimeError:
                    block_manager._trigger_gpu_swap_out_if_needed()
                    try:
                        gpu_block_id = block_manager.swap_in_cpu_to_gpu(cpu_bid)
                    except RuntimeError:
                        continue

                cpu_block_ids.append(cpu_bid)
                gpu_block_ids.append(gpu_block_id)
                indices.append(i)

            if cpu_block_ids:
                self.swap_in_blocks(cpu_block_ids, gpu_block_ids)

                for idx, gpu_bid in zip(indices, gpu_block_ids):
                    seq.block_table[idx] = gpu_bid
                    seq.block_locations[idx] = BlockLocation.GPU

    # ==================== 预取 ====================

    def prefetch_ahead(self, seq: Sequence, block_manager: MultiLevelBlockManager):
        """
        预取：提前将即将访问的块换入 GPU

        根据当前序列的位置，预取接下来的 prefetch_lookahead 个块。
        """
        if not self.enable_prefetch or not self.enable_multilevel:
            return

        current_block = (seq.num_tokens - 1) // self.block_size
        lookahead_end = min(
            current_block + self.prefetch_lookahead + 1,
            len(seq.block_table),
        )

        # 检查这些块是否在 GPU 中
        blocks_to_prefetch = []
        for i in range(current_block + 1, lookahead_end):
            if i < len(seq.block_table) and seq.block_locations[i] != BlockLocation.GPU:
                blocks_to_prefetch.append(i)

        if not blocks_to_prefetch:
            return

        # 逐个换入
        for i in blocks_to_prefetch:
            block_id = seq.block_table[i]
            location = seq.block_locations[i]

            if location == BlockLocation.DISK and self.enable_disk:
                try:
                    cpu_block_id = block_manager.swap_in_disk_to_cpu(block_id)
                    disk_lid = block_id - block_manager.disk_base_id
                    self.swap_in_disk_to_cpu([disk_lid], [cpu_block_id])
                    seq.block_table[i] = cpu_block_id
                    seq.block_locations[i] = BlockLocation.CPU
                    block_id = cpu_block_id
                    location = BlockLocation.CPU
                except RuntimeError:
                    continue

            if location == BlockLocation.CPU:
                try:
                    gpu_block_id = block_manager.swap_in_cpu_to_gpu(block_id)
                    self.swap_in_blocks([block_id], [gpu_block_id])
                    seq.block_table[i] = gpu_block_id
                    seq.block_locations[i] = BlockLocation.GPU
                except RuntimeError:
                    continue

    # ==================== 运行时集成 ====================

    def run_with_multilevel(
        self,
        seqs: list[Sequence],
        is_prefill: bool,
        block_manager: MultiLevelBlockManager,
    ) -> list[int]:
        """
        多级缓存感知的 run 方法

        在执行推理前确保所有需要的块都在 GPU 中，
        执行后触发预取。
        """
        if not self.enable_multilevel:
            return self.run(seqs, is_prefill)

        # 确保所有序列的块都在 GPU 中
        for seq in seqs:
            self.ensure_blocks_in_gpu(seq, block_manager)

        # 执行推理
        token_ids = self.run(seqs, is_prefill)

        # 触发预取（decode 阶段）
        if not is_prefill and self.enable_prefetch:
            for seq in seqs:
                self.prefetch_ahead(seq, block_manager)

        return token_ids

    # ==================== 统计 ====================

    def get_swap_stats(self) -> dict:
        """获取传输统计信息"""
        stats = self.swap_stats.copy()

        # 计算带宽
        if stats["gpu_to_cpu_time"] > 0:
            stats["gpu_to_cpu_bandwidth_gbps"] = (
                stats["gpu_to_cpu_bytes"] / stats["gpu_to_cpu_time"] / 1e9
            )
        else:
            stats["gpu_to_cpu_bandwidth_gbps"] = 0.0

        if stats["cpu_to_gpu_time"] > 0:
            stats["cpu_to_gpu_bandwidth_gbps"] = (
                stats["cpu_to_gpu_bytes"] / stats["cpu_to_gpu_time"] / 1e9
            )
        else:
            stats["cpu_to_gpu_bandwidth_gbps"] = 0.0

        if stats["cpu_to_disk_time"] > 0:
            stats["cpu_to_disk_bandwidth_mbps"] = (
                stats["cpu_to_disk_bytes"] / stats["cpu_to_disk_time"] / 1e6
            )
        else:
            stats["cpu_to_disk_bandwidth_mbps"] = 0.0

        if stats["disk_to_cpu_time"] > 0:
            stats["disk_to_cpu_bandwidth_mbps"] = (
                stats["disk_to_cpu_bytes"] / stats["disk_to_cpu_time"] / 1e6
            )
        else:
            stats["disk_to_cpu_bandwidth_mbps"] = 0.0

        return stats


class AsyncMultiLevelModelRunnerMixin(MultiLevelModelRunnerMixin):
    """
    异步版本的多级缓存 Mixin

    使用独立的 CUDA stream 进行数据传输，
    与计算 stream 重叠，隐藏传输延迟。
    """

    def init_multilevel_kvcache(self, config: Config):
        super().init_multilevel_kvcache(config)
        if self.enable_multilevel:
            # 额外的事件用于同步
            self.swap_events = []

    def swap_out_blocks_async(self, gpu_block_ids: list[int], cpu_block_ids: list[int]):
        """异步换出（不等待完成）"""
        if not gpu_block_ids:
            return

        with torch.cuda.stream(self.swap_stream):
            for gpu_bid, cpu_bid in zip(gpu_block_ids, cpu_block_ids):
                for kv_idx in range(2):
                    src = self.kv_cache[kv_idx, :, gpu_bid]
                    dst = self.cpu_kv_cache[kv_idx, :, cpu_bid - self.config.num_kvcache_blocks]
                    dst.copy_(src, non_blocking=True)

        event = torch.cuda.Event()
        event.record(self.swap_stream)
        self.swap_events.append(event)

    def swap_in_blocks_async(self, cpu_block_ids: list[int], gpu_block_ids: list[int]):
        """异步换入（不等待完成）"""
        if not cpu_block_ids:
            return

        with torch.cuda.stream(self.swap_stream):
            for cpu_bid, gpu_bid in zip(cpu_block_ids, gpu_block_ids):
                for kv_idx in range(2):
                    src = self.cpu_kv_cache[kv_idx, :, cpu_bid - self.config.num_kvcache_blocks]
                    dst = self.kv_cache[kv_idx, :, gpu_bid]
                    dst.copy_(src, non_blocking=True)

        event = torch.cuda.Event()
        event.record(self.swap_stream)
        self.swap_events.append(event)

    def wait_all_swaps(self):
        """等待所有异步传输完成"""
        for event in self.swap_events:
            event.synchronize()
        self.swap_events.clear()


# 延迟导入避免循环依赖
def _get_model_runner_class():
    from nanovllm.engine.model_runner import ModelRunner
    return ModelRunner


class MultiLevelModelRunner:
    """
    多级缓存 ModelRunner

    包装 ModelRunner，增加 GPU + CPU + SSD 三级 KV Cache 支持。

    使用方式：
        from nanovllm.engine.multi_level_model_runner import MultiLevelModelRunner
        runner = MultiLevelModelRunner(config, rank, event)
    """

    def __init__(self, config: Config, rank: int, event):
        ModelRunner = _get_model_runner_class()
        self._runner = ModelRunner(config, rank, event)

        # 复制关键属性
        self.config = config
        self.rank = rank
        self.world_size = config.tensor_parallel_size
        self.block_size = config.kvcache_block_size
        self.kv_cache = self._runner.kv_cache

        # 初始化多级缓存
        self.enable_multilevel = False
        self.init_multilevel_kvcache(config)

        # 块管理器引用（由调度器设置）
        self.block_manager_ref = None

    def __getattr__(self, name):
        """代理到 base runner"""
        return getattr(self._runner, name)

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """重写 run 方法，支持多级缓存"""
        if self.enable_multilevel and self.block_manager_ref is not None:
            # 确保所有需要的块都在 GPU 中
            for seq in seqs:
                self.ensure_blocks_in_gpu(seq, self.block_manager_ref)

        # 执行推理
        token_ids = self._runner.run(seqs, is_prefill)

        # 预取（decode 阶段）
        if (
            self.enable_multilevel
            and not is_prefill
            and self.enable_prefetch
            and self.block_manager_ref is not None
        ):
            for seq in seqs:
                self.prefetch_ahead(seq, self.block_manager_ref)

        return token_ids

    def call(self, method_name, *args):
        """重写 call 方法"""
        if method_name == "run":
            return self.run(*args)
        return self._runner.call(method_name, *args)

    def set_block_manager(self, block_manager):
        """设置块管理器引用"""
        self.block_manager_ref = block_manager

    def exit(self):
        """退出"""
        self._runner.exit()

    def loop(self):
        """子进程循环"""
        self._runner.loop()
