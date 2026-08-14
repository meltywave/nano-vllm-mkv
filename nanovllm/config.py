import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    # ==================== 多级 KV Cache 配置 ====================
    # 是否启用多级缓存
    enable_multilevel_kvcache: bool = False

    # CPU 缓存块数量（0 表示不使用 CPU 缓存）
    cpu_num_kvcache_blocks: int = 0

    # CPU 内存利用率（用于自动计算 CPU 块数）
    cpu_memory_utilization: float = 0.5

    # 替换策略：lru / lfu / lru_k / arc
    replacement_policy: str = "lru"

    # 换出水水位线（GPU 使用率超过此值触发换出）
    swap_watermark_high: float = 0.9

    # 换入低水位线（换出到此值以下停止）
    swap_watermark_low: float = 0.7

    # 是否启用预取
    enable_prefetch: bool = True

    # 预取超前块数
    prefetch_lookahead: int = 2

    # PCIe 带宽（GB/s），用于估算换入换出时间
    swap_bandwidth_gbps: float = 10.0

    # ==================== SSD 磁盘缓存配置 ====================
    # 是否启用磁盘缓存（第三级）
    enable_disk_cache: bool = False

    # 磁盘缓存块数量（0 表示不使用磁盘缓存）
    disk_num_kvcache_blocks: int = 0

    # 磁盘缓存目录
    disk_cache_dir: str = "./kv_disk_cache"

    # 磁盘换出高水位线（CPU 使用率超过此值触发换出到磁盘）
    disk_swap_watermark_high: float = 0.9

    # 磁盘换入低水位线
    disk_swap_watermark_low: float = 0.7

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)

        # 验证多级缓存配置
        if self.enable_multilevel_kvcache:
            assert self.cpu_num_kvcache_blocks >= 0
            assert 0 < self.swap_watermark_low < self.swap_watermark_high <= 1.0
            assert self.replacement_policy in ["lru", "lfu", "lru_k", "arc"]
            assert self.prefetch_lookahead >= 0

            # 验证磁盘缓存配置
            if self.enable_disk_cache:
                assert self.disk_num_kvcache_blocks >= 0
                assert 0 < self.disk_swap_watermark_low < self.disk_swap_watermark_high <= 1.0
                # 启用磁盘缓存时，CPU 缓存也必须启用
                assert self.cpu_num_kvcache_blocks > 0, "Disk cache requires CPU cache to be enabled"
