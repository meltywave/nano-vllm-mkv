import os
import tempfile
import uuid
from typing import Any
from dataclasses import dataclass, field


ENGINE_TIERS = {
    1: ("gpu",),
    2: ("gpu", "cpu"),
    3: ("gpu", "cpu", "ssd"),
    4: ("gpu", "cpu", "ssd", "remote"),
}


def _default_high_watermarks():
    return {"gpu": 0.90, "cpu": 0.90, "ssd": 0.90, "remote": 0.95}


def _default_low_watermarks():
    return {"gpu": 0.70, "cpu": 0.70, "ssd": 0.70, "remote": 0.80}


@dataclass(slots=True)
class Config:
    model: str
    cache_engine: int = 1
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: Any = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    cpu_kvcache_gb: float = 0.0
    num_cpu_kvcache_blocks: int = -1
    ssd_kvcache_gb: float = 0.0
    num_ssd_kvcache_blocks: int = -1
    ssd_kvcache_path: str | None = None
    remote_kvcache_gb: float = 0.0
    num_remote_kvcache_blocks: int = -1
    remote_kvcache_host: str = "127.0.0.1"
    remote_kvcache_port: int = 19090
    remote_kvcache_timeout: float = 30.0
    kv_swap_policy: str = "waterfall_lru"
    kv_high_watermarks: dict[str, float] = field(
        default_factory=_default_high_watermarks
    )
    kv_low_watermarks: dict[str, float] = field(
        default_factory=_default_low_watermarks
    )
    kv_max_transfer_blocks: int = -1
    kv_enable_prefix_cache: bool = True
    kv_cache_tiers: tuple[str, ...] = field(init=False)
    ssd_cache_id: str = field(init=False, repr=False)

    def __post_init__(self):
        if not os.path.isdir(self.model):
            raise ValueError(f"Model directory does not exist: {self.model}")
        if self.cache_engine not in ENGINE_TIERS:
            raise ValueError("cache_engine must be one of 1, 2, 3, or 4")
        if self.kvcache_block_size <= 0 or self.kvcache_block_size % 256:
            raise ValueError("kvcache_block_size must be a positive multiple of 256")
        if not 1 <= self.tensor_parallel_size <= 8:
            raise ValueError("tensor_parallel_size must be between 1 and 8")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if self.kv_swap_policy != "waterfall_lru":
            raise ValueError("Only kv_swap_policy='waterfall_lru' is supported")
        if self.kv_max_transfer_blocks == 0 or self.kv_max_transfer_blocks < -1:
            raise ValueError("kv_max_transfer_blocks must be -1 or positive")

        self.kv_cache_tiers = ENGINE_TIERS[self.cache_engine]
        for tier in self.kv_cache_tiers:
            high = self.kv_high_watermarks.get(tier)
            low = self.kv_low_watermarks.get(tier)
            if high is None or low is None:
                raise ValueError(f"Missing watermarks for enabled tier {tier!r}")
            if not 0 <= low < high <= 1:
                raise ValueError(
                    f"Invalid {tier} watermarks: require 0 <= low < high <= 1"
                )

        capacity_fields = (
            ("cpu", "cpu_kvcache_gb", "num_cpu_kvcache_blocks"),
            ("ssd", "ssd_kvcache_gb", "num_ssd_kvcache_blocks"),
            ("remote", "remote_kvcache_gb", "num_remote_kvcache_blocks"),
        )
        for tier, gb_name, blocks_name in capacity_fields:
            gb = getattr(self, gb_name)
            blocks = getattr(self, blocks_name)
            if gb < 0 or blocks < -1:
                raise ValueError(f"Invalid capacity for {tier} KV cache")
            if tier not in self.kv_cache_tiers:
                setattr(self, gb_name, 0.0)
                setattr(self, blocks_name, 0)

        if not self.remote_kvcache_host:
            raise ValueError("remote_kvcache_host must not be empty")
        if not 1 <= self.remote_kvcache_port <= 65535:
            raise ValueError("remote_kvcache_port must be between 1 and 65535")
        if self.remote_kvcache_timeout <= 0:
            raise ValueError("remote_kvcache_timeout must be positive")
        if self.ssd_kvcache_path is None:
            self.ssd_kvcache_path = tempfile.gettempdir()
        self.ssd_kvcache_path = os.path.abspath(
            os.path.expanduser(self.ssd_kvcache_path)
        )
        self.ssd_cache_id = uuid.uuid4().hex
        from transformers import AutoConfig

        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)

    def watermarks(self, tier: str) -> tuple[float, float]:
        return self.kv_high_watermarks[tier], self.kv_low_watermarks[tier]
