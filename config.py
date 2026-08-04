# nanovllm/config.py
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
    cpu_kvcache_gb: float = 0.0
    num_cpu_kvcache_blocks: int = -1

    # ========== 远端网络 Swap 配置 ==========
    enable_remote_swap: bool = False
    remote_host: str = "127.0.0.1"
    remote_port: int = 12345
    remote_connect_retry: int = 5
    remote_op_retry: int = 3
    remote_socket_timeout: float = 10.0
    remote_evict_batch_size: int = 8
    # =======================================

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.cpu_kvcache_gb >= 0
        assert self.num_cpu_kvcache_blocks >= -1
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)