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
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    # 原有字段
    hf_config: AutoConfig | None = None

    # ========== 新增远端Swap全部配置字段 ==========
    enable_remote_swap: bool = True
    remote_host: str = "127.0.0.1"
    remote_port: int = 12345
    remote_connect_retry: int = 5
    remote_op_retry: int = 3
    remote_socket_timeout: float = 10.0
    remote_evict_batch_size: int = 8
    # ===========================================

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)

        # 远端参数合法性校验
        assert 1 <= self.remote_connect_retry <= 10
        assert 1 <= self.remote_op_retry <= 5
        assert 0.5 < self.remote_socket_timeout < 30
        assert self.remote_evict_batch_size >= 1
        
        # 【新增】如果用户指定了 num_kvcache_blocks，验证合法性
        if self.num_kvcache_blocks != -1:
            assert self.num_kvcache_blocks > 0, f"num_kvcache_blocks 必须为正整数，当前值: {self.num_kvcache_blocks}"