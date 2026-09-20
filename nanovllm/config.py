import os
import tempfile
import uuid
from dataclasses import dataclass, field
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
    ssd_kvcache_gb: float = 0.0
    num_ssd_kvcache_blocks: int = -1
    ssd_kvcache_path: str | None = None
    ssd_cache_id: str = field(init=False, repr=False)

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.cpu_kvcache_gb >= 0
        assert self.num_cpu_kvcache_blocks >= -1
        assert self.ssd_kvcache_gb >= 0
        assert self.num_ssd_kvcache_blocks >= -1
        if self.ssd_kvcache_path is None:
            self.ssd_kvcache_path = tempfile.gettempdir()
        self.ssd_kvcache_path = os.path.abspath(
            os.path.expanduser(self.ssd_kvcache_path)
        )
        self.ssd_cache_id = uuid.uuid4().hex
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
