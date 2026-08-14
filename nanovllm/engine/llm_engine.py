import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")

        # 选择 ModelRunner 类
        enable_multilevel = getattr(config, "enable_multilevel_kvcache", False) and getattr(config, "cpu_num_kvcache_blocks", 0) > 0

        if enable_multilevel:
            from nanovllm.engine.multi_level_model_runner import MultiLevelModelRunner
            ModelRunnerClass = MultiLevelModelRunner
            print(f"[LLMEngine] Using MultiLevelModelRunner")
        else:
            from nanovllm.engine.model_runner import ModelRunner
            ModelRunnerClass = ModelRunner

        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunnerClass, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)

        self.model_runner = ModelRunnerClass(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id

        # 根据配置选择调度器
        if enable_multilevel:
            from nanovllm.engine.multi_level_scheduler import MultiLevelScheduler
            self.scheduler = MultiLevelScheduler(config)
            print(f"[LLMEngine] Using MultiLevelScheduler with {config.cpu_num_kvcache_blocks} CPU blocks")

            # 将块管理器传递给 model_runner
            if hasattr(self.model_runner, 'set_block_manager'):
                self.model_runner.set_block_manager(self.scheduler.block_manager)

            # 初始化磁盘存储（如果启用了磁盘缓存）
            if getattr(config, "enable_disk_cache", False) and getattr(config, "disk_num_kvcache_blocks", 0) > 0:
                hf_config = config.hf_config
                num_kv_heads = hf_config.num_key_value_heads // config.tensor_parallel_size
                head_dim = getattr(
                    hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads
                )
                block_shape = (
                    2,  # K and V
                    hf_config.num_hidden_layers,
                    config.kvcache_block_size,
                    num_kv_heads,
                    head_dim,
                )
                # 从 model_runner 获取 dtype
                dtype = self.model_runner.kv_cache.dtype
                self.scheduler.block_manager.init_disk_store(
                    cache_dir=getattr(config, "disk_cache_dir", "./kv_disk_cache"),
                    block_shape=block_shape,
                    dtype=dtype,
                )
                print(f"[LLMEngine] Disk cache initialized: {config.disk_num_kvcache_blocks} blocks")
        else:
            self.scheduler = Scheduler(config)
            print(f"[LLMEngine] Using standard Scheduler")

        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
