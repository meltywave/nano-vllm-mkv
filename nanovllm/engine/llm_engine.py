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
        config_fields = {field.name for field in fields(Config) if field.init}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.config = config
        Sequence.block_size = config.kvcache_block_size
        self._exited = False
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        if self._exited:
            return
        self._exited = True
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        return seq.seq_id

    def step(self):
        seqs, is_prefill, swap_in, swap_out = self.scheduler.schedule()
        completion_lengths = {seq.seq_id: seq.num_completion_tokens for seq in seqs}
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill, swap_in, swap_out)
        if seqs:
            self.scheduler.postprocess(seqs, token_ids, is_prefill)
        self.last_step_generated_seq_ids = [
            seq.seq_id
            for seq in seqs
            if seq.num_completion_tokens > completion_lengths[seq.seq_id]
        ]

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
        if len(sampling_params) != len(prompts):
            raise ValueError("sampling_params must have one entry per prompt")
        self.scheduler.reset_stats()
        self.model_runner.call("reset_cache_stats")
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        run_started = perf_counter()
        first_token_s = None
        prefill_tokens = decode_tokens = 0
        prefill_time_s = decode_time_s = 0.0
        prefill_throughput = decode_throughput = 0.0
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            elapsed = perf_counter() - t
            if first_token_s is None and self.last_step_generated_seq_ids:
                first_token_s = perf_counter() - run_started
            if num_tokens > 0:
                prefill_tokens += num_tokens
                prefill_time_s += elapsed
                prefill_throughput = prefill_tokens / prefill_time_s
            elif num_tokens < 0:
                decode_tokens += -num_tokens
                decode_time_s += elapsed
                decode_throughput = decode_tokens / decode_time_s
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
        wall_time_s = perf_counter() - run_started
        output_token_count = sum(len(output["token_ids"]) for output in outputs)
        self.last_generate_stats = {
            "latency_s": wall_time_s,
            "ttft_ms": None if first_token_s is None else first_token_s * 1000,
            "prefill_tokens": prefill_tokens,
            "decode_tokens": output_token_count,
            "prefill_time_s": prefill_time_s,
            "decode_time_s": decode_time_s,
            "prefill_tokens_per_s": (
                prefill_tokens / prefill_time_s if prefill_time_s else 0.0
            ),
            "decode_tokens_per_s": (
                decode_tokens / decode_time_s if decode_time_s else 0.0
            ),
            "total_tokens_per_s": (
                (prefill_tokens + output_token_count) / wall_time_s
                if wall_time_s else 0.0
            ),
        }
        return outputs

    def cache_stats(self):
        return {
            "scheduler": self.scheduler.stats(),
            "runner": self.model_runner.call("get_cache_stats"),
        }

    def reset_for_benchmark(self):
        """Reset measurements and cached blocks between isolated repetitions."""
        self.scheduler.reset_stats(clear_prefix_cache=True)
        self.model_runner.call("reset_cache_stats")
