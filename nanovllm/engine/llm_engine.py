import atexit
import sys
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
        self.scheduler = Scheduler(
            config,
            transfer_blocks=lambda transfers: self.model_runner.call(
                "swap_blocks", [], transfers
            ),
        )
        self._greedy_margins_by_seq = {}
        self._trace_seq_id = None
        self._trace_logits = None
        self._replay_tokens_by_seq = None
        self._replay_rows_by_seq = None
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
        replay_targets = None
        if self._replay_tokens_by_seq is not None:
            replay_targets = {
                seq.seq_id: self._replay_tokens_by_seq[seq.seq_id][seq.num_completion_tokens]
                for seq in seqs
                if seq.num_completion_tokens < len(self._replay_tokens_by_seq[seq.seq_id])
            }
        token_ids, greedy_margins, trace, replay_rows = self.model_runner.call(
            "run", seqs, is_prefill, swap_in, swap_out,
            self._trace_seq_id, self.config.kv_trace_logits_token, replay_targets,
        )
        if replay_rows:
            for row in replay_rows:
                self._replay_rows_by_seq[row["seq_id"]].append(row)
        if trace is not None:
            self._trace_logits = trace
        if seqs:
            self.scheduler.postprocess(seqs, token_ids, is_prefill)
        self.last_step_generated_seq_ids = [
            seq.seq_id
            for seq in seqs
            if seq.num_completion_tokens > completion_lengths[seq.seq_id]
        ]
        if greedy_margins is not None:
            generated = set(self.last_step_generated_seq_ids)
            for seq, margin in zip(seqs, greedy_margins):
                if seq.seq_id in generated:
                    self._greedy_margins_by_seq.setdefault(seq.seq_id, []).append(margin)

        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
        report_progress: bool = False,
        replay_token_ids: list[list[int]] | None = None,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        if len(sampling_params) != len(prompts):
            raise ValueError("sampling_params must have one entry per prompt")
        if replay_token_ids is not None:
            if len(replay_token_ids) != len(prompts):
                raise ValueError("replay_token_ids must have one entry per prompt")
            for tokens, sp in zip(replay_token_ids, sampling_params):
                if (len(tokens) != sp.max_tokens or sp.temperature != 0
                        or not sp.ignore_eos or any(not isinstance(t, int) or t < 0 for t in tokens)):
                    raise ValueError("Replay requires max_tokens valid token IDs, greedy sampling, and ignore_eos")
        self.scheduler.reset_stats()
        self.model_runner.call("reset_cache_stats")
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        run_started = perf_counter()
        next_progress_s = 30.0
        request_ids = [seq.seq_id for seq in self.scheduler.waiting]
        self._replay_tokens_by_seq = (
            dict(zip(request_ids, replay_token_ids)) if replay_token_ids is not None else None
        )
        self._replay_rows_by_seq = (
            {seq_id: [] for seq_id in request_ids} if replay_token_ids is not None else None
        )
        if self.config.kv_trace_logits_request >= len(request_ids):
            raise ValueError("KV logit trace request index exceeds request count")
        self._trace_seq_id = (
            request_ids[self.config.kv_trace_logits_request]
            if self.config.kv_trace_logits_request >= 0 else None
        )
        self._trace_logits = None
        self._greedy_margins_by_seq = {seq_id: [] for seq_id in request_ids}
        first_token_ms = {}
        finish_ms = {}
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
            for seq_id in self.last_step_generated_seq_ids:
                first_token_ms.setdefault(
                    seq_id, (perf_counter() - run_started) * 1000
                )
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
                finish_ms[seq_id] = (perf_counter() - run_started) * 1000
                pbar.update(1)
            if report_progress and perf_counter() - run_started >= next_progress_s:
                print(
                    f"progress elapsed_s={perf_counter() - run_started:.1f} "
                    f"completed={len(outputs)}/{len(prompts)} "
                    f"running={len(self.scheduler.running)} "
                    f"swapped={len(self.scheduler.swapped)} "
                    f"waiting={len(self.scheduler.waiting)} "
                    f"swaps={self.scheduler.num_swap_out_blocks} "
                    f"recompute_tokens={self.scheduler.num_recompute_tokens}",
                    file=sys.stderr, flush=True,
                )
                next_progress_s += 30.0
        pbar.close()
        if len(outputs) != len(prompts):
            raise RuntimeError(
                f"Only {len(outputs)} of {len(prompts)} requests completed"
            )
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        wall_time_s = perf_counter() - run_started
        output_token_count = sum(len(output["token_ids"]) for output in outputs)
        request_timings = []
        for seq_id, output in zip(sorted(request_ids), outputs):
            first = first_token_ms.get(seq_id)
            finished = finish_ms.get(seq_id)
            count = len(output["token_ids"])
            request_timings.append(
                {
                    "seq_id": seq_id,
                    "ttft_ms": first,
                    "e2e_ms": finished,
                    "tpot_ms": (
                        (finished - first) / (count - 1)
                        if first is not None and finished is not None and count > 1
                        else None
                    ),
                    "output_tokens": count,
                }
            )
        self.last_generate_stats = {
            "latency_s": wall_time_s,
            "ttft_ms": None if first_token_s is None else first_token_s * 1000,
            "prefill_tokens": prefill_tokens,
            "decode_tokens": output_token_count,
            "request_timings": request_timings,
            "greedy_logit_margins": (
                [self._greedy_margins_by_seq[seq_id] for seq_id in sorted(request_ids)]
                if self.config.kv_record_logit_margins else None
            ),
            "trace_logits": self._trace_logits,
            "replay_logits": (
                [self._replay_rows_by_seq[seq_id] for seq_id in sorted(request_ids)]
                if self._replay_rows_by_seq is not None else None
            ),
            "requests_per_s": len(outputs) / wall_time_s if wall_time_s else 0.0,
            "output_tokens_per_s": (
                output_token_count / wall_time_s if wall_time_s else 0.0
            ),
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
