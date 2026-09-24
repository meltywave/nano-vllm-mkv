import random
from pathlib import Path
from time import perf_counter

from experiments.configuration import load_benchmark_preset, load_cache_engine
from experiments.metrics import output_sha256, process_rss_bytes
from experiments.results import software_metadata, summarize, write_result
from experiments.workload import load_workload, tokenize_workload


def _set_seed(torch, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _measurement(repeat, llm, prompts, sampling_params, input_tokens):
    rss_before = process_rss_bytes()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    rss_after = process_rss_bytes()
    generation = dict(llm.last_generate_stats)
    cache = llm.cache_stats()
    runner = cache["runner"]
    scheduler = cache["scheduler"]
    output_tokens = sum(len(output["token_ids"]) for output in outputs)
    generation.update(
        repeat=repeat,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        output_sha256=output_sha256(outputs),
        gpu_peak_allocated_bytes=runner["gpu_peak_allocated_bytes"],
        gpu_peak_reserved_bytes=runner["gpu_peak_reserved_bytes"],
        cpu_peak_rss_bytes=max(
            value for value in (rss_before, rss_after) if value is not None
        ) if rss_before is not None or rss_after is not None else None,
        cpu_pinned_capacity_bytes=runner["cpu_pinned_capacity_bytes"],
        transfer_counts=runner["transfer_counts"],
        transfer_bytes=runner["transfer_bytes"],
        transfer_time_s=runner["transfer_time_s"],
        io_errors=runner["io_errors"],
        swap_in_blocks=scheduler["swap_in_blocks"],
        swap_out_blocks=scheduler["swap_out_blocks"],
        preemptions=scheduler["preemptions"],
        recompute_preemptions=scheduler["recompute_preemptions"],
        tier_stats=scheduler["tiers"],
        error=None,
    )
    return generation


def run_nano_experiment(
    engine: int,
    model: str,
    workload_path: str | Path,
    result_dir: str | Path | None = None,
):
    import torch
    from nanovllm import LLM, SamplingParams

    workload = load_workload(workload_path)
    benchmark = load_benchmark_preset(workload.benchmark_preset)
    cache_kwargs, cache_snapshot = load_cache_engine(engine)
    max_model_len = int(workload.document["max_model_len"])
    cache_kwargs.update(
        max_model_len=max_model_len,
        max_num_seqs=int(workload.document["num_seqs"]),
        max_num_batched_tokens=(
            int(workload.document["target_prompt_tokens"])
            * int(workload.document["num_seqs"])
        ),
    )

    init_started = perf_counter()
    llm = LLM(model, **cache_kwargs)
    initialization_s = perf_counter() - init_started
    try:
        prompts, prompt_records = tokenize_workload(workload, llm.tokenizer)
        input_tokens = sum(len(prompt) for prompt in prompts)
        sampling = workload.document["sampling"]
        sampling_params = SamplingParams(
            temperature=float(sampling["temperature"]),
            ignore_eos=bool(sampling["ignore_eos"]),
            max_tokens=int(workload.document["output_tokens"]),
        )
        seed = int(workload.document["seed"])

        for _ in range(benchmark["warmup_runs"]):
            _set_seed(torch, seed)
            llm.reset_for_benchmark()
            llm.generate(prompts, sampling_params, use_tqdm=False)

        measurements = []
        for repeat in range(1, benchmark["measurement_repeats"] + 1):
            _set_seed(torch, seed)
            llm.reset_for_benchmark()
            measurements.append(
                _measurement(repeat, llm, prompts, sampling_params, input_tokens)
            )

        hf_config = llm.config.hf_config
        result = {
            "schema_version": 1,
            "run_id": None,
            "engine": {
                "level": engine,
                "tiers": list(llm.config.kv_cache_tiers),
                "policy": llm.config.kv_swap_policy,
            },
            "model": {
                "path": str(Path(model).resolve()),
                "id": Path(model).name,
                "revision": getattr(hf_config, "_commit_hash", None),
                "dtype": str(hf_config.dtype),
            },
            "workload": {
                **workload.document,
                "sha256": workload.sha256,
                "actual_input_tokens": input_tokens,
                "prompt_records": prompt_records,
            },
            "software": software_metadata(torch),
            "hardware": {
                "gpu": torch.cuda.get_device_name(0),
                "gpu_count": torch.cuda.device_count(),
                "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
            },
            "config_snapshot": {
                "cache": cache_snapshot,
                "benchmark": benchmark,
                "resolved_num_kvcache_blocks": llm.config.num_kvcache_blocks,
                "resolved_num_cpu_kvcache_blocks": llm.config.num_cpu_kvcache_blocks,
                "resolved_num_ssd_kvcache_blocks": llm.config.num_ssd_kvcache_blocks,
                "resolved_num_remote_kvcache_blocks": llm.config.num_remote_kvcache_blocks,
            },
            "initialization": {
                "total_s": initialization_s,
                "included_in_measurement": False,
            },
            "measurements": measurements,
            "summary": summarize(measurements),
        }
        if engine == 1:
            result["summary"]["correctness"]["matches_engine_1"] = result[
                "summary"
            ]["correctness"]["repeat_outputs_match"]
        path = write_result(result, result_dir)
        return path, result
    finally:
        llm.exit()
