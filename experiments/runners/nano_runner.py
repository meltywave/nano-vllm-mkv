import random
from pathlib import Path
from time import perf_counter
from dataclasses import fields

from experiments.configuration import load_benchmark_preset, load_cache_engine
from experiments.metrics import PeakRssSampler, output_sha256, tokenizer_files_sha256
from experiments.results import (
    model_config_sha256, model_manifest_sha256, software_metadata,
    summarize, write_result,
)
from experiments.workload import load_workload, tokenize_workload


def _set_seed(torch, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _measurement(repeat, llm, prompts, sampling_params, input_tokens):
    rss_sampler = PeakRssSampler()
    rss_sampler.start()
    try:
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False,
                               report_progress=True)
    finally:
        peak_rss = rss_sampler.stop()
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
        output_token_ids=[output["token_ids"] for output in outputs],
        gpu_peak_allocated_bytes=runner["gpu_peak_allocated_bytes"],
        gpu_peak_reserved_bytes=runner["gpu_peak_reserved_bytes"],
        cpu_peak_rss_bytes=peak_rss,
        cpu_rss_includes_children=rss_sampler.includes_children,
        cpu_rss_sample_interval_s=rss_sampler.interval_s,
        cpu_pinned_capacity_bytes=runner["cpu_pinned_capacity_bytes"],
        transfer_counts=runner["transfer_counts"],
        transfer_bytes=runner["transfer_bytes"],
        transfer_time_s=runner["transfer_time_s"],
        transfer_time_by_direction_s=runner["transfer_time_by_direction_s"],
        io_errors=runner["io_errors"],
        swap_in_blocks=scheduler["swap_in_blocks"],
        swap_out_blocks=scheduler["swap_out_blocks"],
        preemptions=scheduler["preemptions"],
        recompute_preemptions=scheduler["recompute_preemptions"],
        recompute_tokens=scheduler["recompute_tokens"],
        partial_swap_events=scheduler["partial_swap_events"],
        peak_mixed_sequences=scheduler["peak_mixed_sequences"],
        tier_stats=scheduler["tiers"],
        peak_tier_blocks={
            tier: stats["peak_used_blocks"]
            for tier, stats in scheduler["tiers"].items()
        },
        error=None,
    )
    return generation


def run_nano_experiment(
    engine: int,
    model: str,
    workload_path: str | Path,
    result_dir: str | Path | None = None,
    remote_mode: str | None = None,
    remote_host: str | None = None,
    remote_namespace: str | None = None,
    remote_port: int | None = None,
    remote_timeout: float | None = None,
    ssd_path: str | Path | None = None,
    gpu_blocks: int | None = None,
    cpu_blocks: int | None = None,
    ssd_blocks: int | None = None,
    remote_blocks: int | None = None,
    warmup_runs: int | None = None,
    measurement_repeats: int | None = None,
    watermark_overrides: list[str] | None = None,
    config_id: str | None = None,
    kv_placement: str = "sequence",
    record_logit_margins: bool = False,
    fp32_greedy_rerank: bool = False,
    verify_kv_transfers: bool = False,
    trace_logits: str | None = None,
    enforce_eager: bool = False,
    disable_prefix_cache: bool = False,
):
    import torch
    from nanovllm import LLM, SamplingParams

    workload = load_workload(workload_path)
    benchmark = load_benchmark_preset(workload.benchmark_preset)
    if warmup_runs is not None:
        if warmup_runs < 0:
            raise ValueError("--warmup-runs must be non-negative")
        benchmark["warmup_runs"] = warmup_runs
    if measurement_repeats is not None:
        if measurement_repeats <= 0:
            raise ValueError("--measurement-repeats must be positive")
        benchmark["measurement_repeats"] = measurement_repeats
    cache_kwargs, cache_snapshot = load_cache_engine(engine)
    cache_kwargs["kv_placement"] = kv_placement
    cache_kwargs["kv_record_logit_margins"] = record_logit_margins
    cache_kwargs["kv_fp32_greedy_rerank"] = fp32_greedy_rerank
    cache_kwargs["kv_verify_transfers"] = verify_kv_transfers
    if trace_logits is not None:
        try:
            request_index, token_index = (int(value) for value in trace_logits.split(":"))
        except ValueError as exc:
            raise ValueError("--trace-logits must be REQUEST:TOKEN") from exc
        if request_index < 0 or token_index < 0:
            raise ValueError("--trace-logits indices must be non-negative")
        cache_kwargs["kv_trace_logits_request"] = request_index
        cache_kwargs["kv_trace_logits_token"] = token_index
    cache_kwargs["enforce_eager"] = enforce_eager
    cache_kwargs["kv_enable_prefix_cache"] = not disable_prefix_cache
    if engine == 4 and remote_mode is None:
        raise ValueError("cache_engine=4 requires an explicit --remote-mode")
    runtime_overrides = {}
    runtime_overrides["kv_placement"] = kv_placement
    if enforce_eager:
        runtime_overrides["enforce_eager"] = True
    if disable_prefix_cache:
        runtime_overrides["kv_enable_prefix_cache"] = False
    if config_id is not None:
        runtime_overrides["config_id"] = config_id
    if watermark_overrides:
        high = dict(cache_kwargs["kv_high_watermarks"])
        low = dict(cache_kwargs["kv_low_watermarks"])
        active = tuple(cache_snapshot["tiers"][:-1])
        for raw in watermark_overrides:
            try:
                tier, raw_low, raw_high = raw.split(":")
                low_value, high_value = float(raw_low), float(raw_high)
            except (ValueError, TypeError) as exc:
                raise ValueError("--watermark must be TIER:LOW:HIGH") from exc
            if tier not in active:
                raise ValueError(
                    f"{tier!r} is not an active demotion tier for engine={engine}"
                )
            if not 0 <= low_value < high_value <= 1:
                raise ValueError("watermark requires 0 <= low < high <= 1")
            low[tier], high[tier] = low_value, high_value
        cache_kwargs["kv_low_watermarks"] = low
        cache_kwargs["kv_high_watermarks"] = high
        runtime_overrides["watermarks"] = {
            tier: {"low": low[tier], "high": high[tier]} for tier in active
        }
    if remote_mode is not None:
        if engine != 4:
            raise ValueError("--remote-mode is only valid for cache_engine=4")
        if remote_mode == "network":
            if not remote_host:
                raise ValueError("--remote-host is required for --remote-mode network")
            if remote_host in {"127.0.0.1", "localhost", "::1"}:
                raise ValueError("network remote mode cannot use a loopback host")
        if remote_mode == "loopback" and remote_host is None:
            remote_host = "127.0.0.1"
        if remote_mode == "loopback" and remote_host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("loopback remote mode requires a loopback host")
        runtime_overrides["remote_mode"] = remote_mode
    if remote_host is not None:
        cache_kwargs["remote_kvcache_host"] = remote_host
        runtime_overrides["remote_kvcache_host"] = remote_host
    if remote_namespace is not None:
        if engine != 4:
            raise ValueError("--remote-namespace is only valid for cache_engine=4")
        cache_kwargs["remote_kvcache_namespace"] = remote_namespace
        runtime_overrides["remote_kvcache_namespace"] = remote_namespace
    if remote_port is not None:
        if not 1 <= remote_port <= 65535:
            raise ValueError("--remote-port must be between 1 and 65535")
        cache_kwargs["remote_kvcache_port"] = remote_port
        runtime_overrides["remote_kvcache_port"] = remote_port
    if remote_timeout is not None:
        if remote_timeout <= 0:
            raise ValueError("--remote-timeout must be positive")
        cache_kwargs["remote_kvcache_timeout"] = remote_timeout
        runtime_overrides["remote_kvcache_timeout"] = remote_timeout
    if ssd_path is not None:
        resolved_ssd_path = str(Path(ssd_path).expanduser().resolve())
        cache_kwargs["ssd_kvcache_path"] = resolved_ssd_path
        runtime_overrides["ssd_kvcache_path"] = resolved_ssd_path
    capacity_overrides = {
        "num_kvcache_blocks": gpu_blocks,
        "num_cpu_kvcache_blocks": cpu_blocks,
        "num_ssd_kvcache_blocks": ssd_blocks,
        "num_remote_kvcache_blocks": remote_blocks,
    }
    for field_name, value in capacity_overrides.items():
        if value is None:
            continue
        if value < 0:
            raise ValueError(f"{field_name} must be non-negative")
        cache_kwargs[field_name] = value
        runtime_overrides[field_name] = value
    cache_snapshot["runtime_overrides"] = runtime_overrides
    max_model_len = int(workload.document["max_model_len"])
    max_num_seqs = int(
        workload.document.get("max_num_seqs", workload.document["num_seqs"])
    )
    cache_kwargs.update(
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=int(
            workload.document.get(
                "max_num_batched_tokens",
                int(workload.document["target_prompt_tokens"]) * max_num_seqs,
            )
        ),
    )

    init_started = perf_counter()
    llm = LLM(model, **cache_kwargs)
    initialization_s = perf_counter() - init_started
    try:
        expected = {
            "num_kvcache_blocks": gpu_blocks,
            "num_cpu_kvcache_blocks": cpu_blocks if engine >= 2 else None,
            "num_ssd_kvcache_blocks": ssd_blocks if engine >= 3 else None,
            "num_remote_kvcache_blocks": remote_blocks if engine >= 4 else None,
        }
        for name, requested in expected.items():
            if requested is not None and getattr(llm.config, name) != requested:
                raise RuntimeError(
                    f"Requested {name}={requested}, resolved {getattr(llm.config, name)}; "
                    "fix capacity before comparing experiments"
                )
        prompts, prompt_records = tokenize_workload(workload, llm.tokenizer)
        if max(map(len, prompts)) + int(workload.document["output_tokens"]) > llm.config.max_model_len:
            raise ValueError(
                "Tokenized prompt plus output_tokens exceeds the model's resolved max_model_len"
            )
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
                "weights_sha256": model_manifest_sha256(model),
                "config_sha256": model_config_sha256(model),
                "tokenizer_sha256": tokenizer_files_sha256(model),
            },
            "workload": {
                **workload.document,
                "sha256": workload.sha256,
                "actual_input_tokens": input_tokens,
                "input_token_ids_sha256": output_sha256(
                    [{"token_ids": prompt} for prompt in prompts]
                ),
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
                "resolved_watermarks": {
                    tier: {
                        "low": llm.config.kv_low_watermarks[tier],
                        "high": llm.config.kv_high_watermarks[tier],
                    }
                    for tier in llm.config.kv_cache_tiers
                },
                "kv_block_bytes": llm.model_runner.kv_block_bytes,
                "resolved_ssd_cache_path": llm.config.ssd_kvcache_path,
                "resolved_remote_host": llm.config.remote_kvcache_host,
                "resolved_remote_port": llm.config.remote_kvcache_port,
                "resolved_config": {
                    field.name: getattr(llm.config, field.name)
                    for field in fields(llm.config)
                    if field.name not in {"hf_config", "ssd_cache_id"}
                },
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
