import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from experiments.configuration import REPO_ROOT
from experiments.sweep_thresholds import (
    first_output_mismatch, required_tier_active, unexpected_tier_activity,
)
from experiments.workload import load_workload


FIELDS = (
    "source", "engine", "model", "model_weights_sha256",
    "model_config_sha256", "tokenizer_sha256", "workload",
    "workload_sha256", "input_token_ids_sha256",
    "run_id", "config_id", "cache_tiers", "kv_placement", "gpu_blocks", "cpu_blocks",
    "ssd_blocks", "remote_blocks", "gpu_low", "gpu_high",
    "cpu_low", "cpu_high", "ssd_low", "ssd_high",
    "measurement_repeats", "latency_mean_s", "output_tokens_per_s",
    "requests_per_s", "ttft_p50_ms", "ttft_p95_ms",
    "tpot_p50_ms", "tpot_p95_ms", "e2e_p50_ms", "e2e_p95_ms",
    "gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes",
    "cpu_pinned_capacity_bytes", "repeat_outputs_match", "exact_token_rate",
    "quality_status", "quality_reason", "quality_mean_f1",
    "quality_f1_lower_95", "quality_manifest_sha256",
    "cpu_peak_rss_bytes", "cpu_rss_includes_children",
    "matches_internal_gpu_only", "transfer_counts", "transfer_bytes",
    "watermark_triggers", "watermark_demotions",
    "new_tier_active", "expected_tier_active", "unexpected_tier_active",
    "io_errors", "recompute_tokens", "partial_swap_events",
    "peak_mixed_sequences", "correctness_mismatch",
    "status", "correctness_policy", "quality_directory", "result_path",
    "remote_mode", "remote_host", "remote_port", "ssd_path",
    "transfer_time_by_direction_s", "peak_tier_blocks",
    "git_commit", "source_sha256", "model_dtype", "sampling_temperature",
    "sampling_ignore_eos", "request_count", "output_tokens",
    "max_num_seqs", "max_num_batched_tokens", "gpu_total_bytes",
    "ttft_count", "tpot_count", "e2e_count",
)


def _path_key(result):
    return (
        result["model"]["path"],
        result["model"].get("weights_sha256"),
        result["model"].get("config_sha256"),
        result["model"].get("tokenizer_sha256"),
        result["workload"]["sha256"],
        result["workload"].get("input_token_ids_sha256"),
        result["workload"]["sampling"]["temperature"],
        result["model"].get("dtype"),
        result["config_snapshot"]["resolved_num_kvcache_blocks"],
        result["workload"].get("max_num_seqs", result["workload"].get("num_seqs")),
        result["config_snapshot"].get("resolved_config", {}).get("max_num_batched_tokens"),
        result["config_snapshot"].get("resolved_config", {}).get("kvcache_block_size"),
        result["config_snapshot"].get("resolved_config", {}).get("kv_enable_prefix_cache"),
        result["config_snapshot"].get("resolved_config", {}).get("enforce_eager"),
        result.get("software", {}).get("source_sha256"),
    )


def _baseline_lookup(results):
    baselines = defaultdict(set)
    for _, result in results:
        if result["engine"]["level"] != 1:
            continue
        output_hash = result["summary"]["correctness"]["output_sha256"]
        if output_hash and result["summary"]["correctness"]["repeat_outputs_match"] and all(
            measurement.get("io_errors", 0) == 0
            for measurement in result.get("measurements", [])
        ):
            baselines[_path_key(result)].add(output_hash)
    return baselines


def _result_row(path, result, baseline_hashes, baseline_results=None):
    summary = result["summary"]
    config = result["config_snapshot"]
    cache = config["cache"]
    engine = result["engine"]["level"]
    measurements = result["measurements"]
    transfer_counts = defaultdict(int)
    transfer_bytes = defaultdict(int)
    triggers = defaultdict(int)
    demotions = defaultdict(int)
    io_errors = 0
    recompute_tokens = 0
    partial_swap_events = 0
    peak_mixed_sequences = 0
    transfer_times = defaultdict(float)
    peak_tier_blocks = defaultdict(int)
    rss_values = []
    rss_child_flags = []
    for measurement in measurements:
        for name, count in measurement.get("transfer_counts", {}).items():
            transfer_counts[name] += count
        for name, count in measurement.get("transfer_bytes", {}).items():
            transfer_bytes[name] += count
        for tier, stats in measurement.get("tier_stats", {}).items():
            triggers[tier] += stats.get("watermark_triggers", 0)
            demotions[tier] += stats.get("watermark_demotions", 0)
        io_errors += measurement.get("io_errors", 0)
        recompute_tokens += measurement.get("recompute_tokens", 0)
        partial_swap_events += measurement.get("partial_swap_events", 0)
        peak_mixed_sequences = max(
            peak_mixed_sequences, measurement.get("peak_mixed_sequences", 0)
        )
        for name, elapsed in measurement.get("transfer_time_by_direction_s", {}).items():
            transfer_times[name] += elapsed
        for tier, used in measurement.get("peak_tier_blocks", {}).items():
            peak_tier_blocks[tier] = max(peak_tier_blocks[tier], used)
        if measurement.get("cpu_peak_rss_bytes") is not None:
            rss_values.append(measurement["cpu_peak_rss_bytes"])
        rss_child_flags.append(measurement.get("cpu_rss_includes_children", False))
    new_tier = {2: "cpu", 3: "ssd", 4: "remote"}.get(engine)
    active = None if new_tier is None else any(
        any(direction.endswith(f"_to_{new_tier}") and count > 0
            for direction, count in measurement.get("transfer_counts", {}).items())
        for measurement in measurements
    )
    expected_active = required_tier_active(result, engine)
    unexpected_active = unexpected_tier_activity(result, engine)
    baselines = baseline_hashes.get(_path_key(result), set())
    output_hash = summary["correctness"]["output_sha256"]
    matches = (output_hash in baselines if len(baselines) == 1 and output_hash else None)
    reference = (baseline_results or {}).get(_path_key(result), [])
    mismatch = None
    if engine != 1 and len(reference) == 1 and matches is not True:
        if all("output_token_ids" in item for item in result["measurements"]):
            mismatch = first_output_mismatch(result, reference[0])
    resolved = config.get("resolved_watermarks", cache.get("watermarks", {}))
    row = {name: "" for name in FIELDS}
    row.update(
        source="nano-vLLM-MKV", engine=engine, model=result["model"]["id"],
        model_weights_sha256=result["model"].get("weights_sha256", ""),
        model_config_sha256=result["model"].get("config_sha256", ""),
        tokenizer_sha256=result["model"].get("tokenizer_sha256", ""),
        workload=result["workload"]["name"],
        workload_sha256=result["workload"]["sha256"],
        input_token_ids_sha256=result["workload"].get("input_token_ids_sha256", ""),
        run_id=result["run_id"],
        config_id=cache.get("runtime_overrides", {}).get("config_id", ""),
        cache_tiers="+".join(result["engine"]["tiers"]),
        kv_placement=config.get("resolved_config", {}).get("kv_placement", "sequence"),
        gpu_blocks=config.get("resolved_num_kvcache_blocks"),
        cpu_blocks=config.get("resolved_num_cpu_kvcache_blocks"),
        ssd_blocks=config.get("resolved_num_ssd_kvcache_blocks"),
        remote_blocks=config.get("resolved_num_remote_kvcache_blocks"),
        measurement_repeats=summary["measurement_repeats"],
        latency_mean_s=summary["latency_s"]["mean"],
        output_tokens_per_s=summary.get("output_tokens_per_s", {}).get("mean"),
        requests_per_s=summary.get("requests_per_s", {}).get("mean"),
        gpu_peak_allocated_bytes=summary["gpu_peak_allocated_bytes"]["max"],
        gpu_peak_reserved_bytes=max(
            (measurement.get("gpu_peak_reserved_bytes", 0) for measurement in measurements),
            default=0,
        ),
        cpu_pinned_capacity_bytes=max(
            (measurement.get("cpu_pinned_capacity_bytes", 0) for measurement in measurements),
            default=0,
        ),
        cpu_peak_rss_bytes=max(rss_values) if rss_values else "",
        cpu_rss_includes_children=all(rss_child_flags) if rss_child_flags else False,
        repeat_outputs_match=summary["correctness"]["repeat_outputs_match"],
        exact_token_rate=(1.0 if engine == 1 else ""),
        quality_status="quality_not_evaluated",
        quality_reason="No matched quality artifact",
        matches_internal_gpu_only=matches,
        transfer_counts=json.dumps(dict(transfer_counts), sort_keys=True),
        transfer_bytes=json.dumps(dict(transfer_bytes), sort_keys=True),
        watermark_triggers=json.dumps(dict(triggers), sort_keys=True),
        watermark_demotions=json.dumps(dict(demotions), sort_keys=True),
        new_tier_active=active, expected_tier_active=expected_active,
        unexpected_tier_active=unexpected_active, io_errors=io_errors,
        recompute_tokens=recompute_tokens,
        partial_swap_events=partial_swap_events,
        peak_mixed_sequences=peak_mixed_sequences,
        correctness_mismatch=mismatch or "",
        status=("valid" if io_errors == 0 and summary["correctness"]["repeat_outputs_match"]
                and matches is not False else "invalid"),
        result_path=str(path),
        remote_mode=cache.get("runtime_overrides", {}).get("remote_mode", ""),
        remote_host=config.get("resolved_remote_host", ""),
        remote_port=config.get("resolved_remote_port", ""),
        ssd_path=config.get("resolved_ssd_cache_path", ""),
        transfer_time_by_direction_s=json.dumps(dict(transfer_times), sort_keys=True),
        peak_tier_blocks=json.dumps(dict(peak_tier_blocks), sort_keys=True),
        git_commit=result.get("software", {}).get("git_commit", ""),
        source_sha256=result.get("software", {}).get("source_sha256", ""),
        model_dtype=result["model"].get("dtype", ""),
        sampling_temperature=result["workload"].get("sampling", {}).get("temperature", ""),
        sampling_ignore_eos=result["workload"].get("sampling", {}).get("ignore_eos", ""),
        request_count=result["workload"].get("request_count", result["workload"].get("num_seqs")),
        output_tokens=result["workload"].get("output_tokens", ""),
        max_num_seqs=result["workload"].get("max_num_seqs", result["workload"].get("num_seqs")),
        max_num_batched_tokens=config.get("resolved_config", {}).get("max_num_batched_tokens", ""),
        gpu_total_bytes=result.get("hardware", {}).get("gpu_total_bytes", ""),
    )
    for tier in ("gpu", "cpu", "ssd"):
        if tier in result["engine"]["tiers"][:-1]:
            row[f"{tier}_low"] = resolved.get(tier, {}).get("low", "")
            row[f"{tier}_high"] = resolved.get(tier, {}).get("high", "")
    latencies = summary.get("request_latency_ms", {})
    for name in ("ttft", "tpot", "e2e"):
        row[f"{name}_count"] = latencies.get(name, {}).get("count", "")
        row[f"{name}_p50_ms"] = latencies.get(name, {}).get("p50", "")
        row[f"{name}_p95_ms"] = latencies.get(name, {}).get("p95", "")
    if unexpected_active and row["status"] == "valid":
        row["status"] = "miscalibrated"
    elif not expected_active and row["status"] == "valid":
        row["status"] = "inactive"
    return row


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def external_rows(root):
    rows = []
    for source, filename in (("vLLM", "vllm.jsonl"), ("nano-vLLM", "nano_vllm.jsonl")):
        path = root / "external" / filename
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                required = (
                    "model", "model_weights_sha256", "model_config_sha256",
                    "tokenizer_sha256",
                    "workload", "workload_sha256", "input_token_ids_sha256",
                    "model_dtype", "sampling_temperature", "sampling_ignore_eos",
                    "request_count", "output_tokens", "max_num_seqs",
                    "max_num_batched_tokens", "gpu_total_bytes", "git_commit",
                    "output_tokens_per_s", "requests_per_s",
                )
                missing = [name for name in required if name not in record]
                if missing:
                    raise ValueError(f"{path}:{line_number} missing {missing}")
                row = {name: record.get(name, "") for name in FIELDS}
                row["source"] = source
                row["engine"] = source
                row["result_path"] = f"{path}:{line_number}"
                row["status"] = "external"
                rows.append(row)
    return rows


def validate_external(rows, internal):
    if not rows:
        return rows
    internal = [row for row in internal if row.get("status", "valid") in {"valid", "inactive"}]
    if not internal:
        raise ValueError("External rows have no matching completed MKV L4 run")
    def key(row):
        return (
            row["model"], row["model_weights_sha256"],
            row.get("model_config_sha256", ""),
            row["tokenizer_sha256"], row["workload_sha256"],
            row["input_token_ids_sha256"],
            str(row.get("model_dtype", "")).removeprefix("torch."),
            float(row["sampling_temperature"]),
            str(row.get("sampling_ignore_eos", "")).lower(),
            int(row.get("request_count", 0)), int(row.get("output_tokens", 0)),
            int(row.get("max_num_seqs", 0)),
            int(row.get("max_num_batched_tokens", 0)),
            int(row["gpu_total_bytes"]),
        )

    known = {
        key(row)
        for row in internal
    }
    for row in rows:
        if key(row) not in known:
            raise ValueError(
                f"External row {row.get('result_path', '<external>')} has no matching MKV model/workload"
            )
    return rows


def select_ablation(rows, config_ids):
    selected = [
        row for row in rows
        if row["engine"] in config_ids and row["config_id"] == config_ids[row["engine"]]
    ]
    workloads = {
        (row["model"], row["model_weights_sha256"],
         row.get("model_config_sha256", ""), row["tokenizer_sha256"],
         row["workload"], row["workload_sha256"], row.get("input_token_ids_sha256", ""))
        for row in rows
    }
    for model, weights_hash, config_hash, tokenizer_hash, workload, workload_hash, input_hash in sorted(workloads):
        for engine, config_id in sorted(config_ids.items()):
            if any(
                row["engine"] == engine and row["config_id"] == config_id
                and row["model"] == model
                and row.get("model_config_sha256", "") == config_hash
                and row["workload_sha256"] == workload_hash
                and row.get("input_token_ids_sha256", "") == input_hash
                for row in selected
            ):
                continue
            pending = {name: "" for name in FIELDS}
            pending.update(
                source="nano-vLLM-MKV", engine=engine, model=model,
                model_weights_sha256=weights_hash,
                model_config_sha256=config_hash,
                tokenizer_sha256=tokenizer_hash, workload=workload,
                workload_sha256=workload_hash, config_id=config_id,
                input_token_ids_sha256=input_hash,
                status="not_run",
            )
            selected.append(pending)
    _require_unique_runs(selected, "ablation")
    return sorted(selected, key=lambda row: (row["workload"], row["engine"], row["run_id"]))


def _require_unique_runs(rows, table_name):
    seen = set()
    for row in rows:
        key = (
            row.get("source", "nano-vLLM-MKV"), row["engine"], row["model_weights_sha256"],
            row.get("model_config_sha256", ""),
            row["tokenizer_sha256"], row["workload_sha256"],
        )
        if key in seen:
            raise ValueError(
                f"Multiple {table_name} runs for the same engine/model/workload; "
                "use a separate result directory or select one run explicitly"
            )
        seen.add(key)


def _ledger_entries(root, pattern):
    entries = []
    for ledger in sorted(root.rglob(pattern)):
        with ledger.open("r", newline="", encoding="utf-8") as file:
            latest = {}
            for row in csv.DictReader(file):
                latest[row["config_id"]] = row
            entries.extend(latest.values())
    return entries


def _local_result_path(root, raw_path):
    path = Path(raw_path)
    if path.exists():
        return path.resolve()
    parts = raw_path.replace("\\", "/").split("/")
    if root.name in parts:
        candidate = root.joinpath(*parts[parts.index(root.name) + 1:])
        if candidate.exists():
            return candidate.resolve()
    return path.resolve()


def _ledger_table_rows(entries, result_rows, root=None):
    root = Path(root) if root is not None else REPO_ROOT / "results"
    rows = []
    for entry in entries:
        result_path = entry.get("result_path", "")
        row = result_rows.get(str(_local_result_path(root, result_path))) if result_path else None
        if row is None and entry["status"] in {"ok", "inactive", "invalid"}:
            raise ValueError(
                f"{entry['config_id']} is marked {entry['status']} but its result.json is missing"
            )
        if row is not None:
            row = dict(row)
            row["status"] = "valid" if entry["status"] == "ok" else entry["status"]
        else:
            row = {name: "" for name in FIELDS}
            command = json.loads(entry.get("command") or "[]")
            options = dict(zip(command[3::2], command[4::2]))
            model = Path(options.get("--model", "")).name
            workload_path = Path(options.get("--workload", ""))
            workload = workload_path.stem
            workload_hash = entry.get("workload_sha256", "")
            if workload_path:
                manifest = workload_path if workload_path.is_absolute() else REPO_ROOT / workload_path
                try:
                    loaded = load_workload(manifest)
                except (OSError, ValueError):
                    pass
                else:
                    workload = loaded.name
                    workload_hash = loaded.sha256
            reference = next(
                (item for item in result_rows.values()
                 if item["model"] == model and item["workload"] == workload),
                None,
            )
            row.update(
                source="nano-vLLM-MKV", engine=int(entry["engine"]),
                model=model, workload=workload, workload_sha256=workload_hash,
                status=entry["status"],
            )
            if reference:
                for name in ("model_weights_sha256", "model_config_sha256", "tokenizer_sha256", "workload_sha256", "input_token_ids_sha256"):
                    row[name] = reference[name]
        row["config_id"] = entry["config_id"]
        row["result_path"] = result_path
        if entry.get("correctness_mismatch"):
            row["correctness_mismatch"] = entry["correctness_mismatch"]
        row["correctness_policy"] = entry.get("correctness_policy", "strict")
        row["quality_directory"] = entry.get("quality_directory", "")
        for name in ("exact_token_rate", "quality_status", "quality_reason",
                     "quality_mean_f1", "quality_f1_lower_95", "quality_manifest_sha256"):
            if entry.get(name) not in (None, ""):
                row[name] = entry[name]
        for name in ("source_sha256", "model_weights_sha256", "model_config_sha256", "workload_sha256"):
            if entry.get(name) and row[name] and row[name] != entry[name]:
                raise ValueError(f"{entry['config_id']} result has a different {name} than its ledger")
            if entry.get(name):
                row[name] = entry[name]
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Build the three experiment tables")
    parser.add_argument("result_dir", nargs="?", default="results")
    parser.add_argument("--main-config-id", help="Pre-locked L4 config ID for main table")
    parser.add_argument("--thresholds-only", action="store_true", help="Export completed sweep rows before ablation runs")
    parser.add_argument(
        "--ablation-config-id", action="append", default=[], metavar="ENGINE:ID",
        help="Lock one config ID for an ablation level; repeat for engines 1-4",
    )
    args = parser.parse_args()
    if not args.main_config_id and not args.thresholds_only:
        raise ValueError("--main-config-id must name the pre-locked L4 configuration")
    ablation_ids = {
        1: "e1", 2: "e2-default", 3: "e3-default", 4: args.main_config_id
    }
    for value in args.ablation_config_id:
        try:
            level, config_id = value.split(":", 1)
            level = int(level)
        except ValueError as exc:
            raise ValueError("--ablation-config-id must be ENGINE:ID") from exc
        if level not in ablation_ids or not config_id:
            raise ValueError("Ablation engine must be 1-4 with nonempty ID")
        ablation_ids[level] = config_id
    root = Path(args.result_dir)
    results = []
    for path in root.rglob("result.json"):
        with path.open("r", encoding="utf-8") as file:
            results.append((path, json.load(file)))
    ablation_entries = _ledger_entries(root, "ablation_runs_*.csv")
    sweep_entries = _ledger_entries(root, "threshold_sweep_*.csv")
    if not results and not ablation_entries and not sweep_entries:
        raise ValueError(f"No experiment results found under {root}")
    selected_paths = {
        str(_local_result_path(root, entry["result_path"]))
        for entry in [*ablation_entries, *sweep_entries]
        if entry.get("result_path")
    }
    if ablation_entries or sweep_entries:
        results = [(path, result) for path, result in results
                   if str(path.resolve()) in selected_paths]
    baseline_hashes = _baseline_lookup(results)
    baseline_results = defaultdict(list)
    for _, result in results:
        if result["engine"]["level"] == 1 and result["summary"]["correctness"]["repeat_outputs_match"]:
            baseline_results[_path_key(result)].append(result)
    rows = [_result_row(path, result, baseline_hashes, baseline_results)
            for path, result in results]
    for row in rows:
        if row["engine"] != 1 and row["matches_internal_gpu_only"] is None and row["status"] == "valid":
            row["status"] = "unpaired"
    rows.sort(key=lambda row: (row["workload"], row["engine"], row["run_id"]))
    by_path = {str(Path(row["result_path"]).resolve()): row for row in rows}
    ablation_source = (
        _ledger_table_rows(ablation_entries, by_path, root) if ablation_entries else rows
    )
    threshold_rows = (
        _ledger_table_rows(sweep_entries, by_path, root) if sweep_entries else rows
    )
    output = root / "tables"
    if args.thresholds_only:
        write_csv(output / "thresholds.csv", threshold_rows)
        print((output / "thresholds.csv").resolve())
        return
    ablation_rows = select_ablation(ablation_source, ablation_ids)
    main_rows = [
        row for row in ablation_source
        if row["engine"] == 4 and row["config_id"] == args.main_config_id
    ]
    if (rows or ablation_entries or sweep_entries) and not main_rows:
        raise ValueError(
            f"No L4 result with pre-locked config_id={args.main_config_id!r}"
        )
    main_rows.extend(validate_external(external_rows(root), main_rows))
    workloads = {
        (row["model"], row["model_weights_sha256"],
         row.get("model_config_sha256", ""), row["tokenizer_sha256"],
         row["workload"], row["workload_sha256"],
         row.get("input_token_ids_sha256", ""))
        for row in main_rows
    }
    for model, weights_hash, config_hash, tokenizer_hash, workload, workload_hash, input_hash in sorted(workloads):
        for source in ("vLLM", "nano-vLLM"):
            if any(
                row["source"] == source
                and row["model"] == model
                and row["model_weights_sha256"] == weights_hash
                and row.get("model_config_sha256", "") == config_hash
                and row["tokenizer_sha256"] == tokenizer_hash
                and row["workload_sha256"] == workload_hash
                and row.get("input_token_ids_sha256", "") == input_hash
                for row in main_rows
            ):
                continue
            pending = {name: "" for name in FIELDS}
            pending.update(
                source=source, engine=source, model=model, workload=workload,
                model_weights_sha256=weights_hash,
                model_config_sha256=config_hash,
                tokenizer_sha256=tokenizer_hash,
                workload_sha256=workload_hash, input_token_ids_sha256=input_hash,
                status="external_pending",
            )
            main_rows.append(pending)
    _require_unique_runs(main_rows, "main-table")
    write_csv(output / "ablation.csv", ablation_rows)
    write_csv(output / "thresholds.csv", threshold_rows)
    write_csv(output / "main.csv", main_rows)
    print(output.resolve())


if __name__ == "__main__":
    main()
