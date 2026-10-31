"""Separate reproducible generation diagnostics from experiment validity."""

import json
from pathlib import Path


def exact_token_rate(result, baseline):
    reference = baseline["measurements"][0]["output_token_ids"]
    matches = total = 0
    for measurement in result["measurements"]:
        outputs = measurement["output_token_ids"]
        if len(outputs) != len(reference):
            raise ValueError("Candidate and L1 have different request counts")
        for left, right in zip(reference, outputs):
            if len(left) != len(right):
                raise ValueError("Candidate and L1 have different output lengths")
            total += len(left)
            matches += sum(a == b for a, b in zip(left, right))
    if total == 0:
        raise ValueError("No output tokens to compare")
    return matches / total


def require_quality_source(quality_dir, expected_source):
    baseline = Path(quality_dir) / "e1.json"
    if not baseline.is_file():
        raise ValueError(f"Quality baseline is missing: {baseline}")
    with baseline.open("r", encoding="utf-8") as file:
        record = json.load(file)
    if record.get("source_sha256") != expected_source:
        raise ValueError("Quality baseline has a different source; rerun quality with this code")


def quality_evidence(result, quality_dir):
    """Return a status only for quality data from the exact runtime and config."""
    if quality_dir is None:
        return {"status": "quality_not_evaluated", "reason": "No quality directory"}
    engine = result["engine"]["level"]
    root = Path(quality_dir)
    config_id = result["config_snapshot"].get("cache", {}).get("runtime_overrides", {}).get("config_id")
    named_run = root / f"{config_id}.json" if config_id else None
    run_path = named_run if named_run is not None and named_run.is_file() else root / f"e{engine}.json"
    baseline_path = root / "e1.json"
    report_path = root / f"{run_path.stem}_vs_e1.json"
    if not run_path.is_file() or not baseline_path.is_file() or (engine > 1 and not report_path.is_file()):
        return {"status": "quality_not_evaluated", "reason": "Quality artifact missing"}
    with run_path.open("r", encoding="utf-8") as file:
        run = json.load(file)
    with baseline_path.open("r", encoding="utf-8") as file:
        baseline = json.load(file)
    if run["engine"] != engine or baseline["engine"] != 1:
        raise ValueError("Quality artifact engine does not match")
    expected = (
        (run["source_sha256"], result["software"]["source_sha256"], "source"),
        (run["model_weights_sha256"], result["model"]["weights_sha256"], "weights"),
        (run["model_config_sha256"], result["model"]["config_sha256"], "model config"),
        (run["tokenizer_sha256"], result["model"]["tokenizer_sha256"], "tokenizer"),
        (run["gpu_blocks"], result["config_snapshot"]["resolved_num_kvcache_blocks"], "GPU blocks"),
        (run["gpu_total_bytes"], result["hardware"]["gpu_total_bytes"], "GPU memory"),
    )
    for actual, required, name in expected:
        if actual != required:
            return {"status": "quality_not_evaluated", "reason": f"{name} differs"}
    for name in ("source_sha256", "manifest_sha256", "gpu_blocks", "gpu_total_bytes"):
        if run[name] != baseline[name]:
            raise ValueError(f"Quality L1 and L{engine} differ in {name}")
    if run.get("kv_placement", "sequence") != baseline.get("kv_placement", "sequence"):
        raise ValueError("Quality L1 and candidate used different KV placement")
    config = result["config_snapshot"]
    if run.get("kv_placement", "sequence") != config.get(
        "resolved_config", {}
    ).get("kv_placement", "sequence"):
        return {"status": "quality_not_evaluated", "reason": "KV placement differs"}
    for tier in result["engine"]["tiers"]:
        name = f"{tier}_blocks"
        resolved_name = (
            "resolved_num_kvcache_blocks" if tier == "gpu"
            else f"resolved_num_{tier}_kvcache_blocks"
        )
        if run[name] != config[resolved_name]:
            return {"status": "quality_not_evaluated", "reason": f"{tier} capacity differs"}
    if engine > 1:
        active = result["engine"]["tiers"][:-1]
        resolved = result["config_snapshot"]["resolved_watermarks"]
        for tier in active:
            if run["watermarks"].get(tier) != [resolved[tier]["low"], resolved[tier]["high"]]:
                return {"status": "quality_not_evaluated", "reason": f"{tier} watermark differs"}
    if engine == 4:
        if config["cache"].get("runtime_overrides", {}).get("remote_mode") != "network":
            return {"status": "quality_not_evaluated", "reason": "Remote mode is not network"}
        if (run["remote_host"] != config["resolved_remote_host"]
                or run["remote_port"] != config["resolved_remote_port"]):
            return {"status": "quality_not_evaluated", "reason": "Remote endpoint differs"}
    if run["cache_stats"]["runner"]["io_errors"]:
        return {"status": "quality_failed", "reason": "Quality KV I/O errors"}
    if engine == 1:
        return {"status": "quality_pass", "reason": "Internal L1 reference", "mean_f1": run["mean_f1"]}
    with report_path.open("r", encoding="utf-8") as file:
        report = json.load(file)
    if report["count"] != len(run["examples"]) or report["candidate_mean_f1"] != run["mean_f1"]:
        raise ValueError("Quality report does not match run")
    if report["baseline_mean_f1"] != baseline["mean_f1"]:
        raise ValueError("Quality report does not match L1")
    if (report["noninferiority_margin"] != 0.05
            or report["bootstrap_repeats"] != 2000
            or report["bootstrap_seed"] != 20261020):
        raise ValueError("Quality report used different registered statistics")
    if not run["verify_kv_transfers"] or not run["target_tier_round_trip"]:
        return {"status": "quality_failed", "reason": "Target tier unverified"}
    passed = report["paired_bootstrap_lower_95"] > -0.05
    if report["internal_quality_gate_pass"] != passed:
        raise ValueError("Quality report gate disagrees with its F1 lower bound")
    return {
        "status": "quality_pass" if passed else "quality_failed",
        "reason": "SQuAD paired F1 versus internal L1",
        "mean_f1": run["mean_f1"],
        "paired_f1_lower_95": report["paired_bootstrap_lower_95"],
        "manifest_sha256": run["manifest_sha256"],
    }
