import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from experiments.configuration import REPO_ROOT
from experiments.remote_namespace import clear_remote_namespace, namespace_for_run
from experiments.results import model_config_sha256, model_manifest_sha256, source_sha256, write_error
from experiments.sweep_thresholds import (
    first_output_mismatch, required_tier_active, unexpected_tier_activity,
    validate_benchmark_interface,
)
from experiments.workload import load_workload
from experiments.validity import exact_token_rate, quality_evidence, require_quality_source


CONFIG_IDS = {1: "e1", 2: "e2-default", 3: "e3-default", 4: "e4-main"}
ACTIVE_TIERS = {1: (), 2: ("gpu",), 3: ("gpu", "cpu"), 4: ("gpu", "cpu", "ssd")}


def parse_args():
    parser = argparse.ArgumentParser(description="Run fixed L1-L4 KV ablation")
    parser.add_argument("--model", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--gpu-blocks", type=int, required=True)
    parser.add_argument("--cpu-blocks", type=int, required=True)
    parser.add_argument("--ssd-blocks", type=int, required=True)
    parser.add_argument("--remote-blocks", type=int, required=True)
    parser.add_argument("--ssd-path", required=True)
    parser.add_argument("--remote-host", required=True)
    parser.add_argument("--remote-port", type=int, default=19090)
    parser.add_argument("--gpu-watermark", default="0.70:0.90", metavar="LOW:HIGH")
    parser.add_argument("--cpu-watermark", default="0.70:0.90", metavar="LOW:HIGH")
    parser.add_argument("--ssd-watermark", default="0.70:0.90", metavar="LOW:HIGH")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measurement-repeats", type=int, default=2)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--disable-prefix-cache", action="store_true")
    parser.add_argument("--fp32-greedy-rerank", action="store_true",
                        help="Diagnostic only; compute the full LM head in FP32")
    parser.add_argument("--verify-kv-transfers", action="store_true",
                        help="Diagnostic only; compare each transferred KV block")
    parser.add_argument("--trace-logits", metavar="REQUEST:TOKEN",
                        help="Diagnostic only; save one full-vocabulary logit vector")
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument("--result-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--correctness-policy", choices=("strict", "task-quality"),
                        default="strict")
    parser.add_argument("--quality-dir", type=Path)
    parser.add_argument("--kv-placement", choices=("sequence", "block-hot"),
                        default="sequence")
    return parser.parse_args()


def command(args, engine):
    cmd = [
        sys.executable, "-B", str(REPO_ROOT / "bench.py"),
        "--engine", str(engine), "--model", args.model,
        "--workload", args.workload,
        "--gpu-blocks", str(args.gpu_blocks),
        "--cpu-blocks", str(args.cpu_blocks),
        "--ssd-blocks", str(args.ssd_blocks),
        "--remote-blocks", str(args.remote_blocks),
        "--ssd-path", args.ssd_path,
        "--warmup-runs", str(args.warmup_runs),
        "--measurement-repeats", str(args.measurement_repeats),
        "--result-dir", str(args.result_dir),
        "--config-id", CONFIG_IDS[engine],
    ]
    cmd.extend(("--kv-placement", getattr(args, "kv_placement", "sequence")))
    for tier in ACTIVE_TIERS[engine]:
        value = getattr(args, f"{tier}_watermark")
        try:
            low, high = (float(item) for item in value.split(":"))
        except ValueError as exc:
            raise ValueError(f"Invalid {tier} watermark: {value}") from exc
        if not 0 <= low < high <= 1:
            raise ValueError(f"Invalid {tier} watermark: {value}")
        cmd.extend(("--watermark", f"{tier}:{low}:{high}"))
    if engine == 4:
        namespace = namespace_for_run(
            getattr(args, "workload_sha256", ""), CONFIG_IDS[engine],
            getattr(args, "source_sha256", ""),
        )
        cmd.extend((
            "--remote-mode", "network", "--remote-host", args.remote_host,
            "--remote-port", str(args.remote_port),
            "--remote-namespace", namespace,
        ))
    if getattr(args, "enforce_eager", False):
        cmd.append("--enforce-eager")
    if getattr(args, "disable_prefix_cache", False):
        cmd.append("--disable-prefix-cache")
    if getattr(args, "fp32_greedy_rerank", False):
        cmd.append("--fp32-greedy-rerank")
    if getattr(args, "verify_kv_transfers", False):
        cmd.append("--verify-kv-transfers")
    if getattr(args, "trace_logits", None):
        cmd.extend(("--trace-logits", args.trace_logits))
    return cmd


def main():
    args = parse_args()
    args.record_logit_margins = False
    validate_benchmark_interface(args)
    if any(getattr(args, f"{tier}_blocks") <= 0 for tier in ("gpu", "cpu", "ssd", "remote")):
        raise ValueError("All four tier block counts must be positive")
    if args.warmup_runs < 0 or args.measurement_repeats <= 0 or args.timeout_s <= 0:
        raise ValueError("Invalid warmup, measurement, or timeout setting")
    if args.correctness_policy == "task-quality" and args.quality_dir is None:
        raise ValueError("--quality-dir is required for task-quality policy")
    output = args.result_dir / f"ablation_runs_{Path(args.model).name}_{Path(args.workload).stem}.csv"
    if output.exists() and not args.resume and not args.dry_run:
        raise FileExistsError(f"{output} exists; pass --resume or use a new --result-dir")
    fingerprints = {
        "source_sha256": source_sha256(),
        "workload_sha256": load_workload(args.workload).sha256,
    }
    if args.correctness_policy == "task-quality" and not args.dry_run:
        require_quality_source(args.quality_dir, fingerprints["source_sha256"])
    if not args.dry_run:
        fingerprints["model_weights_sha256"] = model_manifest_sha256(args.model)
        fingerprints["model_config_sha256"] = model_config_sha256(args.model)
    args.workload_sha256 = fingerprints.get("workload_sha256", "")
    args.source_sha256 = fingerprints.get("source_sha256", "")
    rows = []
    if args.resume and output.exists():
        with output.open("r", newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
    previous = {row["config_id"]: row for row in rows}
    baseline = None
    if previous.get("e1", {}).get("status") == "ok" and not previous["e1"].get("result_path"):
        raise ValueError("GPU-only baseline is marked ok without a result path")
    if previous.get("e1", {}).get("result_path"):
        if any(previous["e1"].get(name) != value for name, value in fingerprints.items()):
            raise ValueError("GPU-only baseline has a different source, model, or workload")
        if previous["e1"].get("status") != "ok":
            raise ValueError("GPU-only baseline is not valid; use a new result directory")
        with Path(previous["e1"]["result_path"]).open("r", encoding="utf-8") as file:
            baseline = json.load(file)
    for engine in CONFIG_IDS:
        config_id = CONFIG_IDS[engine]
        cmd = command(args, engine)
        if args.dry_run:
            print(config_id, " ".join(cmd))
            continue
        command_json = json.dumps(cmd)
        old = previous.get(config_id)
        if old and old.get("command") != command_json:
            raise ValueError(
                f"{config_id} has different run arguments in {output}; use a new result directory"
            )
        if old and any(old.get(name) != value for name, value in fingerprints.items()):
            raise ValueError(
                f"{config_id} has a different source, model, or workload in {output}; "
                "use a new result directory"
            )
        if old and old.get("correctness_policy", "strict") != args.correctness_policy:
            raise ValueError("Correctness policy differs in existing ledger; use a new result directory")
        quality_directory = str(args.quality_dir.resolve()) if args.quality_dir else ""
        if old and old.get("quality_directory", "") != quality_directory:
            raise ValueError("Quality directory differs in existing ledger; use a new result directory")
        if old and old["status"] in {"ok", "inactive", "quality_not_evaluated", "quality_failed"}:
            print(f"{config_id}: skipped (already completed)")
            continue
        try:
            if engine == 4:
                clear_remote_namespace(args.remote_host, args.remote_port,
                                       cmd[cmd.index("--remote-namespace") + 1])
            process = subprocess.run(
                cmd, cwd=REPO_ROOT, capture_output=True, text=True,
                timeout=args.timeout_s,
            )
            status = "ok" if process.returncode == 0 else "error"
            stdout, stderr = process.stdout, process.stderr
        except subprocess.TimeoutExpired as exc:
            status = "timeout"
            stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            stderr += f"\nTimeout after {args.timeout_s}s"
        except Exception as exc:
            status, stdout, stderr = "error", "", f"Remote setup failed: {exc}"
        finally:
            if engine == 4:
                try:
                    clear_remote_namespace(args.remote_host, args.remote_port,
                                           cmd[cmd.index("--remote-namespace") + 1])
                except Exception as exc:
                    stderr += f"\nRemote cleanup failed: {exc}"
                    if status == "ok":
                        status = "error"
        result_path = next(
            (line.removeprefix("Result: ").strip() for line in stdout.splitlines()
             if line.startswith("Result: ")),
            "",
        )
        if status == "ok" and not result_path:
            status = "error"
            stderr = "bench.py exited successfully without reporting a result path"
        correctness_mismatch = ""
        exact_rate = ""
        quality = {"status": "", "reason": ""}
        if result_path:
            with Path(result_path).open("r", encoding="utf-8") as file:
                result = json.load(file)
            if not result["summary"]["correctness"]["repeat_outputs_match"] or any(
                measurement.get("io_errors", 0) for measurement in result["measurements"]
            ):
                status = "invalid"
            elif unexpected_tier_activity(result, engine):
                status = "miscalibrated"
            elif not required_tier_active(result, engine):
                status = "inactive"
            if engine == 1:
                baseline = result
            elif baseline is not None:
                mismatch = first_output_mismatch(result, baseline)
                if mismatch is not None:
                    correctness_mismatch = mismatch
                    if args.correctness_policy == "strict":
                        status = "invalid"
                exact_rate = exact_token_rate(result, baseline)
            if engine == 1:
                exact_rate = 1.0
            if args.correctness_policy == "task-quality":
                quality = quality_evidence(result, args.quality_dir)
                if status in {"ok", "inactive"} and quality["status"] != "quality_pass":
                    status = quality["status"]
        row = {
            "engine": engine, "config_id": config_id, "status": status,
            "correctness_policy": args.correctness_policy,
            "quality_directory": quality_directory,
            "result_path": result_path, "stderr_tail": stderr[-2000:],
            "correctness_mismatch": correctness_mismatch,
            "exact_token_rate": exact_rate,
            "quality_status": quality["status"],
            "quality_reason": quality["reason"],
            "quality_mean_f1": quality.get("mean_f1", ""),
            "quality_f1_lower_95": quality.get("paired_f1_lower_95", ""),
            "quality_manifest_sha256": quality.get("manifest_sha256", ""),
            "command": command_json,
            **fingerprints,
        }
        if old:
            rows[rows.index(old)] = row
        else:
            rows.append(row)
        previous[config_id] = row
        if status in {"error", "timeout"}:
            write_error(args.result_dir, f"{Path(args.workload).stem}-{config_id}",
                        status, cmd, stderr, stdout)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"{config_id}: {status} {result_path}", flush=True)
        if engine == 1 and status == "invalid":
            raise RuntimeError("GPU-only baseline failed repeat or I/O validation; ablation stopped")
    if not args.dry_run:
        print(output.resolve())


if __name__ == "__main__":
    main()
