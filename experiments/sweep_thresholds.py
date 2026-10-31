import argparse
import csv
import inspect
import itertools
import json
import subprocess
import sys
from pathlib import Path

from experiments.configuration import REPO_ROOT
from experiments.remote_namespace import clear_remote_namespace, namespace_for_run
from experiments.results import model_config_sha256, model_manifest_sha256, source_sha256, write_error
from experiments.validity import exact_token_rate, quality_evidence, require_quality_source
from experiments.workload import load_workload


ACTIVE_TIERS = {1: (), 2: ("gpu",), 3: ("gpu", "cpu"), 4: ("gpu", "cpu", "ssd")}


def parse_args():
    parser = argparse.ArgumentParser(description="Run complete KV watermark grid")
    parser.add_argument("--model", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--engines", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--low", type=float, nargs="+", default=[0.50, 0.70])
    parser.add_argument("--high", type=float, nargs="+", default=[0.80, 0.95])
    parser.add_argument("--gpu-blocks", type=int, required=True)
    parser.add_argument("--cpu-blocks", type=int, required=True)
    parser.add_argument("--ssd-blocks", type=int, required=True)
    parser.add_argument("--remote-blocks", type=int, required=True)
    parser.add_argument("--ssd-path", required=True)
    parser.add_argument("--remote-host")
    parser.add_argument("--remote-port", type=int, default=19090)
    parser.add_argument("--remote-mode", choices=("network", "loopback"), default="network")
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--measurement-repeats", type=int, default=2)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--disable-prefix-cache", action="store_true")
    parser.add_argument("--record-logit-margins", action="store_true",
                        help="Diagnostic only; changes measured overhead")
    parser.add_argument("--fp32-greedy-rerank", action="store_true",
                        help="Diagnostic only; compute the full LM head in FP32")
    parser.add_argument("--verify-kv-transfers", action="store_true",
                        help="Diagnostic only; compare each transferred KV block")
    parser.add_argument("--trace-logits", metavar="REQUEST:TOKEN",
                        help="Diagnostic only; save one full-vocabulary logit vector")
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument("--continue-on-timeout", action="store_true",
                        help="Keep running later configurations after a timeout")
    parser.add_argument("--stop-on-invalid", action="store_true",
                        help="Stop after the first invalid correctness result")
    parser.add_argument("--correctness-policy", choices=("strict", "task-quality"),
                        default="strict")
    parser.add_argument("--quality-dir", type=Path)
    parser.add_argument("--kv-placement", choices=("sequence", "block-hot"),
                        default="sequence")
    parser.add_argument("--result-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip completed config IDs")
    parser.add_argument("--config-prefix", default="", help="Prefix for config IDs in this run")
    return parser.parse_args()


def configs(engines, lows, highs):
    pairs = list(dict.fromkeys(
        (low, high) for low in lows for high in highs if 0 <= low < high <= 1
    ))
    if not pairs:
        raise ValueError("No valid low/high pairs")
    for engine in engines:
        if engine not in ACTIVE_TIERS:
            raise ValueError(f"Unknown engine {engine}")
        for values in itertools.product(pairs, repeat=len(ACTIVE_TIERS[engine])):
            yield engine, dict(zip(ACTIVE_TIERS[engine], values))


def required_tier_active(result, engine):
    expectations = result.get("workload", {}).get("required_tiers", {})
    tier = expectations.get(str(engine), {2: "cpu", 3: "ssd", 4: "remote"}.get(engine))
    if tier in {None, "none"}:
        return True
    if tier is None:
        return True
    measurements = result.get("measurements", [])
    return bool(measurements) and all(
        any(
            direction.endswith(f"_to_{tier}") and count > 0
            for direction, count in measurement.get("transfer_counts", {}).items()
        )
        and measurement.get("transfer_counts", {}).get(f"{tier}_to_gpu", 0) > 0
        for measurement in measurements
    )


def unexpected_tier_activity(result, engine):
    forbidden = result.get("workload", {}).get("forbidden_tiers", {}).get(str(engine), [])
    return any(
        count > 0 and any(
            direction.endswith(f"_to_{tier}") or direction.startswith(f"{tier}_to_")
            for tier in forbidden
        )
        for measurement in result.get("measurements", [])
        for direction, count in measurement.get("transfer_counts", {}).items()
    )


def first_output_mismatch(result, baseline):
    expected = baseline["measurements"][0]["output_token_ids"]
    for repeat, measurement in enumerate(result["measurements"], 1):
        actual = measurement["output_token_ids"]
        if len(actual) != len(expected):
            return f"repeat={repeat} expected_requests={len(expected)} actual_requests={len(actual)}"
        for request, (reference, output) in enumerate(zip(expected, actual)):
            for token, (left, right) in enumerate(zip(reference, output)):
                if left != right:
                    return (
                        f"repeat={repeat} request={request} token={token} "
                        f"expected={left} actual={right}"
                    )
            if len(reference) != len(output):
                return (
                    f"repeat={repeat} request={request} "
                    f"expected_tokens={len(reference)} actual_tokens={len(output)}"
                )
    return None


def validate_benchmark_interface(args):
    from experiments.runners.nano_runner import run_nano_experiment

    parameters = inspect.signature(run_nano_experiment).parameters
    required = set()
    if args.fp32_greedy_rerank:
        required.add("fp32_greedy_rerank")
    if args.record_logit_margins:
        required.add("record_logit_margins")
    if args.enforce_eager:
        required.add("enforce_eager")
    if args.disable_prefix_cache:
        required.add("disable_prefix_cache")
    if args.verify_kv_transfers:
        required.add("verify_kv_transfers")
    if getattr(args, "trace_logits", None):
        required.add("trace_logits")
    missing = sorted(required - parameters.keys())
    if missing:
        raise RuntimeError(
            "bench.py and experiments/runners/nano_runner.py are out of sync; "
            f"missing runner parameters: {', '.join(missing)}. "
            "Copy the complete repository before running the sweep."
        )


def main():
    args = parse_args()
    validate_benchmark_interface(args)
    for name in ("gpu_blocks", "cpu_blocks", "ssd_blocks", "remote_blocks"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if args.gpu_blocks == 0:
        raise ValueError("--gpu-blocks must be positive")
    if args.measurement_repeats <= 0 or args.warmup_runs < 0 or args.timeout_s <= 0:
        raise ValueError("Invalid repeat or timeout settings")
    if 4 in args.engines and not args.remote_host and not args.dry_run:
        raise ValueError("--remote-host is required when sweeping engine 4")
    if args.correctness_policy == "task-quality" and args.quality_dir is None:
        raise ValueError("--quality-dir is required for task-quality policy")
    workload_tag = Path(args.workload).stem
    model_tag = Path(args.model).name
    prefix_tag = args.config_prefix or "default"
    output = args.result_dir / f"threshold_sweep_{model_tag}_{workload_tag}_{prefix_tag}.csv"
    if output.exists() and not args.resume and not args.dry_run:
        raise FileExistsError(
            f"{output} already exists; use --resume or a new --result-dir"
        )
    fingerprints = {
        "source_sha256": source_sha256(),
        "workload_sha256": load_workload(args.workload).sha256,
    }
    if args.correctness_policy == "task-quality" and not args.dry_run:
        require_quality_source(args.quality_dir, fingerprints["source_sha256"])
    if not args.dry_run:
        fingerprints["model_weights_sha256"] = model_manifest_sha256(args.model)
        fingerprints["model_config_sha256"] = model_config_sha256(args.model)
    rows = []
    previous = {}
    baseline = None
    if args.resume and output.exists():
        with output.open("r", newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        previous = {row["config_id"]: row for row in rows}
        baseline_entry = previous.get(args.config_prefix + "e1")
        if baseline_entry and baseline_entry.get("status") == "ok" and not baseline_entry.get("result_path"):
            raise ValueError("GPU-only baseline is marked ok without a result path")
        if baseline_entry and baseline_entry.get("result_path"):
            if any(baseline_entry.get(name) != value for name, value in fingerprints.items()):
                raise ValueError("GPU-only baseline has a different source, model, or workload")
            if baseline_entry.get("status") != "ok":
                raise ValueError("GPU-only baseline is not valid; use a new result directory")
            with Path(baseline_entry["result_path"]).open("r", encoding="utf-8") as file:
                baseline = json.load(file)
    if not args.dry_run and baseline is None and 1 not in args.engines:
        raise ValueError("Sweep requires engine 1 or a valid resumed GPU-only baseline")
    for engine, watermarks in configs(args.engines, args.low, args.high):
        config_id = "e" + str(engine) + (
            "-" + "-".join(
                f"{tier}{low!r}_{high!r}"
                for tier, (low, high) in watermarks.items()
            ) if watermarks else ""
        )
        config_id = args.config_prefix + config_id
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
            "--config-id", config_id,
        ]
        cmd.extend(("--kv-placement", args.kv_placement))
        for tier, (low, high) in watermarks.items():
            cmd.extend(["--watermark", f"{tier}:{low}:{high}"])
        remote_namespace = None
        if engine == 4:
            remote_namespace = namespace_for_run(
                fingerprints.get("workload_sha256", ""), config_id,
                fingerprints.get("source_sha256", ""),
            )
            cmd.extend(["--remote-mode", args.remote_mode, "--remote-host", args.remote_host or "REMOTE_HOST", "--remote-port", str(args.remote_port)])
            cmd.extend(["--remote-namespace", remote_namespace])
        if args.enforce_eager:
            cmd.append("--enforce-eager")
        if args.disable_prefix_cache:
            cmd.append("--disable-prefix-cache")
        if args.record_logit_margins:
            cmd.append("--record-logit-margins")
        if args.fp32_greedy_rerank:
            cmd.append("--fp32-greedy-rerank")
        if args.verify_kv_transfers:
            cmd.append("--verify-kv-transfers")
        if args.trace_logits:
            cmd.extend(["--trace-logits", args.trace_logits])
        if args.dry_run:
            print(config_id)
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
            if remote_namespace:
                clear_remote_namespace(args.remote_host, args.remote_port, remote_namespace)
            process = subprocess.run(
                cmd, cwd=REPO_ROOT, capture_output=True, text=True,
                timeout=args.timeout_s,
            )
            returncode, stdout, stderr = (
                process.returncode, process.stdout, process.stderr
            )
        except subprocess.TimeoutExpired as exc:
            returncode = -1
            stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            stderr += f"\nTimeout after {args.timeout_s}s"
        except Exception as exc:
            returncode, stdout, stderr = 1, "", f"Remote setup failed: {exc}"
        finally:
            if remote_namespace:
                try:
                    clear_remote_namespace(args.remote_host, args.remote_port, remote_namespace)
                except Exception as exc:
                    stderr += f"\nRemote cleanup failed: {exc}"
                    if returncode == 0:
                        returncode = 1
        result_path = ""
        for line in stdout.splitlines():
            if line.startswith("Result: "):
                result_path = line.removeprefix("Result: ").strip()
                break
        row = {
            "config_id": config_id, "engine": engine,
            "correctness_policy": args.correctness_policy,
            "quality_directory": quality_directory,
            "watermarks": json.dumps(watermarks, sort_keys=True),
            "status": "ok" if returncode == 0 else ("timeout" if returncode == -1 else "error"),
            "result_path": result_path,
            "stderr_tail": stderr[-2000:],
            "correctness_mismatch": "",
            "exact_token_rate": "",
            "quality_status": "",
            "quality_reason": "",
            "quality_mean_f1": "",
            "quality_f1_lower_95": "",
            "quality_manifest_sha256": "",
            "command": command_json,
            **fingerprints,
        }
        if result_path:
            with Path(result_path).open("r", encoding="utf-8") as file:
                result = json.load(file)
            if unexpected_tier_activity(result, engine):
                row["status"] = "miscalibrated"
            elif not required_tier_active(result, engine):
                row["status"] = "inactive"
            if not result["summary"]["correctness"]["repeat_outputs_match"]:
                row["status"] = "invalid"
            if any(measurement.get("io_errors", 0) for measurement in result["measurements"]):
                row["status"] = "invalid"
            if engine == 1:
                baseline = result
            elif baseline is not None and row["status"] not in {"error", "timeout"}:
                mismatch = first_output_mismatch(result, baseline)
                if mismatch is not None:
                    row["correctness_mismatch"] = mismatch
                    if args.correctness_policy == "strict":
                        row["status"] = "invalid"
                row["exact_token_rate"] = exact_token_rate(result, baseline)
            if engine == 1:
                row["exact_token_rate"] = 1.0
            if args.correctness_policy == "task-quality":
                quality = quality_evidence(result, args.quality_dir)
                row["quality_status"] = quality["status"]
                row["quality_reason"] = quality["reason"]
                row["quality_mean_f1"] = quality.get("mean_f1", "")
                row["quality_f1_lower_95"] = quality.get("paired_f1_lower_95", "")
                row["quality_manifest_sha256"] = quality.get("manifest_sha256", "")
                if row["status"] in {"ok", "inactive"} and quality["status"] != "quality_pass":
                    row["status"] = quality["status"]
        if old:
            rows[rows.index(old)] = row
        else:
            rows.append(row)
        previous[config_id] = row
        if row["status"] in {"error", "timeout"}:
            write_error(args.result_dir, f"{workload_tag}-{config_id}",
                        row["status"], cmd, stderr, stdout)
        print(f"{config_id}: {row['status']} {result_path}", flush=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        if engine == 1 and row["status"] == "invalid":
            raise RuntimeError("GPU-only baseline failed repeat or I/O validation; sweep stopped")
        if row["status"] in {"invalid", "quality_failed"} and args.stop_on_invalid:
            raise RuntimeError(
                "Sweep stopped after invalid output; inspect correctness_mismatch"
            )
        if row["status"] == "timeout" and not args.continue_on_timeout:
            print("Sweep stopped after timeout; inspect the error artifact before --resume")
            break


if __name__ == "__main__":
    main()
