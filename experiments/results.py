import json
import hashlib
from importlib.metadata import PackageNotFoundError, version
import platform
import re
import statistics
import subprocess
from datetime import datetime
from pathlib import Path

from experiments.configuration import REPO_ROOT
from experiments.metrics import percentile


DEFAULT_RESULT_DIR = REPO_ROOT / "results"


def write_error(result_root, config_id, status, command, stderr, stdout=""):
    safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", config_id)
    directory = Path(result_root) / "errors"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{safe_id}.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "config_id": config_id,
                "status": status,
                "command": command,
                "stderr_tail": stderr[-4000:],
                "stdout_tail": stdout[-4000:],
            },
            file, ensure_ascii=False, indent=2,
        )
        file.write("\n")
    return path


def git_revision():
    try:
        process = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return process.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def source_sha256():
    digest = hashlib.sha256()
    paths = [REPO_ROOT / "bench.py"]
    for package in ("nanovllm", "experiments"):
        paths.extend((REPO_ROOT / package).rglob("*.py"))
    for path in sorted(paths):
        digest.update(path.relative_to(REPO_ROOT).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def model_manifest_sha256(path):
    root = Path(path)
    digest = hashlib.sha256()
    files = sorted(root.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors weight files found in {path}")
    for item in files:
        digest.update(item.name.encode("utf-8"))
        with item.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def model_config_sha256(path):
    config = Path(path) / "config.json"
    if not config.is_file():
        raise FileNotFoundError(f"No model config.json found in {path}")
    return hashlib.sha256(config.read_bytes()).hexdigest()


def summarize(measurements):
    latencies = [item["latency_s"] for item in measurements]
    throughputs = [item["total_tokens_per_s"] for item in measurements]
    output_hashes = {item["output_sha256"] for item in measurements}
    request_timings = [
        timing
        for measurement in measurements
        for timing in measurement.get("request_timings", [])
    ]
    def timing_stats(name):
        values = [
            item[name] for item in request_timings if item.get(name) is not None
        ]
        return {
            "count": len(values),
            "p50": percentile(values, 0.50),
            "p95": percentile(values, 0.95),
        }
    return {
        "measurement_repeats": len(measurements),
        "latency_s": {
            "mean": statistics.fmean(latencies),
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
        },
        "total_tokens_per_s": {
            "mean": statistics.fmean(throughputs),
            "std": statistics.stdev(throughputs) if len(throughputs) > 1 else 0.0,
        },
        "output_tokens_per_s": {
            "mean": statistics.fmean(
                item["output_tokens_per_s"] for item in measurements
            ),
        },
        "requests_per_s": {
            "mean": statistics.fmean(
                item["requests_per_s"] for item in measurements
            ),
        },
        "request_latency_ms": {
            "ttft": timing_stats("ttft_ms"),
            "tpot": timing_stats("tpot_ms"),
            "e2e": timing_stats("e2e_ms"),
        },
        "gpu_peak_allocated_bytes": {
            "max": max(item["gpu_peak_allocated_bytes"] for item in measurements)
        },
        "correctness": {
            "repeat_outputs_match": len(output_hashes) == 1,
            "output_sha256": next(iter(output_hashes)) if len(output_hashes) == 1 else None,
            "matches_engine_1": None,
        },
    }


def write_result(result, result_root: str | Path | None = None):
    root = Path(result_root).resolve() if result_root else DEFAULT_RESULT_DIR
    model_id = Path(result["model"]["path"]).name or "model"
    revision = result["software"]["git_commit"]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_id = f"{timestamp}-{revision}"
    result["run_id"] = run_id
    run_dir = (
        root
        / result["workload"]["name"]
        / f"engine-{result['engine']['level']}"
        / model_id
        / f"run-{run_id}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "result.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write("\n")
    with (run_dir / "config.snapshot.json").open("w", encoding="utf-8") as file:
        json.dump(result["config_snapshot"], file, ensure_ascii=False, indent=2)
        file.write("\n")
    with (run_dir / "run.summary.txt").open("w", encoding="utf-8") as file:
        file.write(
            f"engine={result['engine']['level']}\n"
            f"workload={result['workload']['name']}\n"
            f"latency_mean_s={result['summary']['latency_s']['mean']}\n"
            "output_tokens_per_s="
            f"{result['summary']['output_tokens_per_s']['mean']}\n"
            "input_plus_output_tokens_per_s="
            f"{result['summary']['total_tokens_per_s']['mean']}\n"
        )
    return run_dir / "result.json"


def software_metadata(torch_module):
    dependencies = {}
    for package in ("transformers", "triton", "flash-attn", "safetensors"):
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = None
    return {
        "git_commit": git_revision(),
        "source_sha256": source_sha256(),
        "python": platform.python_version(),
        "torch": torch_module.__version__,
        "cuda": torch_module.version.cuda,
        "platform": platform.platform(),
        "dependencies": dependencies,
    }
