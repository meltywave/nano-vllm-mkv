import json
import platform
import statistics
import subprocess
from datetime import datetime
from pathlib import Path

from experiments.configuration import REPO_ROOT
from experiments.metrics import percentile


DEFAULT_RESULT_DIR = REPO_ROOT / "results"


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


def summarize(measurements):
    latencies = [item["latency_s"] for item in measurements]
    throughputs = [item["total_tokens_per_s"] for item in measurements]
    output_hashes = {item["output_sha256"] for item in measurements}
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
    with (run_dir / "stdout.log").open("w", encoding="utf-8") as file:
        file.write(
            f"engine={result['engine']['level']}\n"
            f"workload={result['workload']['name']}\n"
            f"latency_mean_s={result['summary']['latency_s']['mean']}\n"
            "throughput_mean_tokens_s="
            f"{result['summary']['total_tokens_per_s']['mean']}\n"
        )
    return run_dir / "result.json"


def software_metadata(torch_module):
    return {
        "git_commit": git_revision(),
        "python": platform.python_version(),
        "torch": torch_module.__version__,
        "cuda": torch_module.version.cuda,
        "platform": platform.platform(),
    }
