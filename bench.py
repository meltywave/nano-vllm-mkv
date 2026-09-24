import argparse
from pathlib import Path

from experiments.results import DEFAULT_RESULT_DIR
from experiments.runners.nano_runner import run_nano_experiment


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a structured nano-vLLM multi-level KV cache experiment."
    )
    parser.add_argument("--engine", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--model", required=True, help="Local model directory")
    parser.add_argument("--workload", required=True, help="Workload manifest JSON")
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
        help=f"Result root (default: {DEFAULT_RESULT_DIR})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    result_path, result = run_nano_experiment(
        args.engine,
        args.model,
        args.workload,
        args.result_dir,
    )
    summary = result["summary"]
    print(f"Result: {result_path}")
    print(f"Mean latency: {summary['latency_s']['mean']:.3f}s")
    print(
        "Mean throughput: "
        f"{summary['total_tokens_per_s']['mean']:.2f} tokens/s"
    )


if __name__ == "__main__":
    main()
