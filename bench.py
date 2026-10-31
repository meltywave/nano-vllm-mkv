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
    parser.add_argument(
        "--remote-mode",
        choices=("loopback", "network"),
        help="Remote cache mode; network requires --remote-host.",
    )
    parser.add_argument("--remote-host", help="Remote cache server IPv4/hostname")
    parser.add_argument("--remote-namespace", help="Remote RAM namespace for this run")
    parser.add_argument("--remote-port", type=int, help="Remote cache server port")
    parser.add_argument(
        "--remote-timeout", type=float, help="Remote cache request timeout in seconds"
    )
    parser.add_argument(
        "--ssd-path", type=Path, help="Absolute path for the local SSD KV cache"
    )
    parser.add_argument("--gpu-blocks", type=int, help="Fixed GPU KV block count")
    parser.add_argument("--cpu-blocks", type=int, help="Fixed CPU KV block count")
    parser.add_argument("--ssd-blocks", type=int, help="Fixed SSD KV block count")
    parser.add_argument(
        "--remote-blocks", type=int, help="Fixed Remote KV block count"
    )
    parser.add_argument("--warmup-runs", type=int, help="Warmup repetitions")
    parser.add_argument("--config-id", help="Stable identifier for table grouping")
    parser.add_argument("--kv-placement", choices=("sequence", "block-hot"),
                        default="sequence", help="KV placement granularity")
    parser.add_argument("--record-logit-margins", action="store_true",
                        help="Store greedy top-1 versus top-2 logit margins for diagnosis")
    parser.add_argument("--fp32-greedy-rerank", action="store_true",
                        help="Diagnose BF16 ties with a full-vocabulary FP32 LM head")
    parser.add_argument("--verify-kv-transfers", action="store_true",
                        help="Read back every swapped KV block and compare bytes")
    parser.add_argument("--trace-logits", metavar="REQUEST:TOKEN",
                        help="Diagnostic only; save one full-vocabulary logit vector")
    parser.add_argument("--enforce-eager", action="store_true",
                        help="Run decode without CUDA graphs")
    parser.add_argument("--disable-prefix-cache", action="store_true",
                        help="Disable GPU prefix block reuse")
    parser.add_argument(
        "--measurement-repeats", type=int, help="Measured repetitions"
    )
    parser.add_argument(
        "--watermark", action="append", default=[], metavar="TIER:LOW:HIGH",
        help="Override one active tier watermark, e.g. gpu:0.5:0.8",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    result_path, result = run_nano_experiment(
        args.engine,
        args.model,
        args.workload,
        args.result_dir,
        remote_mode=args.remote_mode,
        remote_host=args.remote_host,
        remote_namespace=args.remote_namespace,
        remote_port=args.remote_port,
        remote_timeout=args.remote_timeout,
        ssd_path=args.ssd_path,
        gpu_blocks=args.gpu_blocks,
        cpu_blocks=args.cpu_blocks,
        ssd_blocks=args.ssd_blocks,
        remote_blocks=args.remote_blocks,
        warmup_runs=args.warmup_runs,
        measurement_repeats=args.measurement_repeats,
        watermark_overrides=args.watermark,
        config_id=args.config_id,
        kv_placement=args.kv_placement,
        record_logit_margins=args.record_logit_margins,
        fp32_greedy_rerank=args.fp32_greedy_rerank,
        verify_kv_transfers=args.verify_kv_transfers,
        trace_logits=args.trace_logits,
        enforce_eager=args.enforce_eager,
        disable_prefix_cache=args.disable_prefix_cache,
    )
    summary = result["summary"]
    print(f"Result: {result_path}")
    print(f"Mean latency: {summary['latency_s']['mean']:.3f}s")
    print(
        "Mean output throughput: "
        f"{summary['output_tokens_per_s']['mean']:.2f} generated tokens/s"
    )


if __name__ == "__main__":
    main()
