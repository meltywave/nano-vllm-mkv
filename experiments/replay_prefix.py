"""Diagnose tier-dependent logits under the same generated token prefixes."""

import argparse
import csv
import json
from pathlib import Path

from experiments.configuration import load_cache_engine
from experiments.metrics import output_sha256, tokenizer_files_sha256
from experiments.results import model_config_sha256, model_manifest_sha256, source_sha256
from experiments.tables import _local_result_path
from experiments.workload import load_workload, tokenize_workload


def reference_from_ledger(path, config_id="e1"):
    with path.open("r", newline="", encoding="utf-8") as file:
        matches = [row for row in csv.DictReader(file) if row["config_id"] == config_id]
    if len(matches) != 1 or not matches[0].get("result_path"):
        raise ValueError(f"Ledger must contain exactly one {config_id} result path")
    return _local_result_path(path.parent, matches[0]["result_path"])


def validate_reference(reference, model, workload, prompts):
    if reference["engine"]["level"] not in (1, 2, 3, 4):
        raise ValueError("Replay reference has an invalid engine level")
    expected_model = {
        "weights_sha256": model_manifest_sha256(model),
        "config_sha256": model_config_sha256(model),
        "tokenizer_sha256": tokenizer_files_sha256(model),
    }
    for name, value in expected_model.items():
        if reference["model"].get(name) != value:
            raise ValueError(f"Reference model {name} differs")
    if reference["workload"].get("sha256") != workload.sha256:
        raise ValueError("Reference workload differs")
    prompt_hash = output_sha256([{"token_ids": prompt} for prompt in prompts])
    if reference["workload"].get("input_token_ids_sha256") != prompt_hash:
        raise ValueError("Reference tokenized inputs differ")
    if reference["workload"].get("sampling") != workload.document["sampling"]:
        raise ValueError("Reference sampling settings differ")
    tokens = reference["measurements"][0]["output_token_ids"]
    length = int(workload.document["output_tokens"])
    if len(tokens) != len(prompts) or any(len(row) != length for row in tokens):
        raise ValueError("Reference output length differs from workload")
    return tokens


def summarize(rows):
    flat = [row for request in rows for row in request]
    gaps = sorted(row["gap"] for row in flat)
    return {
        "tokens": len(flat),
        "top_id_differs": sum(row["top_id"] != row["reference_id"] for row in flat),
        "positive_gap_count": sum(gap > 0 for gap in gaps),
        "max_gap": gaps[-1] if gaps else None,
        "p99_gap": gaps[min(len(gaps) - 1, int(0.99 * len(gaps)))] if gaps else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True,
                        help="Sweep ledger containing the selected reference config")
    parser.add_argument("--reference-config", default="e1",
                        help="Config ID whose free output supplies the replay prefix")
    parser.add_argument("--output", type=Path, required=True, help="New diagnostic JSON path")
    parser.add_argument("--gpu-blocks", type=int, required=True)
    parser.add_argument("--cpu-blocks", type=int, required=True)
    parser.add_argument("--ssd-blocks", type=int, required=True)
    parser.add_argument("--remote-blocks", type=int, required=True)
    parser.add_argument("--ssd-path", type=Path, required=True)
    parser.add_argument("--remote-host")
    parser.add_argument("--remote-port", type=int, default=19090)
    parser.add_argument("--low", type=float, default=0.5)
    parser.add_argument("--high", type=float, default=0.8)
    parser.add_argument("--verify-kv-transfers", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    if not 0 <= args.low < args.high <= 1:
        parser.error("Require 0 <= low < high <= 1")
    if args.engine == 4 and not args.remote_host:
        parser.error("Engine 4 requires --remote-host")
    for name in ("gpu_blocks", "cpu_blocks", "ssd_blocks", "remote_blocks"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")

    import torch
    from nanovllm import LLM, SamplingParams

    workload = load_workload(args.workload)
    reference_path = reference_from_ledger(args.ledger, args.reference_config)
    with reference_path.open("r", encoding="utf-8") as file:
        reference = json.load(file)
    current_source = source_sha256()
    if reference["software"]["source_sha256"] != current_source:
        raise ValueError("Reference source differs; rerun the free-generation sweep")
    kwargs, snapshot = load_cache_engine(args.engine)
    active = snapshot["tiers"][:-1]
    for tier in active:
        kwargs["kv_low_watermarks"][tier] = args.low
        kwargs["kv_high_watermarks"][tier] = args.high
    kwargs.update(
        num_kvcache_blocks=args.gpu_blocks,
        num_cpu_kvcache_blocks=args.cpu_blocks,
        num_ssd_kvcache_blocks=args.ssd_blocks,
        num_remote_kvcache_blocks=args.remote_blocks,
        ssd_kvcache_path=str(args.ssd_path.resolve()),
        remote_kvcache_port=args.remote_port,
        kv_verify_transfers=args.verify_kv_transfers,
        max_model_len=int(workload.document["max_model_len"]),
        max_num_seqs=int(workload.document.get("max_num_seqs", workload.document["num_seqs"])),
        max_num_batched_tokens=int(workload.document.get(
            "max_num_batched_tokens",
            int(workload.document["target_prompt_tokens"]) * int(workload.document["num_seqs"]),
        )),
    )
    if args.remote_host:
        kwargs["remote_kvcache_host"] = args.remote_host
    llm = LLM(str(args.model), **kwargs)
    try:
        for name, requested in (
            ("num_kvcache_blocks", args.gpu_blocks),
            ("num_cpu_kvcache_blocks", args.cpu_blocks if args.engine >= 2 else 0),
            ("num_ssd_kvcache_blocks", args.ssd_blocks if args.engine >= 3 else 0),
            ("num_remote_kvcache_blocks", args.remote_blocks if args.engine >= 4 else 0),
        ):
            if getattr(llm.config, name) != requested:
                raise ValueError(f"Resolved {name} differs from requested count")
        prompts, _ = tokenize_workload(workload, llm.tokenizer)
        tokens = validate_reference(reference, args.model, workload, prompts)
        original = reference["config_snapshot"]
        if original["resolved_num_kvcache_blocks"] != args.gpu_blocks:
            raise ValueError("Reference GPU KV block count differs")
        for name in ("max_model_len", "max_num_seqs", "max_num_batched_tokens", "kvcache_block_size"):
            if original["resolved_config"][name] != getattr(llm.config, name):
                raise ValueError(f"Reference {name} differs")
        if reference["model"]["dtype"] != str(llm.config.hf_config.dtype):
            raise ValueError("Reference model dtype differs")
        sampling = workload.document["sampling"]
        params = SamplingParams(
            temperature=float(sampling["temperature"]),
            ignore_eos=bool(sampling["ignore_eos"]),
            max_tokens=int(workload.document["output_tokens"]),
        )
        seed = int(workload.document["seed"])
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        outputs = llm.generate(
            prompts, params, use_tqdm=False, report_progress=True,
            replay_token_ids=tokens,
        )
        if [item["token_ids"] for item in outputs] != tokens:
            raise RuntimeError("Replay did not preserve the reference prefixes")
        rows = llm.last_generate_stats["replay_logits"]
        if any(len(row) != len(expected) for row, expected in zip(rows, tokens)):
            raise RuntimeError("Replay logit record count differs from reference tokens")
        result = {
            "diagnostic_only": True,
            "engine": args.engine,
            "source_sha256": current_source,
            "reference_result": str(reference_path.resolve()),
            "reference_config": args.reference_config,
            "reference_engine": reference["engine"]["level"],
            "reference_output_sha256": reference["measurements"][0]["output_sha256"],
            "verify_kv_transfers": args.verify_kv_transfers,
            "model": str(args.model.resolve()),
            "workload_sha256": workload.sha256,
            "input_token_ids_sha256": reference["workload"]["input_token_ids_sha256"],
            "watermarks": {tier: [args.low, args.high] for tier in active},
            "capacities": {name: getattr(args, name) for name in (
                "gpu_blocks", "cpu_blocks", "ssd_blocks", "remote_blocks"
            )},
            "summary": summarize(rows),
            "replay_logits": rows,
            "cache_stats": llm.cache_stats(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)
            file.write("\n")
        print(f"Diagnostic: {args.output}")
        print(json.dumps(result["summary"], ensure_ascii=False))
    finally:
        llm.exit()


if __name__ == "__main__":
    main()
