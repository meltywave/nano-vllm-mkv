import argparse
import csv
import json
import math
from pathlib import Path

from experiments.tables import _local_result_path


def first_mismatches(reference, candidate):
    ref_outputs = reference["measurements"][0]["output_token_ids"]
    rows = []
    for repeat, measurement in enumerate(candidate["measurements"], 1):
        outputs = measurement["output_token_ids"]
        if len(outputs) != len(ref_outputs):
            raise ValueError("Reference and candidate have different request counts")
        for request, (left, right) in enumerate(zip(ref_outputs, outputs)):
            token = next(
                (index for index, pair in enumerate(zip(left, right))
                 if pair[0] != pair[1]), None
            )
            if token is None:
                if len(left) == len(right):
                    continue
                token = min(len(left), len(right))
            margins = measurement.get("greedy_logit_margins")
            ref_margins = reference["measurements"][0].get("greedy_logit_margins")
            rows.append({
                "repeat": repeat,
                "request": request,
                "token": token,
                "reference_id": left[token] if token < len(left) else None,
                "candidate_id": right[token] if token < len(right) else None,
                "reference_margin": (
                    ref_margins[request][token]
                    if ref_margins and token < len(ref_margins[request]) else None
                ),
                "candidate_margin": (
                    margins[request][token]
                    if margins and token < len(margins[request]) else None
                ),
            })
    return rows


def compare_logit_traces(reference, candidate):
    left = reference["measurements"][0].get("trace_logits")
    right = candidate["measurements"][0].get("trace_logits")
    if left is None or right is None:
        return None
    if len(left) != len(right):
        raise ValueError("Traced logit vectors have different vocabulary sizes")
    differences = [a - b for a, b in zip(left, right)]
    top_ids = sorted(
        range(len(left)), key=lambda index: max(left[index], right[index]),
        reverse=True,
    )[:4]
    return {
        "vocab_size": len(left),
        "max_abs": max(map(abs, differences)),
        "rms": math.sqrt(math.fsum(value * value for value in differences) / len(left)),
        "reference_top_id": max(range(len(left)), key=left.__getitem__),
        "candidate_top_id": max(range(len(right)), key=right.__getitem__),
        "top_candidates": [
            {
                "token_id": token_id,
                "reference": left[token_id],
                "candidate": right[token_id],
            }
            for token_id in top_ids
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Compare Qwen output tokens across cache tiers")
    parser.add_argument("reference", type=Path, nargs="?")
    parser.add_argument("candidate", type=Path, nargs="?")
    parser.add_argument("--ledger", type=Path, help="Sweep CSV containing result paths")
    parser.add_argument("--reference-config", default="e1")
    parser.add_argument("--candidate-config")
    args = parser.parse_args()
    if args.ledger:
        if args.reference or args.candidate or not args.candidate_config:
            parser.error("--ledger requires --candidate-config and no positional paths")
        with args.ledger.open("r", newline="", encoding="utf-8") as file:
            entries = {row["config_id"]: row for row in csv.DictReader(file)}
        for config_id in (args.reference_config, args.candidate_config):
            if config_id not in entries or not entries[config_id].get("result_path"):
                parser.error(f"No result path for {config_id!r} in {args.ledger}")
        root = args.ledger.parent
        args.reference = _local_result_path(
            root, entries[args.reference_config]["result_path"]
        )
        args.candidate = _local_result_path(
            root, entries[args.candidate_config]["result_path"]
        )
    elif not args.reference or not args.candidate:
        parser.error("Provide two result.json paths or --ledger and --candidate-config")
    with args.reference.open("r", encoding="utf-8") as file:
        reference = json.load(file)
    with args.candidate.open("r", encoding="utf-8") as file:
        candidate = json.load(file)
    for name in ("weights_sha256", "config_sha256", "tokenizer_sha256"):
        if reference["model"].get(name) != candidate["model"].get(name):
            raise ValueError(f"Model {name} differs")
    for name in ("sha256", "input_token_ids_sha256"):
        if reference["workload"].get(name) != candidate["workload"].get(name):
            raise ValueError(f"Workload {name} differs")
    mismatches = first_mismatches(reference, candidate)
    print(f"first mismatches: {len(mismatches)} request/repeat pairs")
    for row in mismatches:
        print(" ".join(f"{key}={value}" for key, value in row.items()))
    trace = compare_logit_traces(reference, candidate)
    if trace is not None:
        print("logit trace: " + " ".join(
            f"{key}={value}" for key, value in trace.items()
            if key != "top_candidates"
        ))
        for candidate in trace["top_candidates"]:
            print("top candidate: " + " ".join(
                f"{key}={value}" for key, value in candidate.items()
            ))


if __name__ == "__main__":
    main()
