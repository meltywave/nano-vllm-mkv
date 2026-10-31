"""Compare bidirectional fixed-prefix replays without changing sweep validity."""

import argparse
import json
import math
import struct
from pathlib import Path

from experiments.replay_prefix import reference_from_ledger


def bf16_ulp_below(value):
    if not math.isfinite(value):
        raise ValueError("Non-finite replay logit")
    bits = struct.unpack(">I", struct.pack(">f", value))[0]
    if bits & 0xFFFF:
        raise ValueError("Replay logit is not a BF16 value")
    bf16_bits = bits >> 16
    if bf16_bits in (0, 0x8000):
        predecessor = 0x8001
    elif bf16_bits & 0x8000:
        predecessor = bf16_bits + 1
    else:
        predecessor = bf16_bits - 1
    previous = struct.unpack(">f", struct.pack(">I", predecessor << 16))[0]
    return value - previous


def inspect_replay(replay, expected_tokens):
    rows = replay["replay_logits"]
    if len(rows) != len(expected_tokens):
        raise ValueError("Replay request count differs")
    total = top_differs = positive = exceeded = 0
    max_gap = max_ratio = 0.0
    first_exceeded = None
    for request, (recorded, expected) in enumerate(zip(rows, expected_tokens)):
        if len(recorded) != len(expected):
            raise ValueError(f"Replay request {request} has the wrong length")
        for index, (row, token_id) in enumerate(zip(recorded, expected)):
            if row["token_index"] != index or row["reference_id"] != token_id:
                raise ValueError(f"Replay prefix differs at request {request}, token {index}")
            gap = row["top_logit"] - row["reference_logit"]
            if gap < 0 or not math.isclose(gap, row["gap"], abs_tol=1e-8):
                raise ValueError("Replay gap is inconsistent with recorded logits")
            ulp = bf16_ulp_below(row["top_logit"])
            total += 1
            top_differs += row["top_id"] != token_id
            positive += gap > 0
            max_gap = max(max_gap, gap)
            max_ratio = max(max_ratio, gap / ulp)
            if gap > ulp:
                exceeded += 1
                if first_exceeded is None:
                    first_exceeded = {
                        "request": request, "token": index,
                        "reference_id": token_id, "top_id": row["top_id"],
                        "gap": gap, "bf16_ulp": ulp,
                    }
    return {
        "tokens": total,
        "top_id_differs": top_differs,
        "positive_gap_count": positive,
        "over_one_bf16_ulp": exceeded,
        "max_gap": max_gap,
        "max_gap_in_ulps": max_ratio,
        "first_over_one_bf16_ulp": first_exceeded,
    }


def compare(baseline, candidate, forward, reverse):
    if baseline["engine"]["level"] != 1 or candidate["engine"]["level"] == 1:
        raise ValueError("Expected engine 1 baseline and a multi-tier candidate")
    if len(baseline["measurements"]) != 1 or len(candidate["measurements"]) != 1:
        raise ValueError("Use one measurement repeat for this diagnostic")
    for section, key in (
        ("software", "source_sha256"),
        ("model", "weights_sha256"),
        ("model", "config_sha256"),
        ("model", "tokenizer_sha256"),
        ("model", "dtype"),
        ("workload", "sha256"),
        ("workload", "input_token_ids_sha256"),
    ):
        if baseline[section][key] != candidate[section][key]:
            raise ValueError(f"Free runs differ in {section}.{key}")
    if baseline["model"]["dtype"] != "torch.bfloat16":
        raise ValueError("BF16 ULP comparison requires torch.bfloat16 outputs")
    levels = (candidate["engine"]["level"], 1)
    references = (baseline, candidate)
    for label, replay, level, reference in zip(
        ("forward", "reverse"), (forward, reverse), levels, references
    ):
        if replay["engine"] != level or replay["reference_engine"] != reference["engine"]["level"]:
            raise ValueError(f"{label} replay has the wrong engine direction")
        if replay["source_sha256"] != baseline["software"]["source_sha256"]:
            raise ValueError(f"{label} replay source differs from free runs")
        if replay["reference_output_sha256"] != reference["measurements"][0]["output_sha256"]:
            raise ValueError(f"{label} replay uses different free output tokens")
        for name in ("sha256", "input_token_ids_sha256"):
            field = "workload_sha256" if name == "sha256" else name
            if replay[field] != baseline["workload"][name]:
                raise ValueError(f"{label} replay input differs")
        if replay["cache_stats"]["runner"]["io_errors"]:
            raise ValueError(f"{label} replay has KV I/O errors")
    if not forward["verify_kv_transfers"]:
        raise ValueError("Forward replay must verify every KV transfer")
    transfer_counts = forward["cache_stats"]["runner"]["transfer_counts"]
    if not any(count > 0 for count in transfer_counts.values()):
        raise ValueError("Forward replay did not exercise a KV transfer")

    baseline_tokens = baseline["measurements"][0]["output_token_ids"]
    candidate_tokens = candidate["measurements"][0]["output_token_ids"]
    if len(baseline_tokens) != len(candidate_tokens) or any(
        len(left) != len(right)
        for left, right in zip(baseline_tokens, candidate_tokens)
    ):
        raise ValueError("Free runs have different output lengths")
    total = sum(len(row) for row in baseline_tokens)
    if total == 0:
        raise ValueError("Free runs have no output tokens")
    matches = sum(
        left == right
        for a, b in zip(baseline_tokens, candidate_tokens)
        for left, right in zip(a, b)
    )
    forward_stats = inspect_replay(forward, baseline_tokens)
    reverse_stats = inspect_replay(reverse, candidate_tokens)
    return {
        "diagnostic_only": True,
        "baseline_engine": 1,
        "candidate_engine": levels[0],
        "free_exact_token_rate": matches / total,
        "free_exact_tokens": matches,
        "free_total_tokens": total,
        "forward": forward_stats,
        "reverse": reverse_stats,
        "verified_transfer_counts": transfer_counts,
        "near_tie_candidate": (
            forward_stats["over_one_bf16_ulp"] == 0
            and reverse_stats["over_one_bf16_ulp"] == 0
        ),
        "task_quality_assessed": False,
        "formal_correctness_pass": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--candidate-config", required=True)
    parser.add_argument("--forward", type=Path, required=True)
    parser.add_argument("--reverse", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    paths = (
        reference_from_ledger(args.ledger),
        reference_from_ledger(args.ledger, args.candidate_config),
        args.forward, args.reverse,
    )
    records = []
    for path in paths:
        with path.open("r", encoding="utf-8") as file:
            records.append(json.load(file))
    report = compare(*records)
    report["inputs"] = {
        name: str(path.resolve())
        for name, path in zip(("baseline", "candidate", "forward", "reverse"), paths)
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"Diagnostic: {args.output}")
    print(json.dumps({key: report[key] for key in (
        "free_exact_token_rate", "forward", "reverse", "near_tie_candidate",
        "formal_correctness_pass",
    )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
