"""Prepare, run, and compare a fixed SQuAD v1.1 quality set."""

import argparse
import hashlib
import json
import random
import re
import statistics
import string
from bisect import bisect_right
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

from experiments.metrics import output_sha256, tokenizer_files_sha256
from experiments.results import model_config_sha256, model_manifest_sha256, source_sha256


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_questions(path):
    with path.open("r", encoding="utf-8") as file:
        source = json.load(file)
    if source.get("version") != "1.1":
        raise ValueError("Expected SQuAD v1.1 JSON")
    questions = []
    for article in source["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                answers = [answer["text"] for answer in qa["answers"]]
                if not answers:
                    raise ValueError(f"Question {qa['id']} has no gold answer")
                questions.append({
                    "id": qa["id"], "context": paragraph["context"],
                    "question": qa["question"], "answers": answers,
                })
    if len({item["id"] for item in questions}) != len(questions):
        raise ValueError("Duplicate SQuAD question ID")
    return questions


def prompt_tokens(tokenizer, item):
    content = (
        "Answer the question using the context. Give only the shortest answer "
        "phrase.\n\nContext:\n" + item["context"] +
        "\n\nQuestion:\n" + item["question"]
    )
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False,
    )
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if len(encoded) == 1 and isinstance(encoded[0], (list, tuple)):
        encoded = encoded[0]
    if not isinstance(encoded, (list, tuple)) or any(
        not isinstance(token, int) or token < 0 for token in encoded
    ):
        raise ValueError(f"Tokenizer returned invalid input_ids for question {item['id']}")
    return list(encoded)


def prepare(args):
    from transformers import AutoTokenizer

    source_path = args.dataset.resolve()
    model_path = args.model.resolve()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    candidates = load_questions(source_path)
    eligible = []
    skipped_short = skipped_long = 0
    prompt_lengths = []
    max_prompt_tokens = args.max_model_len - args.output_tokens
    for item in candidates:
        ids = prompt_tokens(tokenizer, item)
        prompt_lengths.append(len(ids))
        if not ids or len(ids) < args.min_prompt_tokens:
            skipped_short += 1
            continue
        if len(ids) > max_prompt_tokens:
            skipped_long += 1
            continue
        eligible.append({
            "id": item["id"], "input_ids": ids, "answers": item["answers"],
        })
    if len(eligible) < args.count:
        lengths = sorted(prompt_lengths)
        bins = (64, 96, 128, 160, 192, 224, 256, 320, 512, 768, max_prompt_tokens)
        eligible_counts = {
            str(bound): len(lengths) - bisect_right(lengths, bound - 1)
            - (len(lengths) - bisect_right(lengths, max_prompt_tokens))
            for bound in bins if bound <= max_prompt_tokens
        }
        raise ValueError(
            f"Only {len(eligible)} eligible questions; need {args.count}. "
            f"Tokenized prompt lengths: min={lengths[0]}, "
            f"p50={lengths[len(lengths)//2]}, "
            f"p90={lengths[int(0.9*(len(lengths)-1))]}, max={lengths[-1]}; "
            f"counts in [minimum, {max_prompt_tokens}]={eligible_counts}. "
            "Review the requested count and model length before any quality run."
        )
    question_hash = lambda item: hashlib.sha256(item["id"].encode()).digest()
    eligible.sort(key=question_hash)
    examples = eligible[:args.count]
    selected_lengths = [len(item["input_ids"]) for item in examples]
    manifest = {
        "schema_version": 1, "task": "squad_v1.1", "source_sha256": sha256_file(source_path),
        "source_name": source_path.name,
        "selection": "sha256(question_id) ascending among prompts within the fixed length interval",
        "min_prompt_tokens": args.min_prompt_tokens,
        "selected_prompt_tokens_min": min(selected_lengths),
        "selected_prompt_tokens_max": max(selected_lengths),
        "eligible_questions": len(eligible),
        "skipped_short": skipped_short, "skipped_long": skipped_long,
        "model_weights_sha256": model_manifest_sha256(model_path),
        "model_config_sha256": model_config_sha256(model_path),
        "tokenizer_sha256": tokenizer_files_sha256(model_path),
        "input_token_ids_sha256": output_sha256([{"token_ids": e["input_ids"]} for e in examples]),
        "max_model_len": args.max_model_len, "output_tokens": args.output_tokens,
        "count": len(examples), "examples": examples,
    }
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"Prepared {len(examples)} examples: {args.output}")


def normalize_answer(value):
    value = value.lower().translate(str.maketrans("", "", string.punctuation))
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def answer_scores(prediction, answers):
    predicted = normalize_answer(prediction).split()
    scores = []
    for answer in answers:
        gold = normalize_answer(answer).split()
        common = sum((Counter(predicted) & Counter(gold)).values())
        f1 = 1.0 if not predicted and not gold else (
            2 * common / (len(predicted) + len(gold))
            if predicted and gold else 0.0
        )
        scores.append((float(predicted == gold), f1))
    return {"em": max(score[0] for score in scores), "f1": max(score[1] for score in scores)}


def target_tier_round_trip(engine, transfer_counts):
    if engine == 1:
        return True
    tier = {2: "cpu", 3: "ssd", 4: "remote"}[engine]
    wrote = any(
        direction.endswith(f"_to_{tier}") and count > 0
        for direction, count in transfer_counts.items()
    )
    return wrote and transfer_counts.get(f"{tier}_to_gpu", 0) > 0


def load_manifest(path):
    with path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    if manifest.get("schema_version") != 1 or manifest.get("task") != "squad_v1.1":
        raise ValueError("Expected prepared SQuAD v1.1 manifest")
    if len(manifest["examples"]) != manifest["count"]:
        raise ValueError("Manifest example count differs")
    digest = output_sha256([{"token_ids": item["input_ids"]} for item in manifest["examples"]])
    if digest != manifest["input_token_ids_sha256"]:
        raise ValueError("Manifest tokenized inputs changed")
    return manifest


def apply_watermarks(cache, tiers, low, high, overrides):
    active = set(tiers[:-1])
    for tier in active:
        cache["kv_low_watermarks"][tier] = low
        cache["kv_high_watermarks"][tier] = high
    for override in overrides:
        try:
            tier, override_low, override_high = override.split(":")
            override_low, override_high = float(override_low), float(override_high)
        except ValueError as exc:
            raise ValueError("--watermark must be TIER:LOW:HIGH") from exc
        if tier not in active or not 0 <= override_low < override_high <= 1:
            raise ValueError(f"Invalid active-tier watermark: {override}")
        cache["kv_low_watermarks"][tier] = override_low
        cache["kv_high_watermarks"][tier] = override_high


def run(args):
    import torch
    from nanovllm import LLM, SamplingParams
    from experiments.configuration import load_cache_engine

    manifest = load_manifest(args.manifest)
    model = args.model.resolve()
    for key, value in (
        ("model_weights_sha256", model_manifest_sha256(model)),
        ("model_config_sha256", model_config_sha256(model)),
        ("tokenizer_sha256", tokenizer_files_sha256(model)),
    ):
        if manifest[key] != value:
            raise ValueError(f"Quality manifest {key} differs from model")
    if args.engine == 4 and not args.remote_host:
        raise ValueError("Engine 4 requires --remote-host")
    if args.engine == 4 and args.remote_host in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Engine 4 quality run requires a real network Remote host")
    cache, snapshot = load_cache_engine(args.engine)
    cache["kv_placement"] = args.kv_placement
    apply_watermarks(cache, snapshot["tiers"], args.low, args.high, args.watermark)
    cache.update(
        num_kvcache_blocks=args.gpu_blocks,
        num_cpu_kvcache_blocks=args.cpu_blocks,
        num_ssd_kvcache_blocks=args.ssd_blocks,
        num_remote_kvcache_blocks=args.remote_blocks,
        ssd_kvcache_path=str(args.ssd_path.resolve()),
        max_model_len=manifest["max_model_len"],
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        kv_verify_transfers=args.verify_kv_transfers,
    )
    if args.remote_host:
        cache["remote_kvcache_host"] = args.remote_host
        cache["remote_kvcache_port"] = args.remote_port
    llm = LLM(str(model), **cache)
    try:
        expected = (
            ("num_kvcache_blocks", args.gpu_blocks),
            ("num_cpu_kvcache_blocks", args.cpu_blocks if args.engine >= 2 else 0),
            ("num_ssd_kvcache_blocks", args.ssd_blocks if args.engine >= 3 else 0),
            ("num_remote_kvcache_blocks", args.remote_blocks if args.engine >= 4 else 0),
        )
        for name, value in expected:
            if getattr(llm.config, name) != value:
                raise ValueError(f"Resolved {name} differs from requested {value}")
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        params = SamplingParams(temperature=0.0, ignore_eos=False,
                                max_tokens=manifest["output_tokens"])
        outputs = llm.generate(
            [example["input_ids"] for example in manifest["examples"]],
            params, use_tqdm=False, report_progress=True,
        )
        if len(outputs) != manifest["count"]:
            raise RuntimeError("Quality run did not complete every request")
        cache_stats = llm.cache_stats()
        if cache_stats["runner"]["io_errors"]:
            raise RuntimeError("Quality run had KV I/O errors")
        rows = []
        for example, output in zip(manifest["examples"], outputs):
            prediction = llm.tokenizer.decode(
                output["token_ids"], skip_special_tokens=True
            ).strip()
            scores = answer_scores(prediction, example["answers"])
            rows.append({
                "id": example["id"], "prediction": prediction,
                "output_token_ids": output["token_ids"], **scores,
            })
        result = {
            "schema_version": 1, "task": "squad_v1.1", "engine": args.engine,
            "kv_placement": args.kv_placement,
            "manifest_sha256": sha256_file(args.manifest),
            "source_sha256": source_sha256(),
            "input_token_ids_sha256": manifest["input_token_ids_sha256"],
            "model_weights_sha256": manifest["model_weights_sha256"],
            "model_config_sha256": manifest["model_config_sha256"],
            "tokenizer_sha256": manifest["tokenizer_sha256"],
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
            "seed": args.seed, "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "sampling": {"temperature": 0.0, "ignore_eos": False,
                         "max_tokens": manifest["output_tokens"]},
            "gpu_blocks": args.gpu_blocks, "cpu_blocks": args.cpu_blocks if args.engine >= 2 else 0,
            "ssd_blocks": args.ssd_blocks if args.engine >= 3 else 0,
            "remote_blocks": args.remote_blocks if args.engine >= 4 else 0,
            "watermarks": {
                tier: [cache["kv_low_watermarks"][tier], cache["kv_high_watermarks"][tier]]
                for tier in snapshot["tiers"][:-1]
            },
            "verify_kv_transfers": args.verify_kv_transfers,
            "target_tier_round_trip": target_tier_round_trip(
                args.engine, cache_stats["runner"]["transfer_counts"]
            ),
            "remote_host": args.remote_host if args.engine == 4 else None,
            "remote_port": args.remote_port if args.engine == 4 else None,
            "mean_em": statistics.fmean(row["em"] for row in rows),
            "mean_f1": statistics.fmean(row["f1"] for row in rows),
            "examples": rows, "cache_stats": cache_stats,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)
            file.write("\n")
        print(f"Quality result: {args.output}")
        print(f"engine={args.engine} examples={len(rows)} EM={result['mean_em']:.4f} F1={result['mean_f1']:.4f}")
    finally:
        llm.exit()


def compare(args):
    with args.baseline.open("r", encoding="utf-8") as file:
        baseline = json.load(file)
    with args.candidate.open("r", encoding="utf-8") as file:
        candidate = json.load(file)
    for field in (
        "manifest_sha256", "input_token_ids_sha256", "model_weights_sha256",
        "model_config_sha256", "tokenizer_sha256", "source_sha256",
        "gpu_blocks", "gpu_name", "gpu_total_bytes", "seed",
        "max_num_seqs", "max_num_batched_tokens", "sampling",
    ):
        if baseline[field] != candidate[field]:
            raise ValueError(f"Quality runs differ in {field}")
    if baseline.get("kv_placement", "sequence") != candidate.get("kv_placement", "sequence"):
        raise ValueError("Quality runs used different KV placement strategies")
    if baseline["engine"] != 1 or candidate["engine"] == 1:
        raise ValueError("Expected L1 baseline and L2-L4 candidate")
    left, right = baseline["examples"], candidate["examples"]
    if not left or [row["id"] for row in left] != [row["id"] for row in right]:
        raise ValueError("Quality runs have different question IDs")
    if baseline["cache_stats"]["runner"]["io_errors"] or candidate["cache_stats"]["runner"]["io_errors"]:
        raise ValueError("Quality run has KV I/O errors")
    differences = [b["f1"] - a["f1"] for a, b in zip(left, right)]
    rng = random.Random(args.seed)
    means = sorted(
        statistics.fmean(differences[rng.randrange(len(differences))]
                         for _ in differences)
        for _ in range(args.bootstrap_repeats)
    )
    lower = means[int(0.05 * (len(means) - 1))]
    report = {
        "schema_version": 1, "task": "squad_v1.1",
        "baseline": str(args.baseline.resolve()),
        "candidate": str(args.candidate.resolve()),
        "count": len(differences), "mean_f1_difference": statistics.fmean(differences),
        "mean_em_difference": candidate["mean_em"] - baseline["mean_em"],
        "paired_bootstrap_lower_95": lower,
        "bootstrap_repeats": args.bootstrap_repeats, "bootstrap_seed": args.seed,
        "noninferiority_margin": args.margin,
        "noninferior_to_internal_l1": lower > -args.margin,
        "target_tier_round_trip": candidate["target_tier_round_trip"],
        "kv_transfer_verified": candidate["verify_kv_transfers"],
        "internal_quality_gate_pass": (
            lower > -args.margin
            and candidate["target_tier_round_trip"]
            and candidate["verify_kv_transfers"]
        ),
        "baseline_mean_f1": baseline["mean_f1"],
        "candidate_mean_f1": candidate["mean_f1"],
        "external_reference_validated": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--dataset", type=Path, required=True)
    prep.add_argument("--model", type=Path, required=True)
    prep.add_argument("--count", type=int, default=256)
    prep.add_argument("--max-model-len", type=int, default=1024)
    prep.add_argument("--min-prompt-tokens", type=int, default=256)
    prep.add_argument("--output-tokens", type=int, default=64)
    prep.add_argument("--output", type=Path, required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--manifest", type=Path, required=True)
    run_parser.add_argument("--model", type=Path, required=True)
    run_parser.add_argument("--engine", type=int, choices=(1, 2, 3, 4), required=True)
    for name, default in (("gpu-blocks", 8), ("cpu-blocks", 8),
                          ("ssd-blocks", 32), ("remote-blocks", 16),
                          ("max-num-seqs", 8), ("max-num-batched-tokens", 512)):
        run_parser.add_argument("--" + name, type=int, default=default)
    run_parser.add_argument("--ssd-path", type=Path, required=True)
    run_parser.add_argument("--remote-host")
    run_parser.add_argument("--remote-port", type=int, default=19090)
    run_parser.add_argument("--low", type=float, default=0.5)
    run_parser.add_argument("--high", type=float, default=0.8)
    run_parser.add_argument("--watermark", action="append", default=[],
                            metavar="TIER:LOW:HIGH")
    run_parser.add_argument("--verify-kv-transfers", action="store_true")
    run_parser.add_argument("--kv-placement", choices=("sequence", "block-hot"),
                            default="sequence")
    run_parser.add_argument("--seed", type=int, default=0)
    run_parser.add_argument("--output", type=Path, required=True)
    comp = commands.add_parser("compare")
    comp.add_argument("--baseline", type=Path, required=True)
    comp.add_argument("--candidate", type=Path, required=True)
    comp.add_argument("--margin", type=float, default=0.05)
    comp.add_argument("--bootstrap-repeats", type=int, default=2000)
    comp.add_argument("--seed", type=int, default=20261020)
    comp.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    if args.command == "prepare":
        if (args.count <= 0 or args.output_tokens <= 0
                or args.output_tokens >= args.max_model_len
                or not 0 <= args.min_prompt_tokens <= args.max_model_len - args.output_tokens):
            parser.error("Invalid sample count or token limits")
        prepare(args)
    elif args.command == "run":
        if not 0 <= args.low < args.high <= 1:
            parser.error("Require 0 <= low < high <= 1")
        run(args)
    else:
        if args.bootstrap_repeats < 100 or not 0 < args.margin < 1:
            parser.error("Require at least 100 bootstrap repeats and margin in (0,1)")
        compare(args)


if __name__ == "__main__":
    main()
