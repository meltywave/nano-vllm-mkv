import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Aggregate nano-vLLM result.json files")
    parser.add_argument("result_dir", nargs="?", default="results")
    parser.add_argument("--output", default="results/summary.csv")
    args = parser.parse_args()

    results = []
    for path in Path(args.result_dir).rglob("result.json"):
        with path.open("r", encoding="utf-8") as file:
            results.append((path, json.load(file)))

    baseline_hashes = {
        (result["model"]["id"], result["workload"]["name"]): result["summary"]
        ["correctness"]
        .get("output_sha256")
        for _, result in results
        if result["engine"]["level"] == 1
    }
    rows = []
    for path, result in results:
        key = (result["model"]["id"], result["workload"]["name"])
        output_hash = result["summary"]["correctness"].get("output_sha256")
        baseline_hash = baseline_hashes.get(key)
        rows.append(
            {
                "engine": result["engine"]["level"],
                "tiers": "+".join(result["engine"]["tiers"]),
                "model": result["model"]["id"],
                "workload": result["workload"]["name"],
                "latency_mean_s": result["summary"]["latency_s"]["mean"],
                "throughput_mean_tokens_s": result["summary"]["total_tokens_per_s"]["mean"],
                "gpu_peak_allocated_bytes": result["summary"]["gpu_peak_allocated_bytes"]["max"],
                "output_sha256": output_hash,
                "matches_engine_1": (
                    None if baseline_hash is None or output_hash is None
                    else output_hash == baseline_hash
                ),
                "result_path": str(path),
            }
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys() if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(output.resolve())


if __name__ == "__main__":
    main()
