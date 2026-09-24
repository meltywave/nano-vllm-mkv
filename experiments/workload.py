import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from experiments.configuration import REPO_ROOT


@dataclass(frozen=True, slots=True)
class Workload:
    path: Path
    document: dict
    prompts: tuple[dict, ...]
    sha256: str

    @property
    def name(self):
        return self.document["name"]

    @property
    def benchmark_preset(self):
        return self.document.get("benchmark_preset", "main")


def _sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(value: str):
    path = (REPO_ROOT / value).resolve()
    if not path.is_relative_to(REPO_ROOT):
        raise ValueError(f"Workload path escapes repository root: {value}")
    return path


def load_workload(path: str | Path):
    manifest_path = Path(path).resolve()
    with manifest_path.open("r", encoding="utf-8") as file:
        document = json.load(file)
    required = {
        "schema_version",
        "name",
        "prompt_file",
        "prompt_count",
        "target_prompt_tokens",
        "output_tokens",
        "num_seqs",
        "max_model_len",
        "seed",
        "sampling",
    }
    missing = sorted(required - document.keys())
    if missing:
        raise ValueError(f"Workload is missing required fields: {missing}")
    if document["schema_version"] != 1:
        raise ValueError("Only workload schema_version=1 is supported")
    if document["target_prompt_tokens"] + document["output_tokens"] > document["max_model_len"]:
        raise ValueError("prompt tokens plus output tokens exceed max_model_len")

    prompt_path = _repo_path(document["prompt_file"])
    prompts = []
    with prompt_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if line.strip():
                record = json.loads(line)
                if not record.get("id") or not record.get("text"):
                    raise ValueError(f"Invalid prompt record at line {line_number}")
                prompts.append(record)
    if len(prompts) != document["prompt_count"]:
        raise ValueError(
            f"prompt_count={document['prompt_count']} but {prompt_path} has "
            f"{len(prompts)} records"
        )
    if document["num_seqs"] > len(prompts):
        raise ValueError("num_seqs cannot exceed the number of prompt records")

    digest = hashlib.sha256()
    digest.update(_sha256(manifest_path).encode("ascii"))
    digest.update(_sha256(prompt_path).encode("ascii"))
    return Workload(manifest_path, document, tuple(prompts), digest.hexdigest())


def tokenize_workload(workload: Workload, tokenizer):
    target = int(workload.document["target_prompt_tokens"])
    length_policy = workload.document.get("length_policy", "truncate_only")
    tokenized = []
    records = []
    for record in workload.prompts[: workload.document["num_seqs"]]:
        token_ids = tokenizer.encode(record["text"], add_special_tokens=False)
        source_length = len(token_ids)
        if not token_ids:
            raise ValueError(f"Tokenizer produced no tokens for prompt {record['id']}")
        if len(token_ids) < target:
            if length_policy != "repeat_or_truncate":
                raise ValueError(
                    f"Prompt {record['id']} has {len(token_ids)} tokens; target is "
                    f"{target} and length_policy does not permit expansion"
                )
            repeats = (target + len(token_ids) - 1) // len(token_ids)
            token_ids = (token_ids * repeats)[:target]
        else:
            token_ids = token_ids[:target]
        digest = hashlib.sha256(
            b"".join(int(token).to_bytes(8, "little") for token in token_ids)
        ).hexdigest()
        tokenized.append(token_ids)
        records.append(
            {
                "id": record["id"],
                "source": record.get("source"),
                "source_tokens": source_length,
                "actual_prompt_tokens": len(token_ids),
                "token_sha256": digest,
            }
        )
    return tokenized, records
