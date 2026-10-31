"""Run a frozen MKV main-table workload on a separate inference engine.

Run this script in an environment containing the selected external engine. It
does not import either engine until all reference inputs have been validated.
"""

import argparse
import atexit
import csv
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import re
import statistics
import sys
from pathlib import Path
from time import perf_counter
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parent
OUTPUT_FILES = {"vllm": "vllm.jsonl", "nano-vllm": "nano_vllm.jsonl"}
WORKLOADS = ("three_tier_pressure", "remote_pressure")
VLLM_GPU_MEMORY_UTILIZATION = 0.80


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_tree_sha256(root):
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py"), key=lambda item: item.relative_to(root).as_posix()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def archive_tree_sha256(archive, root_name):
    digest = hashlib.sha256()
    with ZipFile(archive) as source:
        names = sorted(name for name in source.namelist()
                       if name.startswith(root_name + "/") and name.endswith(".py"))
        if not names:
            raise ValueError(f"Archive has no Python source under {root_name}/")
        for name in names:
            digest.update(name[len(root_name) + 1:].encode("utf-8"))
            digest.update(source.read(name))
    return digest.hexdigest()


def input_hash(prompts):
    payload = json.dumps(prompts, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def local_result_path(root, recorded):
    path = Path(recorded)
    if path.is_file():
        return path
    parts = path.parts
    if root.name not in parts:
        raise FileNotFoundError(f"Cannot locate reference result: {recorded}")
    path = root.joinpath(*parts[parts.index(root.name) + 1:])
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_reference(root, workload):
    ledger = root / f"ablation_runs_Qwen3-0.6B_{workload}.csv"
    with ledger.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    matches = [row for row in rows if row["config_id"] == "e4-main"]
    if len(matches) != 1 or matches[0]["status"] != "ok":
        raise ValueError(f"Expected one valid e4-main row in {ledger}")
    row = matches[0]
    result = json.loads(local_result_path(root, row["result_path"]).read_text(encoding="utf-8"))
    records = result["workload"]["prompt_records"]
    prompts = [record["token_ids"] for record in records]
    spec = result["workload"]
    if spec["name"] != workload or len(prompts) != spec["request_count"]:
        raise ValueError("Reference workload or request count differs")
    if any(len(prompt) != spec["target_prompt_tokens"] for prompt in prompts):
        raise ValueError("Reference prompt length differs")
    if input_hash(prompts) != spec["input_token_ids_sha256"]:
        raise ValueError("Reference token IDs do not match their recorded hash")
    if row["source_sha256"] != result["software"]["source_sha256"]:
        raise ValueError("Reference source fingerprint differs from its ledger")
    config = result["config_snapshot"]
    if (config["resolved_num_kvcache_blocks"] != 8
            or config["resolved_config"]["kvcache_block_size"] != 256
            or spec["sampling"] != {"temperature": 0.0, "ignore_eos": True}):
        raise ValueError("Reference is not the frozen greedy 8-block configuration")
    if result["summary"]["measurement_repeats"] != 2 or config["benchmark"]["warmup_runs"] != 1:
        raise ValueError("Reference warmup or measurement count differs")
    return result, prompts


def check_model(model, reference):
    expected = reference["model"]
    weights = sorted(model.glob("*.safetensors"))
    if not weights:
        raise FileNotFoundError(f"No safetensors weight files found in {model}")
    digest = hashlib.sha256()
    for path in weights:
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    config = model / "config.json"
    if not config.is_file():
        raise FileNotFoundError(f"No model config.json found in {model}")
    tokenizer = hashlib.sha256()
    for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json",
                 "special_tokens_map.json"):
        path = model / name
        if path.is_file():
            tokenizer.update(name.encode("utf-8"))
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    tokenizer.update(chunk)
    observed = {
        "weights_sha256": digest.hexdigest(),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "tokenizer_sha256": tokenizer.hexdigest(),
    }
    for name, value in observed.items():
        if value != expected[name]:
            raise ValueError(f"Model {name} differs from the MKV reference")


def import_engine(engine, source_dir=None):
    package = "vllm" if engine == "vllm" else "nanovllm"
    if any(name == package or name.startswith(package + ".") for name in sys.modules):
        raise RuntimeError(f"{package} was imported before its environment was checked")
    original_path = sys.path[:]
    if source_dir is not None:
        source_dir = source_dir.resolve()
        if not source_dir.is_dir():
            raise FileNotFoundError(source_dir)
        sys.path.insert(0, str(source_dir))
    else:
        # The repository also contains its own nanovllm package. Select the
        # installed original without allowing this script's directory to win.
        sys.path[:] = [path for path in sys.path if Path(path or ".").resolve() != ROOT]
    try:
        module = importlib.import_module(package)
    except ImportError as exc:
        raise RuntimeError(
            f"Cannot import {package} in this Python environment: {exc}. "
            "Install the selected engine and run --preflight before timing."
        ) from exc
    finally:
        sys.path[:] = original_path
    location = Path(module.__file__).resolve()
    if source_dir is not None and not location.is_relative_to(source_dir):
        raise RuntimeError(f"Imported {package} from {location}, expected {source_dir}")
    if source_dir is None and location.is_relative_to(ROOT):
        raise RuntimeError(
            f"Imported repository copy of {package} from {location}; "
            "install the original in a separate environment or pass --source-dir explicitly."
        )
    return module


def check_engine_api(engine, module):
    if not callable(getattr(module, "LLM", None)) or not callable(getattr(module, "SamplingParams", None)):
        raise RuntimeError(f"Installed {engine} does not expose LLM and SamplingParams")
    if engine == "nano-vllm":
        from nanovllm.engine.model_runner import ModelRunner
        from nanovllm.layers.sampler import Sampler

        if list(inspect.signature(ModelRunner.allocate_kv_cache).parameters) != ["self"]:
            raise RuntimeError("Installed nano-vLLM KV allocator differs from the tested adapter")
        if list(inspect.signature(Sampler.forward).parameters) != ["self", "logits", "temperatures"]:
            raise RuntimeError("Installed nano-vLLM sampler differs from the tested adapter")
        if list(inspect.signature(ModelRunner.run).parameters) != [
            "self", "seqs", "is_prefill"
        ]:
            raise RuntimeError(
                "Installed nano-vLLM model runner differs from the tested original; "
                "the cloned MKV package is not an external baseline"
            )
    else:
        params = inspect.signature(module.LLM).parameters
        if "kv_cache_memory_bytes" not in params:
            raise RuntimeError("Installed vLLM lacks the required KV capacity option")


def engine_provenance(engine, module, source_dir):
    package_root = Path(module.__file__).resolve().parent
    distribution = "vllm" if engine == "vllm" else "nano-vllm"
    if source_dir is None:
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"{distribution} has no installed distribution metadata; "
                "install it in this environment or pass --source-dir explicitly"
            ) from exc
        return {"package_version": version, "package_path": str(package_root),
                "source_tree_sha256": source_tree_sha256(package_root),
                "source_archive_sha256": ""}
    archive = source_dir.parent / (source_dir.name + ".zip")
    archive_hash = ""
    if archive.is_file():
        if archive_tree_sha256(archive, source_dir.name) != source_tree_sha256(source_dir):
            raise ValueError(f"Source directory differs from its archive: {source_dir}")
        archive_hash = sha256_file(archive)
    return {"package_version": getattr(module, "__version__", "source-snapshot"),
            "package_path": str(package_root),
            "source_tree_sha256": source_tree_sha256(source_dir),
            "source_archive_sha256": archive_hash}


def installed_archive_hash(engine, archive):
    distribution = "vllm" if engine == "vllm" else "nano-vllm"
    installed = importlib.metadata.distribution(distribution)
    raw = installed.read_text("direct_url.json")
    if not raw:
        raise RuntimeError(f"{distribution} has no direct_url.json to verify its archive")
    source_url = urlparse(json.loads(raw).get("url", ""))
    if (source_url.scheme != "file"
            or Path(url2pathname(unquote(source_url.path))).resolve() != archive):
        raise RuntimeError(f"Installed {distribution} was not built from {archive}")
    return sha256_file(archive)


def adapt_nano_greedy_and_capacity(num_blocks):
    """Apply the two required benchmark adaptations to the downloaded snapshot."""
    import torch
    from nanovllm.engine.model_runner import ModelRunner
    from nanovllm.layers.sampler import Sampler
    from nanovllm.sampling_params import SamplingParams

    original_check = SamplingParams.__post_init__

    def check_temperature(self):
        if self.temperature == 0:
            return
        original_check(self)

    def sample(self, logits, temperatures):
        return logits.float().argmax(dim=-1)

    def allocate(self):
        config = self.config
        hf_config = config.hf_config
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = (2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads
                       * head_dim * hf_config.dtype.itemsize)
        free, total = torch.cuda.mem_get_info()
        if free < num_blocks * block_bytes:
            raise RuntimeError("Not enough free GPU memory for the locked KV capacity")
        self.kv_cache = torch.empty(
            (2, hf_config.num_hidden_layers, num_blocks, self.block_size,
             num_kv_heads, head_dim), dtype=hf_config.dtype, device="cuda",
        )
        config.num_kvcache_blocks = num_blocks
        layer = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer]
                module.v_cache = self.kv_cache[1, layer]
                layer += 1
        if layer != self.kv_cache.shape[1]:
            raise RuntimeError("Could not rebind every original nano-vLLM KV layer")

    SamplingParams.__post_init__ = check_temperature
    Sampler.forward = torch.compile(sample)
    ModelRunner.allocate_kv_cache = allocate


def reset_nano_prefix_cache(llm):
    manager = llm.scheduler.block_manager
    if manager.used_block_ids or llm.scheduler.waiting or llm.scheduler.running:
        raise RuntimeError("nano-vLLM still has active requests")
    manager.hash_to_block_id.clear()
    for block in manager.blocks:
        block.hash = -1
        block.token_ids = []


def make_engine(engine, module, model, reference):
    config = reference["config_snapshot"]
    spec = reference["workload"]
    common = {
        "model": str(model),
        "max_model_len": spec["max_model_len"],
        "max_num_seqs": spec["max_num_seqs"],
        "max_num_batched_tokens": spec["max_num_batched_tokens"],
        "tensor_parallel_size": 1,
        "enforce_eager": False,
    }
    if engine == "nano-vllm":
        return module.LLM(**common, kvcache_block_size=256)
    return module.LLM(
        **common, dtype="bfloat16", seed=spec["seed"], block_size=16,
        gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
        kv_cache_memory_bytes=config["kv_block_bytes"] * config["resolved_num_kvcache_blocks"],
        enable_prefix_caching=False,
    )


def run_repetitions(engine, module, model, reference, prompts):
    import torch

    llm = make_engine(engine, module, model, reference)
    if engine == "nano-vllm":
        atexit.unregister(llm.exit)
    spec = reference["workload"]
    sampling = module.SamplingParams(temperature=0, ignore_eos=True, max_tokens=spec["output_tokens"])
    inputs = ([{"prompt_token_ids": prompt} for prompt in prompts]
              if engine == "vllm" else prompts)
    try:
        if (engine == "nano-vllm"
                and len(llm.scheduler.block_manager.blocks) != 8):
            raise RuntimeError("nano-vLLM GPU KV capacity did not lock to 8 blocks")
        observed_blocks = 8
        if engine == "vllm":
            cache_config = llm.llm_engine.vllm_config.cache_config
            observed_blocks = cache_config.num_gpu_blocks
            if observed_blocks is None or observed_blocks * cache_config.block_size != 2048:
                raise RuntimeError(
                    f"vLLM GPU KV capacity is {observed_blocks} x "
                    f"{cache_config.block_size} tokens, expected 2048 tokens"
                )
        measurements = []
        for repeat in range(3):
            if engine == "nano-vllm":
                reset_nano_prefix_cache(llm)
            elif repeat and not llm.reset_prefix_cache():
                raise RuntimeError("vLLM could not reset the prefix cache")
            torch.manual_seed(spec["seed"])
            torch.cuda.manual_seed_all(spec["seed"])
            started = perf_counter()
            outputs = llm.generate(inputs, sampling, use_tqdm=False)
            elapsed = perf_counter() - started
            token_ids = ([list(output.outputs[0].token_ids) for output in outputs]
                         if engine == "vllm" else [output["token_ids"] for output in outputs])
            if len(token_ids) != len(prompts) or any(
                len(ids) != spec["output_tokens"] for ids in token_ids
            ):
                raise RuntimeError(f"Repeat {repeat} returned incomplete output")
            if repeat:
                tokens = sum(map(len, token_ids))
                measurements.append({
                    "repeat": repeat, "latency_s": elapsed,
                    "output_tokens": tokens,
                    "output_tokens_per_s": tokens / elapsed,
                    "requests_per_s": len(prompts) / elapsed,
                    "output_token_ids_sha256": input_hash(token_ids),
                    "output_token_ids": token_ids,
                })
        if len({item["output_token_ids_sha256"] for item in measurements}) != 1:
            raise RuntimeError("Measured repeats produced different output token IDs")
        return measurements, observed_blocks
    finally:
        if engine == "nano-vllm":
            llm.exit()


def make_record(engine, revision, archive_hash, tree_hash, reference,
                measurements, gpu_total_bytes, observed_blocks,
                runtime=None):
    model = reference["model"]
    spec = reference["workload"]
    config = reference["config_snapshot"]
    return {
        "model": model["id"],
        "model_weights_sha256": model["weights_sha256"],
        "model_config_sha256": model["config_sha256"],
        "tokenizer_sha256": model["tokenizer_sha256"],
        "workload": spec["name"],
        "workload_sha256": spec["sha256"],
        "input_token_ids_sha256": spec["input_token_ids_sha256"],
        "model_dtype": model["dtype"],
        "sampling_temperature": spec["sampling"]["temperature"],
        "sampling_ignore_eos": spec["sampling"]["ignore_eos"],
        "request_count": spec["request_count"],
        "output_tokens": spec["output_tokens"],
        "max_num_seqs": spec["max_num_seqs"],
        "max_num_batched_tokens": spec["max_num_batched_tokens"],
        "gpu_total_bytes": gpu_total_bytes,
        "git_commit": revision,
        "source_archive_sha256": archive_hash,
        "source_tree_sha256": tree_hash,
        "benchmark_script_sha256": sha256_file(Path(__file__)),
        "warmup_runs": 1,
        "observed_gpu_kv_blocks": observed_blocks,
        "observed_gpu_kv_block_tokens": 16 if engine == "vllm" else 256,
        "vllm_gpu_memory_utilization": (
            VLLM_GPU_MEMORY_UTILIZATION if engine == "vllm" else None
        ),
        "measurement_repeats": len(measurements),
        "latency_mean_s": statistics.fmean(m["latency_s"] for m in measurements),
        "output_tokens_per_s": statistics.fmean(m["output_tokens_per_s"] for m in measurements),
        "requests_per_s": statistics.fmean(m["requests_per_s"] for m in measurements),
        "benchmark_adapter": (
            "greedy_argmax_and_exact_8x256_gpu_kv_blocks" if engine == "nano-vllm"
            else "16_token_blocks_with_matched_gpu_kv_bytes"
        ),
        "reference_run_id": reference["run_id"],
        "runtime": runtime or {},
        "measurements": measurements,
    }


def write_record(path, record):
    rows = []
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
    if any(row["workload"] == record["workload"] for row in rows):
        raise FileExistsError(f"{path} already has workload {record['workload']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=tuple(OUTPUT_FILES), required=True)
    parser.add_argument("--workload", choices=WORKLOADS, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, default=ROOT / "results_formal_tables_20261020")
    parser.add_argument("--source-dir", type=Path, help="Optional unpacked source; default uses the installed package")
    parser.add_argument("--source-archive", type=Path, help="Archive used to install the selected package")
    parser.add_argument("--git-commit", help="Verified 40-character source commit, if known")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Check frozen inputs without importing an engine")
    mode.add_argument("--preflight", action="store_true", help="Check installed dependencies and GPU without loading the model")
    args = parser.parse_args()
    root = args.result_dir.resolve()
    source_dir = args.source_dir.resolve() if args.source_dir else None
    archive = args.source_archive.resolve() if args.source_archive else None
    if archive is not None and (source_dir is not None or not archive.is_file()):
        raise ValueError("--source-archive needs an existing file and cannot be combined with --source-dir")
    reference, prompts = load_reference(root, args.workload)
    model = args.model.resolve()
    if not model.is_dir():
        raise FileNotFoundError(model)
    check_model(model, reference)
    if source_dir is not None and not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    output = root / "external" / OUTPUT_FILES[args.engine]
    if output.exists():
        with output.open(encoding="utf-8") as stream:
            if any(json.loads(line)["workload"] == args.workload for line in stream if line.strip()):
                raise FileExistsError(f"{output} already has {args.workload}")
    print(json.dumps({
        "engine": args.engine, "workload": args.workload,
        "requests": len(prompts), "input_token_ids_sha256": input_hash(prompts),
        "package_mode": "installed" if source_dir is None else "source-dir",
        "source_dir": str(source_dir) if source_dir else None,
        "output": str(output), "gpu_kv_bytes": reference["config_snapshot"]["kv_block_bytes"] * 8,
    }, indent=2))
    if args.dry_run:
        return
    if args.git_commit and not re.fullmatch(r"[0-9a-fA-F]{40}", args.git_commit):
        raise ValueError("--git-commit must contain 40 hexadecimal characters")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment")
    gpu_bytes = torch.cuda.get_device_properties(0).total_memory
    if gpu_bytes != reference["hardware"]["gpu_total_bytes"]:
        raise ValueError("GPU total memory differs from the MKV main-table run")
    if args.engine == "vllm":
        free_bytes, _ = torch.cuda.mem_get_info(0)
        required_bytes = int(gpu_bytes * VLLM_GPU_MEMORY_UTILIZATION)
        if free_bytes < required_bytes:
            raise RuntimeError(
                f"vLLM startup needs at least {required_bytes / 1024**3:.2f} GiB "
                f"free at gpu_memory_utilization={VLLM_GPU_MEMORY_UTILIZATION:.2f}; "
                f"currently {free_bytes / 1024**3:.2f} GiB is free. "
                "Inspect other GPU processes before timing."
            )
        print(json.dumps({
            "gpu_free_bytes": free_bytes,
            "vllm_startup_required_bytes": required_bytes,
            "vllm_gpu_memory_utilization": VLLM_GPU_MEMORY_UTILIZATION,
        }), flush=True)
    module = import_engine(args.engine, source_dir)
    check_engine_api(args.engine, module)
    provenance = engine_provenance(args.engine, module, source_dir)
    if archive is not None:
        provenance["source_archive_sha256"] = installed_archive_hash(args.engine, archive)
    print(json.dumps({"engine_preflight": "ok", **provenance}, indent=2), flush=True)
    if args.preflight:
        return
    if args.engine == "nano-vllm":
        adapt_nano_greedy_and_capacity(8)
    measurements, observed_blocks = run_repetitions(
        args.engine, module, model, reference, prompts
    )
    record = make_record(
        args.engine, args.git_commit.lower() if args.git_commit else "unknown",
        provenance["source_archive_sha256"], provenance["source_tree_sha256"], reference,
        measurements, gpu_bytes, observed_blocks,
        runtime={
            "python": sys.version.split()[0], "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "engine_package": provenance["package_version"],
            "engine_package_path": provenance["package_path"],
        },
    )
    write_record(output, record)
    print(f"External result: {output} ({args.workload})")


if __name__ == "__main__":
    main()
