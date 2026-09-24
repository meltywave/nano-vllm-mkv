import json
from pathlib import Path

from nanovllm.config import ENGINE_TIERS


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_CONFIG = REPO_ROOT / "experiments" / "config" / "cache_levels.yaml"
DEFAULT_BENCHMARK_CONFIG = REPO_ROOT / "experiments" / "config" / "benchmark.yaml"


def _load_json_yaml(path: Path):
    """Load the repository's JSON-compatible YAML without another dependency."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_cache_engine(engine: int, path: Path = DEFAULT_CACHE_CONFIG):
    document = _load_json_yaml(path)
    try:
        level = document["levels"][str(engine)]
        profile = document["capacity_profiles"][level["capacity_profile"]]
    except KeyError as exc:
        raise ValueError(f"Invalid cache configuration in {path}: missing {exc}") from exc

    expected_tiers = list(ENGINE_TIERS.get(engine, ()))
    if level.get("tiers") != expected_tiers:
        raise ValueError(
            f"Engine {engine} must map to {expected_tiers}, got {level.get('tiers')}"
        )

    watermarks = level.get("watermarks", {})
    high = {tier: values["high"] for tier, values in watermarks.items()}
    low = {tier: values["low"] for tier, values in watermarks.items()}
    kwargs = {
        key: value
        for key, value in profile.items()
        if not key.startswith("_")
    }
    kwargs.update(
        cache_engine=engine,
        kv_high_watermarks=high,
        kv_low_watermarks=low,
        kv_swap_policy=level.get("policy", "waterfall_lru"),
        kv_max_transfer_blocks=level.get("max_transfer_blocks", -1),
    )
    return kwargs, {
        "engine": engine,
        "tiers": expected_tiers,
        "capacity_profile": level["capacity_profile"],
        "watermarks": watermarks,
        "policy": kwargs["kv_swap_policy"],
        "max_transfer_blocks": kwargs["kv_max_transfer_blocks"],
        "capacity": {
            key: value for key, value in profile.items() if not key.startswith("_")
        },
    }


def load_benchmark_preset(name: str, path: Path = DEFAULT_BENCHMARK_CONFIG):
    document = _load_json_yaml(path)
    try:
        preset = document["presets"][name]
    except KeyError as exc:
        raise ValueError(f"Unknown benchmark preset {name!r} in {path}") from exc
    warmup_runs = int(preset["warmup_runs"])
    measurement_repeats = int(preset["measurement_repeats"])
    if warmup_runs < 0 or measurement_repeats <= 0:
        raise ValueError("Benchmark run counts must be non-negative/positive")
    return {
        "name": name,
        "warmup_runs": warmup_runs,
        "measurement_repeats": measurement_repeats,
    }
