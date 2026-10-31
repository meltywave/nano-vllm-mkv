import hashlib
import json
import os
import platform
import threading
from pathlib import Path


def output_sha256(outputs):
    payload = [output["token_ids"] for output in outputs]
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tokenizer_files_sha256(model_dir):
    digest = hashlib.sha256()
    root = Path(model_dir)
    for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json"):
        path = root / name
        if not path.exists():
            continue
        digest.update(name.encode("utf-8"))
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def process_rss_bytes():
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss
    except ImportError:
        try:
            import resource

            value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return value if platform.system() == "Darwin" else value * 1024
        except ImportError:
            return None


class PeakRssSampler:
    def __init__(self, interval_s=0.05):
        self.interval_s = interval_s
        self.peak_bytes = None
        self.includes_children = False
        self._stop = threading.Event()
        self._thread = None

    def _sample(self):
        try:
            import psutil
        except ImportError:
            value = process_rss_bytes()
        else:
            process = psutil.Process(os.getpid())
            value = 0
            for item in [process, *process.children(recursive=True)]:
                try:
                    value += item.memory_info().rss
                except (psutil.Error, OSError):
                    continue
            self.includes_children = True
        if value is not None:
            self.peak_bytes = max(self.peak_bytes or 0, value)

    def _run(self):
        while not self._stop.wait(self.interval_s):
            self._sample()

    def start(self):
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()
        self._sample()
        return self.peak_bytes


def percentile(values, percentile_value):
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile_value)
    return ordered[index]
