import hashlib
import json
import os
import platform


def output_sha256(outputs):
    payload = [output["token_ids"] for output in outputs]
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def percentile(values, percentile_value):
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile_value)
    return ordered[index]
