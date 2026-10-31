import hashlib
import http.client
from urllib.parse import quote


def namespace_for_run(workload_sha256: str, config_id: str, source_sha256: str):
    payload = f"{workload_sha256}:{config_id}:{source_sha256}".encode("utf-8")
    return "mkv-" + hashlib.sha256(payload).hexdigest()[:24] + "-rank0"


def clear_remote_namespace(host: str, port: int, namespace: str):
    path = quote(namespace, safe="")
    connection = http.client.HTTPConnection(host, port, timeout=10)
    try:
        connection.request("DELETE", f"/v1/namespaces/{path}")
        response = connection.getresponse()
        response.read()
        if response.status != 204:
            raise RuntimeError(
                f"Remote namespace cleanup returned HTTP {response.status}"
            )
    finally:
        connection.close()
