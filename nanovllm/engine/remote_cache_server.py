import argparse
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit


class RemoteBlockStore:

    def __init__(self, max_bytes: int = 0):
        if max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        self.max_bytes = max_bytes
        self.used_bytes = 0
        self.blocks: dict[tuple[str, int], bytes] = {}
        self.lock = threading.Lock()

    def put(self, namespace: str, block_id: int, data: bytes) -> bool:
        key = (namespace, block_id)
        with self.lock:
            old_size = len(self.blocks.get(key, b""))
            new_used = self.used_bytes - old_size + len(data)
            if self.max_bytes and new_used > self.max_bytes:
                return False
            self.blocks[key] = data
            self.used_bytes = new_used
        return True

    def get(self, namespace: str, block_id: int) -> bytes | None:
        with self.lock:
            return self.blocks.get((namespace, block_id))

    def clear(self, namespace: str):
        with self.lock:
            keys = [key for key in self.blocks if key[0] == namespace]
            for key in keys:
                self.used_bytes -= len(self.blocks.pop(key))


class RemoteCacheRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    @property
    def store(self) -> RemoteBlockStore:
        return self.server.store

    def _send(self, status: int, body=b"", content_type="text/plain"):
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Type", content_type)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _parse_block_path(self):
        parts = urlsplit(self.path).path.strip("/").split("/")
        if len(parts) != 4 or parts[:2] != ["v1", "blocks"]:
            return None
        namespace = unquote(parts[2])
        if not namespace or "/" in namespace:
            return None
        try:
            block_id = int(parts[3])
        except ValueError:
            return None
        if block_id < 0:
            return None
        return namespace, block_id

    def _parse_namespace_path(self):
        parts = urlsplit(self.path).path.strip("/").split("/")
        if len(parts) != 3 or parts[:2] != ["v1", "namespaces"]:
            return None
        namespace = unquote(parts[2])
        if not namespace or "/" in namespace:
            return None
        return namespace

    def do_PUT(self):
        key = self._parse_block_path()
        if key is None:
            self._send(404, b"unknown endpoint")
            return
        try:
            content_length = int(self.headers.get("Content-Length", "-1"))
        except ValueError:
            content_length = -1
        if content_length < 0:
            self._send(411, b"Content-Length is required")
            return

        data = self.rfile.read(content_length)
        if len(data) != content_length:
            self.close_connection = True
            self._send(400, b"incomplete request body")
            return
        if not self.store.put(*key, data):
            self._send(507, b"remote cache capacity exceeded")
            return
        self._send(204)

    def do_GET(self):
        if urlsplit(self.path).path == "/health":
            self._send(200, b"ok")
            return
        key = self._parse_block_path()
        if key is None:
            self._send(404, b"unknown endpoint")
            return
        data = self.store.get(*key)
        if data is None:
            self._send(404, b"block not found")
            return
        self._send(200, data, "application/octet-stream")

    def do_DELETE(self):
        namespace = self._parse_namespace_path()
        if namespace is None:
            self._send(404, b"unknown endpoint")
            return
        self.store.clear(namespace)
        self._send(204)

    def log_message(self, format, *args):
        pass


class RemoteCacheServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, max_bytes: int = 0):
        super().__init__(address, RemoteCacheRequestHandler)
        self.store = RemoteBlockStore(max_bytes)


def main():
    parser = argparse.ArgumentParser(
        description="Serve nano-vLLM remote KV cache blocks over HTTP."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=19090)
    parser.add_argument(
        "--max-gb",
        type=float,
        default=0.0,
        help="Maximum in-memory cache size; zero means unlimited.",
    )
    args = parser.parse_args()
    if args.max_gb < 0:
        parser.error("--max-gb must be non-negative")

    server = RemoteCacheServer(
        (args.host, args.port), int(args.max_gb * 1024**3)
    )
    print(
        f"Remote KV cache listening on {args.host}:{args.port} "
        f"(max_gb={args.max_gb})"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
