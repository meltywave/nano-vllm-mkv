import http.client
from urllib.parse import quote

import torch


class RemoteCacheClient:

    def __init__(self, host: str, port: int, namespace: str, timeout: float):
        self.host = host
        self.port = port
        self.namespace = namespace
        self.timeout = timeout
        self.connection = None

    def _path(self, block_id: int):
        namespace = quote(self.namespace, safe="")
        return f"/v1/blocks/{namespace}/{block_id}"

    def _connect(self):
        if self.connection is None:
            self.connection = http.client.HTTPConnection(
                self.host, self.port, timeout=self.timeout
            )
        return self.connection

    def _reset(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def put_block(self, block_id: int, data):
        view = memoryview(data).cast("B")
        for attempt in range(2):
            try:
                connection = self._connect()
                connection.request(
                    "PUT",
                    self._path(block_id),
                    body=view,
                    headers={"Content-Length": str(len(view))},
                )
                response = connection.getresponse()
                body = response.read()
                if response.status != 204:
                    raise RuntimeError(
                        f"Remote cache PUT failed with HTTP {response.status}: "
                        f"{body.decode(errors='replace')}"
                    )
                return
            except (OSError, http.client.HTTPException):
                self._reset()
                if attempt == 1:
                    raise

    def get_block(self, block_id: int, destination):
        view = memoryview(destination).cast("B")
        for attempt in range(2):
            try:
                connection = self._connect()
                connection.request("GET", self._path(block_id))
                response = connection.getresponse()
                if response.status != 200:
                    body = response.read()
                    raise RuntimeError(
                        f"Remote cache GET failed with HTTP {response.status}: "
                        f"{body.decode(errors='replace')}"
                    )
                content_length = int(
                    response.getheader("Content-Length", "-1")
                )
                if content_length != len(view):
                    response.read()
                    raise RuntimeError(
                        f"Remote KV block has {content_length} bytes, "
                        f"expected {len(view)}."
                    )
                offset = 0
                while offset < len(view):
                    bytes_read = response.readinto(view[offset:])
                    if not bytes_read:
                        raise OSError("Remote KV block response ended early.")
                    offset += bytes_read
                return
            except (OSError, http.client.HTTPException):
                self._reset()
                if attempt == 1:
                    raise

    def health(self):
        for attempt in range(2):
            try:
                connection = self._connect()
                connection.request("GET", "/health")
                response = connection.getresponse()
                body = response.read()
                if response.status != 200:
                    raise RuntimeError(
                        f"Remote cache health check failed with HTTP "
                        f"{response.status}: {body.decode(errors='replace')}"
                    )
                return
            except (OSError, http.client.HTTPException):
                self._reset()
                if attempt == 1:
                    raise

    def clear(self):
        namespace = quote(self.namespace, safe="")
        connection = self._connect()
        connection.request("DELETE", f"/v1/namespaces/{namespace}")
        response = connection.getresponse()
        body = response.read()
        if response.status != 204:
            raise RuntimeError(
                f"Remote cache DELETE failed with HTTP {response.status}: "
                f"{body.decode(errors='replace')}"
            )

    def close(self):
        try:
            self.clear()
        except (OSError, http.client.HTTPException, RuntimeError):
            pass
        finally:
            self._reset()


class RemoteCache:

    def __init__(
        self,
        host: str,
        port: int,
        timeout: float,
        namespace: str,
        num_blocks: int,
        block_shape: tuple[int, ...],
        dtype: torch.dtype,
    ):
        self.num_blocks = num_blocks
        self.staging = torch.empty(
            block_shape,
            dtype=dtype,
            device="cpu",
            pin_memory=True,
        )
        self.block_bytes = self.staging.numel() * self.staging.element_size()
        self.client = RemoteCacheClient(host, port, namespace, timeout)
        self.client.health()

    def _check_block_id(self, block_id: int):
        if not 0 <= block_id < self.num_blocks:
            raise IndexError(f"Remote KV block id {block_id} is out of range.")

    def write_block(
        self,
        block_id: int,
        source: torch.Tensor,
        stream: torch.cuda.Stream,
    ):
        self._check_block_id(block_id)
        if source.device.type == "cuda":
            with torch.cuda.stream(stream):
                self.staging.copy_(source, non_blocking=True)
            stream.synchronize()
        else:
            self.staging.copy_(source)
        data = self.staging.view(torch.uint8).reshape(-1).numpy()
        self.client.put_block(block_id, data)

    def read_block(
        self,
        block_id: int,
        destination: torch.Tensor,
        stream: torch.cuda.Stream,
    ):
        self._check_block_id(block_id)
        data = self.staging.view(torch.uint8).reshape(-1).numpy()
        self.client.get_block(block_id, data)
        if destination.device.type == "cuda":
            with torch.cuda.stream(stream):
                destination.copy_(self.staging, non_blocking=True)
            stream.synchronize()
        else:
            destination.copy_(self.staging)

    def close(self):
        self.client.close()
