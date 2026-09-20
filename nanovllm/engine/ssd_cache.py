import os

import torch


class SSDCache:

    def __init__(
        self,
        cache_dir: str,
        cache_id: str,
        rank: int,
        num_blocks: int,
        block_shape: tuple[int, ...],
        dtype: torch.dtype,
    ):
        os.makedirs(cache_dir, exist_ok=True)
        self.path = os.path.join(
            cache_dir, f"nanovllm-kv-{cache_id}-rank{rank}.cache"
        )
        self.num_blocks = num_blocks
        self.staging = torch.empty(
            block_shape,
            dtype=dtype,
            device="cpu",
            pin_memory=True,
        )
        self.block_bytes = self.staging.numel() * self.staging.element_size()
        self.file = open(self.path, "w+b", buffering=0)
        self.file.truncate(num_blocks * self.block_bytes)

    def _seek(self, block_id: int):
        if not 0 <= block_id < self.num_blocks:
            raise IndexError(f"SSD KV block id {block_id} is out of range.")
        self.file.seek(block_id * self.block_bytes)

    def write_block(
        self,
        block_id: int,
        source: torch.Tensor,
        stream: torch.cuda.Stream,
    ):
        if source.device.type == "cuda":
            with torch.cuda.stream(stream):
                self.staging.copy_(source, non_blocking=True)
            stream.synchronize()
        else:
            self.staging.copy_(source)

        data = self.staging.view(torch.uint8).reshape(-1).numpy()
        self._seek(block_id)
        written = self.file.write(data)
        if written != self.block_bytes:
            raise OSError(
                f"Short write for SSD KV block {block_id}: "
                f"{written} of {self.block_bytes} bytes."
            )

    def read_block(
        self,
        block_id: int,
        destination: torch.Tensor,
        stream: torch.cuda.Stream,
    ):
        self._seek(block_id)
        data = self.staging.view(torch.uint8).reshape(-1).numpy()
        bytes_read = self.file.readinto(data)
        if bytes_read != self.block_bytes:
            raise OSError(
                f"Short read for SSD KV block {block_id}: "
                f"{bytes_read} of {self.block_bytes} bytes."
            )

        if destination.device.type == "cuda":
            with torch.cuda.stream(stream):
                destination.copy_(self.staging, non_blocking=True)
            stream.synchronize()
        else:
            destination.copy_(self.staging)

    def close(self):
        if self.file.closed:
            return
        # KV data is ephemeral. Truncation releases disk space while leaving
        # the cache marker in place for diagnostics.
        self.file.truncate(0)
        self.file.close()
