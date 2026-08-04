from collections import deque
import xxhash
import numpy as np
import torch
import logging

from nanovllm.engine.sequence import Sequence

logger = logging.getLogger("block_manager")


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []
        # 远端网络存储标记
        self.on_remote: bool = False

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []
        self.on_remote = False


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int, num_cpu_blocks: int = 0, cfg=None):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        self.cpu_blocks = [Block(i) for i in range(num_cpu_blocks)]
        self.free_cpu_block_ids = deque(range(num_cpu_blocks))

        # ========== 远端网络 Swap 模块 ==========
        self.enable_remote = False
        self.remote_client = None
        self.remote_evict_batch_size = 8

        # 远端统计
        self.swap_out_count = 0
        self.swap_in_count = 0
        self.swap_fail_count = 0
        self.remote_hit = 0
        self.remote_miss = 0
        self.swap_out_local_count = 0
        self.swap_in_local_count = 0
        self.local_hit = 0
        self.local_miss = 0

        if cfg is not None and hasattr(cfg, 'enable_remote_swap') and cfg.enable_remote_swap:
            self.enable_remote = True
            from remote_swap.client import RemoteKVClient
            self.remote_client = RemoteKVClient(
                host=cfg.remote_host,
                port=cfg.remote_port,
                connect_retry=getattr(cfg, 'remote_connect_retry', 5),
                op_retry=getattr(cfg, 'remote_op_retry', 3),
                timeout=getattr(cfg, 'remote_socket_timeout', 10.0)
            )
            self.remote_evict_batch_size = getattr(cfg, 'remote_evict_batch_size', 8)
            if self.remote_client._is_available():
                logger.info(f"[BlockManager] Remote KV client connected to {cfg.remote_host}:{cfg.remote_port}")
            else:
                logger.warning("[BlockManager] Remote connection failed, auto-disabled")
        # ==========================================

    def set_model_runner(self, model_runner):
        self.model_runner = model_runner

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end:
            return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id

    def can_swap_out(self, seq):
        return len(self.free_cpu_block_ids) >= seq.num_blocks

    def swap_out(self, seq, model_runner=None):
        gpu_table = seq.block_table
        cpu_table = []
        mappings = []
        for gpu_id in gpu_table:
            cpu_id = self.free_cpu_block_ids.popleft()
            src, dst = self.blocks[gpu_id], self.cpu_blocks[cpu_id]
            assert dst.ref_count == 0
            dst.ref_count = 1
            dst.update(src.hash, list(src.token_ids))
            cpu_table.append(cpu_id)
            mappings.append((gpu_id, cpu_id))

        for gpu_id in reversed(gpu_table):
            block = self.blocks[gpu_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(gpu_id)

        seq.block_table = cpu_table

        # ========== 远端 Swap Out：CPU → 远端 ==========
        logger.info(f"[DEBUG] swap_out 远端入口: enable_remote={self.enable_remote}, client_ok={self.remote_client is not None and self.remote_client._is_available() if self.remote_client else False}, runner_ok={model_runner is not None}")
        if self.enable_remote and self.remote_client is not None and model_runner is not None:
            if self.remote_client._is_available():
                cpu_kv_cache = model_runner.cpu_kv_cache
                if cpu_kv_cache is None:
                    logger.warning("[Remote Swap Out] cpu_kv_cache is None, skip upload")
                else:
                    for cpu_id in cpu_table:
                        kv_tensor = cpu_kv_cache[:, :, cpu_id, :, :, :].clone()
                        ok = self.remote_client.put_block(cpu_id, kv_tensor)
                        if ok:
                            self.cpu_blocks[cpu_id].on_remote = True
                            self.swap_out_count += 1
                            logger.info(f"[Remote Swap Out] block {cpu_id} CPU → remote")
                        else:
                            self.swap_fail_count += 1
                            logger.error(f"[Remote Swap Out] block {cpu_id} upload failed")
        # ==============================================

        return mappings

    def can_swap_in(self, seq, extra_gpu_blocks=0):
        return len(self.free_block_ids) >= len(seq.block_table) + extra_gpu_blocks

    def swap_in(self, seq, model_runner=None):
        cpu_table = seq.block_table
        gpu_table = []
        mappings = []

        # ========== 远端 Swap In：远端 → CPU ==========
        if self.enable_remote and self.remote_client is not None and model_runner is not None:
            if self.remote_client._is_available():
                cpu_kv_cache = model_runner.cpu_kv_cache
                if cpu_kv_cache is not None:
                    for cpu_id in cpu_table:
                        if self.cpu_blocks[cpu_id].on_remote:
                            kv_tensor = self.remote_client.get_block(cpu_id)
                            if kv_tensor is not None:
                                cpu_kv_cache[:, :, cpu_id, :, :, :].copy_(kv_tensor)
                                self.cpu_blocks[cpu_id].on_remote = False
                                self.swap_in_count += 1
                                self.remote_hit += 1
                                logger.info(f"[Remote Swap In] block {cpu_id} remote → CPU")
                            else:
                                self.swap_fail_count += 1
                                self.remote_miss += 1
                                logger.error(f"[Remote Swap In] block {cpu_id} download failed")
                        else:
                            self.remote_miss += 1
        # ==============================================

        for cpu_id in cpu_table:
            cpu_block = self.cpu_blocks[cpu_id]
            gpu_id = self._allocate_block()
            gpu_block = self.blocks[gpu_id]
            gpu_block.update(cpu_block.hash, list(cpu_block.token_ids))
            if gpu_block.hash != -1:
                self.hash_to_block_id[gpu_block.hash] = gpu_id
            gpu_table.append(gpu_id)
            mappings.append((cpu_id, gpu_id))

        for cpu_id in cpu_table:
            block = self.cpu_blocks[cpu_id]
            block.ref_count = 0
            block.hash = -1
            block.token_ids = []
            self.free_cpu_block_ids.append(cpu_id)

        seq.block_table = gpu_table
        return mappings

    # ========== 远端统计接口 ==========
    def print_swap_statistics(self):
        print("=" * 60)
        print(f"[Remote KV Swap] Statistics Summary:")
        print(f"Total Remote Swap Out (CPU -> remote): {self.swap_out_count}")
        print(f"Total Remote Swap In  (remote -> CPU): {self.swap_in_count}")
        print(f"Total Remote Operation Failures:       {self.swap_fail_count}")
        if self.enable_remote and self.remote_client is not None:
            stats = self.remote_client.get_stats()
            send_mb = stats["total_send_bytes"] / 1024 / 1024
            recv_mb = stats["total_recv_bytes"] / 1024 / 1024
            print(f"Total Upload:   {send_mb:.2f} MB")
            print(f"Total Download: {recv_mb:.2f} MB")
            if stats["send_success"] > 0:
                avg_ms = stats["total_upload_ms"] / stats["send_success"]
                tp = send_mb / (stats["total_upload_ms"] / 1000) if stats["total_upload_ms"] > 0 else 0
                print(f"Avg Upload Latency: {avg_ms:.2f} ms/block")
                print(f"Upload Throughput:  {tp:.2f} MB/s")
            if stats["recv_success"] > 0:
                avg_ms = stats["total_download_ms"] / stats["recv_success"]
                tp = recv_mb / (stats["total_download_ms"] / 1000) if stats["total_download_ms"] > 0 else 0
                print(f"Avg Download Latency: {avg_ms:.2f} ms/block")
                print(f"Download Throughput:  {tp:.2f} MB/s")
        print("=" * 60)