from collections import deque
import xxhash
import numpy as np
import torch
import logging

# 模块独立日志，避免循环导入（打破 nanovllm/__init__.py → llm → scheduler → block_manager 的循环依赖）
logger = logging.getLogger("block_manager")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# 按需延迟导入网络客户端，关闭远端时不加载socket模块
from nanovllm.engine.sequence import Sequence
from nanovllm.config import Config


class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []
        # 是否置换到远端网络服务器
        self.on_remote: bool = False
        # 是否已逐出GPU，存于本地CPU/SSD次级存储
        self.evicted_local: bool = False

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []
        self.on_remote = False
        self.evicted_local = False


class BlockManager:
    # 构造函数：移除所有硬编码远端参数，统一接收全局cfg
    def __init__(self, num_blocks: int, block_size: int, cfg: Config):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

        # 持有model_runner引用，用于KV数据读写（LLMEngine初始化后通过set_model_runner注入）
        self.model_runner = None

        # 远端Swap模块初始化，参数全部从cfg读取
        self.remote_client = None
        self.enable_remote = cfg.enable_remote_swap
        if self.enable_remote:
            from remote_swap.client import RemoteKVClient
            self.remote_client = RemoteKVClient(
                host=cfg.remote_host,
                port=cfg.remote_port,
                connect_retry=cfg.remote_connect_retry,
                op_retry=cfg.remote_op_retry,
                timeout=cfg.remote_socket_timeout
            )
            if self.remote_client._is_available():
                logger.info(f"[BlockManager] Remote KV client connected to {cfg.remote_host}:{cfg.remote_port}")
            else:
                logger.warning(f"[BlockManager WARN] Remote KV client connection failed, remote swap auto-disabled")
        else:
            logger.info(f"[BlockManager] Remote Swap Module DISABLED")

        # LRU冷热队列，头部为最冷块，统一管理所有已分配块（活跃+GPU缓存+CPU缓存+远端缓存）
        self.lru_queue: deque[int] = deque()

        # 本地CPU次级Swap缓存 <block_id, CPU端KV张量>
        self.local_swap_cache: dict[int, torch.Tensor] = dict()

        # ========== 置换统计计数器（全链路指标） ==========
        # 远端Swap统计
        self.swap_out_count = 0        # 本地CPU→远端总次数
        self.swap_in_count = 0         # 远端→本地CPU总次数
        self.swap_fail_count = 0       # 远端操作失败总次数
        self.remote_hit = 0            # 访问块命中远端，触发拉回
        self.remote_miss = 0           # 访问块未命中远端，无需拉回
        self.remote_evict_batch_size = cfg.remote_evict_batch_size

        # 本地Swap统计
        self.swap_out_local_count = 0   # GPU→本地CPU总次数
        self.swap_in_local_count = 0    # 本地CPU→GPU总次数
        self.local_hit = 0              # 访问块命中本地CPU缓存
        self.local_miss = 0             # 访问块未命中本地CPU缓存
        # ======================================================

    def set_model_runner(self, model_runner):
        """LLMEngine创建完ModelRunner后调用，注入KV读写依赖"""
        self.model_runner = model_runner

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, byteorder="big"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        """分配一个空闲物理块，空闲不足时自动驱逐冷缓存回收块"""
        if not self.free_block_ids:
            # 空闲块耗尽，自动驱逐冷缓存块回收物理空间
            self._reclaim_cold_blocks(1)
            if not self.free_block_ids:
                raise RuntimeError("KV缓存物理块耗尽，且无可驱逐冷缓存，无法分配新块")

        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        
        # 如果这个块在 LRU 中（作为冷缓存），说明它之前被 deallocate 过
        # 从 used 移除但保留在 lru_queue，现在重新使用需要清理旧状态
        if block_id in self.lru_queue:
            self.lru_queue.remove(block_id)
        
        # 清理旧的hash映射（物理块复用，旧缓存失效）
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        
        # 如果之前被驱逐到 CPU/远端，需要清理本地缓存
        if block_id in self.local_swap_cache:
            del self.local_swap_cache[block_id]
        
        # 重置块状态（自动清空on_remote、evicted_local等标记）
        block.reset()
        self.used_block_ids.add(block_id)
        self.lru_queue.append(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        """彻底回收物理块到空闲池"""
        if self.blocks[block_id].ref_count != 0:
            logger.warning(f"[BlockManager] _deallocate_block: 块 {block_id} ref_count={self.blocks[block_id].ref_count}，跳过回收")
            return
        
        if block_id in self.used_block_ids:
            self.used_block_ids.remove(block_id)
        if block_id not in self.free_block_ids:
            self.free_block_ids.append(block_id)
        if block_id in self.lru_queue:
            self.lru_queue.remove(block_id)
        if block_id in self.local_swap_cache:
            del self.local_swap_cache[block_id]

    def _reclaim_cold_blocks(self, need_num: int) -> int:
        """
        【核心修复重构】回收冷缓存块的物理空间到空闲池，下沉数据后强制归还物理ID
        回收优先级：
        1. 已在远端的冷块 → 直接回收物理块，缓存保留远端
        2. CPU缓存冷块 → 下沉远端 → 回收物理块
        3. GPU冷块 → 下沉CPU → 下沉远端 → 回收物理块
        :param need_num: 需要回收的块数量
        :return: 实际回收的块数
        """
        if self.model_runner is None:
            logger.warning("[BlockManager WARN] model_runner未设置，无法执行缓存回收")
            return 0

        reclaimed = 0
        for _ in range(need_num):
            # 优先级1：已远端冷块，直接回收物理ID
            victim_id = self._evict_one_to_remote_and_reclaim()
            if victim_id is not None:
                reclaimed += 1
                continue

            # 优先级2：CPU冷块下沉远端后回收
            victim_id = self.evict_to_remote()
            if victim_id is not None:
                self._deallocate_block(victim_id)
                reclaimed += 1
                continue

            # 优先级3：GPU冷块先下沉CPU，再尝试下沉远端回收
            victim_gpu = self.evict_to_local()
            if victim_gpu is not None:
                # GPU 驱逐到 CPU 成功后，立即尝试驱逐到远端来真正回收
                victim_remote = self.evict_to_remote()
                if victim_remote is not None:
                    self._deallocate_block(victim_remote)
                    reclaimed += 1
                    continue

            # 无任何可回收冷块，终止循环
            break

        if reclaimed > 0:
            logger.info(f"[Reclaim] 本轮成功回收 {reclaimed} 个物理KV块，当前空闲块总数：{len(self.free_block_ids)}")
        return reclaimed

    def _evict_one_to_remote_and_reclaim(self) -> int | None:
        """
        从LRU中查找已完全下沉远端、无引用的冷块，直接回收物理块
        缓存数据永久保存在远端存储，不丢失
        """
        for bid in list(self.lru_queue):
            blk = self.blocks[bid]
            if blk.ref_count == 0 and blk.on_remote:
                self.lru_queue.remove(bid)
                self.used_block_ids.remove(bid)
                self.free_block_ids.append(bid)
                logger.info(f"[Reclaim Direct] 回收远端冷块 {bid}，缓存留存远端")
                return bid
        return None

    def evict_to_local(self, model_runner=None) -> int | None:
        runner = model_runner if model_runner is not None else self.model_runner
        if runner is None:
            logger.warning("[BlockManager WARN] model_runner未设置，无法执行本地驱逐")
            return None

        max_scan = len(self.lru_queue)
        scan_cnt = 0
        while self.lru_queue and scan_cnt < max_scan:
            scan_cnt += 1
            victim_id = self.lru_queue.popleft()
            victim_block = self.blocks[victim_id]

            # 只驱逐 ref_count == 0 且不在 CPU/远端的 GPU 块
            if victim_block.ref_count > 0:
                self.lru_queue.append(victim_id)
                continue
            if victim_block.evicted_local or victim_block.on_remote:
                self.lru_queue.append(victim_id)
                continue

            kv_tensor = runner.extract_block(victim_id)
            self.local_swap_cache[victim_id] = kv_tensor.cpu()
            runner.clear_block(victim_id)
            victim_block.evicted_local = True
            self.lru_queue.append(victim_id)
            self.swap_out_local_count += 1
            logger.info(f"[Local Swap Out] block {victim_id} GPU -> CPU | ref_count=0")
            return victim_id

        logger.info("[BlockManager INFO] 遍历全部LRU，无符合条件的GPU冷块可驱逐至本地CPU")
        return None

    def evict_to_remote(self, model_runner=None) -> int | None:
        if not self.enable_remote or self.remote_client is None:
            logger.warning("[BlockManager WARN] Remote swap module is not enabled!")
            return None

        if not self.remote_client._is_available():
            logger.warning("[BlockManager WARN] 远端服务不可用，跳过本次远端驱逐")
            self.swap_fail_count += 1
            return None

        max_scan = len(self.lru_queue)
        scan_cnt = 0
        while self.lru_queue and scan_cnt < max_scan:
            scan_cnt += 1
            victim_id = self.lru_queue.popleft()
            victim_block = self.blocks[victim_id]

            # 只驱逐已在本地的冷块到远端
            if victim_block.ref_count > 0:
                self.lru_queue.append(victim_id)
                continue
            if victim_block.on_remote:
                self.lru_queue.append(victim_id)
                continue
            if not victim_block.evicted_local:
                self.lru_queue.append(victim_id)
                continue

            if victim_id not in self.local_swap_cache:
                logger.error(f"[BlockManager ERROR] 块 {victim_id} 标记为本地存储，但本地缓存中不存在")
                victim_block.evicted_local = False
                self.lru_queue.append(victim_id)
                self.swap_fail_count += 1
                return None

            kv_tensor = self.local_swap_cache[victim_id]
            upload_ok = self.remote_client.put_block(victim_id, kv_tensor.cpu())

            if not upload_ok:
                logger.error(f"[BlockManager WARN] 块 {victim_id} 上传远端失败，保留本地存储")
                self.lru_queue.append(victim_id)
                self.swap_fail_count += 1
                return None

            del self.local_swap_cache[victim_id]
            victim_block.on_remote = True
            victim_block.evicted_local = False
            self.lru_queue.append(victim_id)
            logger.info(f"[Remote Swap Out] block {victim_id} CPU -> remote | ref_count=0")
            self.swap_out_count += 1
            return victim_id

        logger.info("[BlockManager INFO] 遍历全部LRU，没有满足条件的本地冷块可驱逐至远端")
        return None

    def load_from_remote(self, block_id: int, model_runner=None) -> bool:
        """
        【第二层Swap回读】远端服务器 → 本地CPU内存次级存储
        分层约束：拉回数据先落地CPU，不直接写回GPU
        返回True成功 / False失败
        """
        block = self.blocks[block_id]
        if not block.on_remote:
            return True
        if not self.enable_remote or self.remote_client is None:
            logger.warning("[BlockManager WARN] Remote swap module is not enabled!")
            return False

        if not self.remote_client._is_available():
            logger.warning(f"[BlockManager WARN] 远端服务不可用，无法加载块 {block_id}")
            self.swap_fail_count += 1
            return False

        kv_tensor = self.remote_client.get_block(block_id)
        if kv_tensor is None:
            logger.error(f"[BlockManager WARN] 远端读取块 {block_id} 失败，跳过加载")
            self.swap_fail_count += 1
            return False

        # 落地本地CPU缓存，不直接写GPU
        self.local_swap_cache[block_id] = kv_tensor.cpu()
        block.on_remote = False
        block.evicted_local = True
        # 重新加入LRU队列尾部（刚被访问，热度升高）
        self.lru_queue.append(block_id)
        logger.info(f"[Remote Swap In] block {block_id} remote -> CPU")
        self.swap_in_count += 1
        return True

    def load_to_gpu(self, block_id: int, model_runner=None) -> bool:
        """
        【第一层Swap回读】本地CPU内存 → GPU显存
        分层约束：仅CPU缓存块可加载到GPU，远端块必须先拉回CPU
        返回True成功 / False失败
        """
        runner = model_runner if model_runner is not None else self.model_runner
        if runner is None:
            logger.warning("[BlockManager WARN] model_runner未设置，无法加载到GPU")
            return False

        block = self.blocks[block_id]
        if not block.evicted_local:
            return True
        if block_id not in self.local_swap_cache:
            logger.error(f"[BlockManager ERROR] 块 {block_id} 标记为本地存储，但本地缓存中不存在")
            block.evicted_local = False
            self.swap_fail_count += 1
            return False

        kv_tensor = self.local_swap_cache[block_id]
        # 写回GPU显存
        runner.restore_block(block_id, kv_tensor)
        # 删除本地CPU副本，更新块状态
        del self.local_swap_cache[block_id]
        block.evicted_local = False
        # 更新LRU热度，移至队尾
        self._touch_block(block_id)
        logger.info(f"[Local Swap In] block {block_id} CPU -> GPU")
        self.swap_in_local_count += 1
        return True

    # 批量驱逐默认读取cfg配置，不再写死1个
    def try_evict_cold_local_to_remote(self, model_runner=None, max_evict_num: int = None) -> list[int]:
        """
        上层调度批量驱逐对外接口，返回成功驱逐块ID列表
        :param max_evict_num: 单次最大驱逐数量，不传则使用cfg.remote_evict_batch_size
        """
        evicted_list = []
        if not self.enable_remote:
            return evicted_list
        if not self.remote_client._is_available():
            return evicted_list

        # 使用配置默认批量大小
        if max_evict_num is None:
            max_evict_num = self.remote_evict_batch_size

        for _ in range(max_evict_num):
            bid = self.evict_to_remote(model_runner)
            if bid is None:
                break
            evicted_list.append(bid)
        return evicted_list

    def _touch_block(self, block_id: int):
        """访问块，更新LRU热度移至队尾"""
        if block_id in self.lru_queue:
            self.lru_queue.remove(block_id)
            self.lru_queue.append(block_id)

    def print_swap_statistics(self):
        """打印全链路四级置换统计：次数、命中率、流量、延迟、吞吐量"""
        print("=" * 60)
        print(f"[KV Cache Multi-Level Swap] Statistics Summary:")
        print("-" * 60)
        print(f"【第一层：GPU ↔ 本地CPU Swap】")
        print(f"Total Local Swap Out (GPU -> CPU)   : {self.swap_out_local_count}")
        print(f"Total Local Swap In  (CPU -> GPU)   : {self.swap_in_local_count}")
        # 本地缓存命中率
        total_local_access = self.local_hit + self.local_miss
        if total_local_access > 0:
            hit_rate = self.local_hit / total_local_access
            print(f"Local Cache Hit Rate                 : {hit_rate:.2%}")
            print(f"Local Cache Hit Count                : {self.local_hit}")
            print(f"Local Cache Miss Count               : {self.local_miss}")

        print("-" * 60)
        print(f"【第二层：本地CPU ↔ 远端网络 Swap】")
        print(f"Total Remote Swap Out (CPU -> remote): {self.swap_out_count}")
        print(f"Total Remote Swap In  (remote -> CPU): {self.swap_in_count}")
        print(f"Total Remote Operation Failures      : {self.swap_fail_count}")

        # 远端块命中率统计
        total_remote_access = self.remote_hit + self.remote_miss
        if total_remote_access > 0:
            hit_rate = self.remote_hit / total_remote_access
            print(f"Remote Block Hit Rate                : {hit_rate:.2%}")
            print(f"Remote Block Hit Count               : {self.remote_hit}")
            print(f"Remote Block Miss Count              : {self.remote_miss}")

        if self.enable_remote and self.remote_client is not None:
            client_stats = self.remote_client.get_stats()
            send_mb = client_stats["total_send_bytes"] / 1024 / 1024
            recv_mb = client_stats["total_recv_bytes"] / 1024 / 1024
            total_upload_ms = client_stats["total_upload_ms"]
            total_download_ms = client_stats["total_download_ms"]
            send_success = client_stats["send_success"]
            recv_success = client_stats["recv_success"]

            print(f"Total Upload Data Size               : {send_mb:.2f} MB")
            print(f"Total Download Data Size             : {recv_mb:.2f} MB")
            print(f"Network Retry Count                  : {client_stats['retry_count']}")
            print(f"Client-side Fail Count               : {client_stats['fail_count']}")

            # 平均延迟、吞吐量计算
            if send_success > 0:
                avg_upload_ms = total_upload_ms / send_success
                upload_throughput = send_mb / (total_upload_ms / 1000) if total_upload_ms > 0 else 0.0
                print(f"Average Upload Latency               : {avg_upload_ms:.2f} ms/block")
                print(f"Upload Throughput                    : {upload_throughput:.2f} MB/s")
            if recv_success > 0:
                avg_download_ms = total_download_ms / recv_success
                download_throughput = recv_mb / (total_download_ms / 1000) if total_download_ms > 0 else 0.0
                print(f"Average Download Latency             : {avg_download_ms:.2f} ms/block")
                print(f"Download Throughput                  : {download_throughput:.2f} MB/s")
        print("=" * 60)

    # ========== 核心修复：修正 can_allocate 循环边界、块命中计数逻辑 ==========
    def can_allocate(self, seq: Sequence) -> int:
        h = -1
        num_cached_blocks = 0
        max_block_idx = seq.num_blocks
        for i in range(max_block_idx):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            # hash不匹配，前缀缓存断裂，停止统计命中块
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
        # 返回连续命中的前缀块数量，调度器用 seq.num_blocks - num_cached_blocks 得到需要分配的新块
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int, model_runner=None):
        assert not seq.block_table
        runner = model_runner if model_runner is not None else self.model_runner
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]

            # 分层缓存命中处理：严格按 远端→CPU→GPU 顺序逐层加载
            if block.on_remote and runner is not None:
                # 远端命中：先拉回CPU，再加载到GPU
                self.remote_hit += 1
                self.local_miss += 1
                success = self.load_from_remote(block_id, runner)
                if not success:
                    raise RuntimeError(f"从远端加载块{block_id}失败")
                success = self.load_to_gpu(block_id, runner)
                if not success:
                    raise RuntimeError(f"从本地加载块{block_id}到GPU失败")
                # 远端块若已被回收物理块（在free中），需重新加入used集合
                if block_id not in self.used_block_ids:
                    self.free_block_ids.remove(block_id)
                    self.used_block_ids.add(block_id)
            elif block.evicted_local and runner is not None:
                # 本地CPU命中：直接加载到GPU
                self.remote_miss += 1
                self.local_hit += 1
                success = self.load_to_gpu(block_id, runner)
                if not success:
                    raise RuntimeError(f"从本地加载块{block_id}到GPU失败")
            else:
                # GPU命中：无需加载
                self.remote_miss += 1
                self.local_miss += 1

            if block_id in self.used_block_ids:
                block.ref_count += 1
                self._touch_block(block_id)
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
                self.lru_queue.append(block_id)
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        """释放序列占用的块，ref_count 降为 0 的块保留为冷缓存"""
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                # 【核心修复】不直接回收物理块，保留在 LRU 中作为冷缓存
                # 块回到 free 池，但仍保留在 lru_queue 中供后续驱逐判断
                if block_id in self.used_block_ids:
                    self.used_block_ids.remove(block_id)
                if block_id not in self.free_block_ids:
                    self.free_block_ids.append(block_id)
                # 注意：不从 lru_queue 移除，保留为冷缓存
                logger.info(f"[Deallocate] 块 {block_id} 释放引用，保留为冷缓存，hash={block.hash}")
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        # 追加块前检查空闲块是否充足，不足则提前驱逐冷缓存回收
        need = 1 if len(seq) % self.block_size == 1 else 0
        if need > 0 and len(self.free_block_ids) < need:
            self._reclaim_cold_blocks(need - len(self.free_block_ids))
        return len(self.free_block_ids) >= need

    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        
        # 【保护】确保索引不越界
        if start >= len(seq.block_table) or start == end:
            return
        if end > len(seq.block_table):
            end = len(seq.block_table)
        
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id