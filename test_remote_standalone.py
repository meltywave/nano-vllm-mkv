# test_remote_standalone.py
import sys
sys.path.insert(0, ".")
import torch
from nanovllm.config import Config
from nanovllm.engine.block_manager import BlockManager

class MockModelRunner:
    def __init__(self):
        self.block_data = {}       # GPU 数据
        self.cpu_cache = {}        # CPU 缓存
        self.ssd_cache = {}        # SSD 缓存

    def extract_block(self, block_id):
        return self.block_data.get(block_id)

    def clear_block(self, block_id):
        if block_id in self.block_data:
            del self.block_data[block_id]

    def restore_block(self, block_id, tensor):
        self.block_data[block_id] = tensor

    # 模拟 CPU 同学接口
    def evict_to_cpu(self, block_id, tensor):
        self.cpu_cache[block_id] = tensor
        print(f"    [CPU层] 块 {block_id} GPU → CPU, 大小: {tensor.numel() * 4 / 1024 / 1024:.2f} MB")

    def load_from_cpu(self, block_id):
        return self.cpu_cache.get(block_id)

    # 模拟 SSD 同学接口
    def evict_to_ssd(self, block_id, tensor):
        self.ssd_cache[block_id] = tensor
        print(f"    [SSD层] 块 {block_id} CPU → SSD, 大小: {tensor.numel() * 4 / 1024 / 1024:.2f} MB")

    def load_from_ssd(self, block_id):
        return self.ssd_cache.get(block_id)

    def clear_ssd(self, block_id):
        if block_id in self.ssd_cache:
            del self.ssd_cache[block_id]


def main():
    cfg = Config(
        model="./models/Qwen2-0.5B",
        enable_remote_swap=True,
        remote_host="127.0.0.1",
        remote_port=12345,
    )
    
    bm = BlockManager(num_blocks=4, block_size=256, cfg=cfg)
    runner = MockModelRunner()
    bm.set_model_runner(runner)

    print("=" * 60)
    print("多级KV Swap 完整链路验证")
    print("GPU → CPU → SSD → 远端")
    print("=" * 60)

    # 1. 分配块，生成KV数据
    bid = bm._allocate_block()
    blk = bm.blocks[bid]
    blk.ref_count = 0
    fake_kv = torch.randn(2, 24, 256, 2, 128)
    runner.block_data[bid] = fake_kv
    print(f"\n[1] 块 {bid} 已生成KV数据，大小: {fake_kv.numel() * 4 / 1024 / 1024:.2f} MB")

    # 2. GPU → CPU（CPU同学）
    blk.evicted_local = True
    bm.local_swap_cache[bid] = fake_kv
    runner.evict_to_cpu(bid, fake_kv)
    print(f"[2] GPU → CPU: 成功")

    # 3. CPU → SSD（SSD同学）
    tensor = runner.load_from_cpu(bid)
    runner.evict_to_ssd(bid, tensor)
    print(f"[3] CPU → SSD: 成功")

    # 4. SSD → 远端（你的模块）
    victim = bm.evict_to_remote(runner)
    print(f"[4] SSD → 远端: {'成功' if victim is not None else '失败'}")

    # 5. 远端 → SSD（你的模块）
    ok = bm.load_from_remote(bid, runner)
    print(f"[5] 远端 → SSD: {'成功' if ok else '失败'}")

    # 6. SSD → CPU（SSD同学）
    tensor = runner.load_from_ssd(bid)
    bm.local_swap_cache[bid] = tensor
    blk.evicted_local = True
    print(f"[6] SSD → CPU: {'成功' if tensor is not None else '失败'}")

    # 7. CPU → GPU（CPU同学）
    ok = bm.load_to_gpu(bid, runner)
    print(f"[7] CPU → GPU: {'成功' if ok else '失败'}")

    # 8. 统计
    print("\n" + "=" * 60)
    bm.print_swap_statistics()

if __name__ == "__main__":
    main()