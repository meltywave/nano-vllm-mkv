import os
import torch
os.environ["HF_HUB_OFFLINE"] = "1"
# 强制默认使用CPU，避免自动触发cuda初始化
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from nanovllm.engine.block_manager import BlockManager

def main():
    print("===== Test Swap Out & Swap In =====")
    bm = BlockManager(num_blocks=8, block_size=128)

    # 模拟一块KV张量（缩小尺寸，降低网络传输压力）
    fake_tensor = torch.randn(8, 16)
    # 简易模拟model_runner最小接口（只满足evict调用需求，不用启动完整引擎）
    class MockModelRunner:
        def extract_block(self, block_id):
            return fake_tensor.clone()
        def clear_block(self, block_id):
            pass
        def restore_block(self, block_id, tensor):
            print(f"Mock restore block {block_id}, tensor shape:{tensor.shape}, device:{tensor.device}")

    mock_mr = MockModelRunner()

    # 分配一块block
    bid = bm._allocate_block()
    print(f"Allocated block {bid}")
    # 模拟引用计数归零（变成可驱逐冷块）
    bm.blocks[bid].ref_count = 0

    # Swap Out 驱逐到远端
    victim = bm.evict_to_remote(mock_mr)
    print(f"Evict block {victim} to remote, on_remote={bm.blocks[victim].on_remote}")

    # Swap In 从远端拉回
    bm.load_from_remote(victim, mock_mr)
    print(f"Load block {victim} back, on_remote={bm.blocks[victim].on_remote}")

    print("\n✅ Swap Out / Swap In 链路测试全部完成！")
    bm.print_swap_statistics()

if __name__ == "__main__":
    main()