import sys
from pathlib import Path
# 自动定位项目根目录
root_path = Path(__file__).parent.parent
sys.path.insert(0, str(root_path))
import torch
# 适配目录：config在nanovllm下
from nanovllm.config import Config
from nanovllm.engine.block_manager import BlockManager

# Mock ModelRunner 不变
class MockModelRunner:
    def __init__(self):
        self.block_data = dict()

    def extract_block(self, block_id):
        return self.block_data[block_id]

    def clear_block(self, block_id):
        if block_id in self.block_data:
            del self.block_data[block_id]

    def restore_block(self, block_id, tensor):
        self.block_data[block_id] = tensor

def test_blockmanager_remote_logic():
    # 初始化全局配置，填入你的模型本地路径
    cfg = Config(
    model="/home/aurora/work/kvcache/nano-vllm/models/Qwen2-0.5B",
    enable_remote_swap=True,
    remote_host="127.0.0.1",
    remote_port=12345
    )
    # 传cfg实例，不再传host/port/enable_remote
    bm = BlockManager(num_blocks=16, block_size=32, cfg=cfg)
    mock_runner = MockModelRunner()

    bid = bm._allocate_block()
    blk = bm.blocks[bid]
    blk.ref_count = 0
    blk.evicted_local = True
    mock_runner.block_data[bid] = torch.randn(128,64)

    print(f"块{bid} 状态 ref_count={blk.ref_count}, on_remote={blk.on_remote}, evicted_local={blk.evicted_local}")
    print(f"LRU队列当前: {list(bm.lru_queue)}")

    out_id = bm.evict_to_remote(mock_runner)
    if out_id is not None:
        print(f"成功驱逐块 {out_id} 到远端")
        print(f"驱逐后 on_remote={blk.on_remote}, evicted_local={blk.evicted_local}")
        load_ok = bm.load_from_remote(out_id, mock_runner)
        if load_ok:
            print(f"拉回后 on_remote={blk.on_remote}, evicted_local={blk.evicted_local}")
        else:
            print(f"块{out_id}远端加载失败，保持原有状态 on_remote={blk.on_remote}")
    else:
        print("远端驱逐失败，无有效block_id，跳过拉回逻辑")

    bm.print_swap_statistics()

    print("\n=====测试网络断开降级=====")
    new_bid = bm._allocate_block()
    new_blk = bm.blocks[new_bid]
    new_blk.ref_count = 0
    new_blk.evicted_local = True
    mock_runner.block_data[new_bid] = torch.randn(128,64)

    res = bm.evict_to_remote(mock_runner)
    if res is None:
        print("网络异常，驱逐失败，程序正常继续（降级生效）")
    bm.print_swap_statistics()

if __name__ == "__main__":
    test_blockmanager_remote_logic()