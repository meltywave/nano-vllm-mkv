import sys
sys.path.insert(0, ".")
import torch
from nanovllm.engine.block_manager import BlockManager

# 模拟简易 ModelRunner 接口，只实现BlockManager需要的两个方法
class MockModelRunner:
    def __init__(self):
        self.block_data = dict()

    def extract_block(self, block_id):
        # 模拟从本地CPU/SSD取出KV张量
        return self.block_data[block_id]

    def clear_block(self, block_id):
        # 清空本地存储
        if block_id in self.block_data:
            del self.block_data[block_id]

    def restore_block(self, block_id, tensor):
        # 远端拉回的数据存入本地次级存储
        self.block_data[block_id] = tensor

def test_blockmanager_remote_logic():
    # 开启远端模块
    bm = BlockManager(num_blocks=16, block_size=32, enable_remote=True)
    mock_runner = MockModelRunner()

    # 1. 分配块
    bid = bm._allocate_block()
    blk = bm.blocks[bid]

    # 模拟序列使用完毕，释放引用（关键！ref_count必须置0才能被驱逐）
    blk.ref_count = 0
    # 模拟块被逐出GPU，进入本地CPU/SSD
    blk.evicted_local = True
    mock_runner.block_data[bid] = torch.randn(128,64)

    print(f"块{bid} 状态 ref_count={blk.ref_count}, on_remote={blk.on_remote}, evicted_local={blk.evicted_local}")
    print(f"LRU队列当前: {list(bm.lru_queue)}")

    # 2. 执行远端驱逐（你的接口）
    out_id = bm.evict_to_remote(mock_runner)
    print(f"成功驱逐块 {out_id} 到远端")
    print(f"驱逐后 on_remote={blk.on_remote}, evicted_local={blk.evicted_local}")

    # 3. 模拟后续访问该块，自动拉回本地
    bm.load_from_remote(out_id, mock_runner)
    print(f"拉回后 on_remote={blk.on_remote}, evicted_local={blk.evicted_local}")

    # 打印统计
    bm.print_swap_statistics()

if __name__ == "__main__":
    test_blockmanager_remote_logic()