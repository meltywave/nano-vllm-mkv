"""
三级 KV Cache 可视化演示
GPU + CPU + SSD 三级缓存工作原理展示

功能：
- 模拟多个序列的分配和释放
- 展示瀑布式换出（GPU → CPU → DISK）
- 展示逐级换入（DISK → CPU → GPU）
- 实时显示各层级使用率
- 展示前缀缓存命中情况
"""

import time
import sys
import os
import types
import importlib.util

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ==================== Mock 外部依赖 ====================
# 模拟 torch，避免真实 GPU 依赖
def _setup_mock():
    """设置 mock 环境"""
    # Mock torch
    torch_mock = types.ModuleType("torch")

    class _FakeTensor:
        def __init__(self, *shape, **kwargs):
            self._shape = shape
            self._dtype = kwargs.get("dtype", None)
            self._data = [0] * (shape[0] if shape else 0)

        def numel(self):
            n = 1
            for s in self._shape:
                n *= s
            return n

        def element_size(self):
            return 2  # float16

        def size(self, dim=None):
            if dim is not None:
                return self._shape[dim]
            return self._shape

        def __getitem__(self, idx):
            return _FakeTensor(16, 8, 16)

        def __setitem__(self, idx, val):
            pass

        def clone(self):
            return _FakeTensor(*self._shape)

        def copy_(self, src, non_blocking=False):
            return self

        def to(self, *args, **kwargs):
            return self

        def cuda(self):
            return self

        def cpu(self):
            return self

        def pin_memory(self):
            return self

        def fill_(self, val):
            return self

        def zero_(self):
            return self

    def _empty(*shape, **kwargs):
        return _FakeTensor(*shape, **kwargs)

    torch_mock.empty = _empty
    torch_mock.Tensor = _FakeTensor
    torch_mock.float16 = "float16"
    torch_mock.float32 = "float32"
    torch_mock.bfloat16 = "bfloat16"

    # Mock cuda
    cuda_mock = types.ModuleType("torch.cuda")

    class _FakeStream:
        def __init__(self):
            pass

        def synchronize(self):
            pass

    class _FakeEvent:
        def __init__(self):
            pass

        def record(self, stream=None):
            pass

        def synchronize(self):
            pass

    cuda_mock.Stream = _FakeStream
    cuda_mock.Event = _FakeEvent
    cuda_mock.synchronize = lambda: None
    cuda_mock.mem_get_info = lambda: (8 * 1024**3, 16 * 1024**3)
    cuda_mock.is_available = lambda: True
    cuda_mock.device_count = lambda: 1

    # Mock stream context
    class _StreamContext:
        def __init__(self, stream):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    cuda_mock.stream = _StreamContext

    torch_mock.cuda = cuda_mock

    # Mock multiprocessing
    mp_mock = types.ModuleType("torch.multiprocessing")

    class _FakeCtx:
        def Process(self, target=None, args=()):
            return _FakeProcess(target, args)

    class _FakeProcess:
        def __init__(self, target, args):
            self.target = target
            self.args = args

        def start(self):
            pass

        def join(self):
            pass

    mp_mock.get_context = lambda method=None: _FakeCtx()
    torch_mock.multiprocessing = mp_mock

    # Mock distributed
    dist_mock = types.ModuleType("torch.distributed")
    dist_mock.init_process_group = lambda **kwargs: None
    dist_mock.get_rank = lambda: 0
    dist_mock.get_world_size = lambda: 1
    dist_mock.barrier = lambda: None
    torch_mock.distributed = dist_mock

    # Mock save/load
    torch_mock.save = lambda obj, path: None
    torch_mock.load = lambda path, **kwargs: _FakeTensor(2, 12, 16, 8, 64)

    sys.modules["torch"] = torch_mock

    # Mock xxhash
    xxhash_mock = types.ModuleType("xxhash")

    class _FakeXXH64:
        def __init__(self, seed=0):
            self._val = seed & 0xFFFFFFFFFFFFFFFF

        def update(self, data):
            if isinstance(data, bytes):
                self._val = (hash(data) & 0xFFFFFFFFFFFFFFFF)
            else:
                self._val = (self._val + 1) & 0xFFFFFFFFFFFFFFFF
            return self

        def intdigest(self):
            return self._val

    xxhash_mock.xxh64 = _FakeXXH64
    sys.modules["xxhash"] = xxhash_mock

    # Mock transformers
    transformers_mock = types.ModuleType("transformers")

    class _FakeHFConfig:
        def __init__(self, **kwargs):
            self.hidden_size = kwargs.get("hidden_size", 512)
            self.num_attention_heads = kwargs.get("num_attention_heads", 8)
            self.num_key_value_heads = kwargs.get("num_key_value_heads", 8)
            self.num_hidden_layers = kwargs.get("num_hidden_layers", 12)
            self.vocab_size = kwargs.get("vocab_size", 32000)
            self.head_dim = kwargs.get("head_dim", 64)
            self.dtype = "float16"

    class _FakeAutoConfig:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            return _FakeHFConfig()

    class _FakeTokenizer:
        def __init__(self):
            self.pad_token_id = 0
            self.eos_token_id = 2

        def encode(self, text, **kwargs):
            return [1, 2, 3, 4, 5]

        def decode(self, ids, **kwargs):
            return "hello"

        def apply_chat_template(self, messages, **kwargs):
            return "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"

    class _FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            return _FakeTokenizer()

    transformers_mock.AutoConfig = _FakeAutoConfig
    transformers_mock.AutoTokenizer = _FakeAutoTokenizer
    sys.modules["transformers"] = transformers_mock

    # Mock numpy
    numpy_mock = types.ModuleType("numpy")

    class _FakeNumpyArray:
        def __init__(self, data):
            self._data = data

        def tobytes(self):
            return bytes(str(self._data), "utf-8")

    numpy_mock.array = lambda data: _FakeNumpyArray(data)
    sys.modules["numpy"] = numpy_mock

    # 创建空的包模块
    for pkg in ["nanovllm", "nanovllm.engine", "nanovllm.layers",
                "nanovllm.models", "nanovllm.utils"]:
        if pkg not in sys.modules:
            mod = types.ModuleType(pkg)
            mod.__path__ = []
            sys.modules[pkg] = mod


def _import_module(name, path):
    """从文件路径导入模块"""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# 设置 mock 环境
_setup_mock()

# 导入需要的模块
base_dir = os.path.dirname(os.path.abspath(__file__))

# 导入 sampling_params
sp_mod = _import_module(
    "nanovllm.sampling_params",
    os.path.join(base_dir, "nanovllm", "sampling_params.py")
)
SamplingParams = sp_mod.SamplingParams

# 导入 sequence
seq_mod = _import_module(
    "nanovllm.engine.sequence",
    os.path.join(base_dir, "nanovllm", "engine", "sequence.py")
)
Sequence = seq_mod.Sequence
SequenceStatus = seq_mod.SequenceStatus

# 导入 multi_level_block_manager
bm_mod = _import_module(
    "nanovllm.engine.multi_level_block_manager",
    os.path.join(base_dir, "nanovllm", "engine", "multi_level_block_manager.py")
)
MultiLevelBlockManager = bm_mod.MultiLevelBlockManager
BlockLocation = bm_mod.BlockLocation


class KVCacheDemo:
    """三级 KV Cache 演示器"""

    def __init__(self, gpu_blocks=8, cpu_blocks=16, disk_blocks=32, block_size=16):
        self.gpu_blocks = gpu_blocks
        self.cpu_blocks = cpu_blocks
        self.disk_blocks = disk_blocks
        self.block_size = block_size

        # 创建块管理器
        self.bm = MultiLevelBlockManager(
            gpu_num_blocks=gpu_blocks,
            cpu_num_blocks=cpu_blocks,
            block_size=block_size,
            disk_num_blocks=disk_blocks,
            disk_cache_dir="./kv_disk_cache_demo",
            replacement_policy="lru",
            swap_watermark_high=0.8,
            swap_watermark_low=0.5,
            enable_prefix_caching=True,
        )

        self.seq_counter = 0
        self.sequences = {}

    def create_sequence(self, prompt_tokens, prefix=""):
        """创建一个新序列"""
        seq = Sequence(prompt_tokens, SamplingParams(temperature=1.0, max_tokens=100))
        seq.prompt_text = prefix + f"seq_{seq.seq_id}"

        self.sequences[seq.seq_id] = seq
        return seq

    def allocate_sequence(self, seq):
        """分配序列的 KV Cache 块"""
        num_cached = self.bm.can_allocate(seq)
        if num_cached == -1:
            return False
        self.bm.allocate(seq, num_cached)
        # 计算哈希（模拟 prefill 后更新前缀缓存）
        self.bm.hash_blocks(seq)
        return True

    def free_sequence(self, seq_id):
        """释放序列"""
        if seq_id not in self.sequences:
            return
        seq = self.sequences[seq_id]
        self.bm.deallocate(seq)
        del self.sequences[seq_id]

    def print_status(self, title=""):
        """打印当前状态"""
        stats = self.bm.get_stats()

        print("\n" + "=" * 70)
        if title:
            print(f"  {title}")
            print("=" * 70)

        # 打印各层级状态
        gpu_pct = stats.gpu_utilization() * 100
        cpu_pct = stats.cpu_utilization() * 100
        disk_pct = stats.disk_utilization() * 100

        print(f"  🟢 GPU  缓存: {stats.gpu_used_blocks:3d}/{self.gpu_blocks:3d} 块  [{gpu_pct:5.1f}%]")
        self._print_bar(gpu_pct, "🟢")

        print(f"  🟡 CPU  缓存: {stats.cpu_used_blocks:3d}/{self.cpu_blocks:3d} 块  [{cpu_pct:5.1f}%]")
        self._print_bar(cpu_pct, "🟡")

        print(f"  🔴 DISK 缓存: {stats.disk_used_blocks:3d}/{self.disk_blocks:3d} 块  [{disk_pct:5.1f}%]")
        self._print_bar(disk_pct, "🔴")

        # 打印换出统计
        print(f"\n  📊 换出统计:")
        print(f"     GPU → CPU:  {stats.gpu_cpu_swap_out:3d} 次")
        print(f"     CPU → DISK: {stats.cpu_disk_swap_out:3d} 次")
        print(f"     总换出:     {stats.total_swap_out:3d} 次, {stats.total_swap_out_blocks:3d} 块")

        # 打印前缀缓存
        print(f"\n  💾 前缀缓存:")
        print(f"     命中率: {stats.prefix_cache_hit_rate()*100:.1f}%")
        print(f"     命中次数: {stats.prefix_cache_hits} 次")

        # 打印活跃序列
        print(f"\n  📝 活跃序列 ({len(self.sequences)} 个):")
        for seq_id, seq in sorted(self.sequences.items()):
            gpu_count = sum(1 for loc in seq.block_locations if loc == BlockLocation.GPU)
            cpu_count = sum(1 for loc in seq.block_locations if loc == BlockLocation.CPU)
            disk_count = sum(1 for loc in seq.block_locations if loc == BlockLocation.DISK)
            total = len(seq.block_table)
            print(f"     seq_{seq_id:2d}: {total:2d} 块  "
                  f"(GPU:{gpu_count:2d} CPU:{cpu_count:2d} DISK:{disk_count:2d})  "
                  f"access={seq.access_count:2d}")

        print("=" * 70)

    def _print_bar(self, pct, color):
        """打印进度条"""
        width = 40
        filled = int(width * pct / 100)
        bar = "█" * filled + "░" * (width - filled)
        print(f"     {bar}")

    def demo_waterfall_swap_out(self):
        """演示瀑布式换出"""
        print("\n" + "#" * 70)
        print("#  演示 1: 瀑布式换出（GPU → CPU → DISK）")
        print("#  逐步分配序列，观察数据如何从 GPU 逐层下沉到 DISK")
        print("#" * 70)

        # 清空之前的状态
        for seq_id in list(self.sequences.keys()):
            self.free_sequence(seq_id)

        self.print_status("初始状态（空缓存）")
        time.sleep(1)

        # 逐步分配序列
        num_seqs = 12
        for i in range(num_seqs):
            # 创建不同长度的序列
            num_tokens = (i % 3 + 1) * self.block_size
            tokens = list(range(num_tokens))
            seq = self.create_sequence(tokens)

            success = self.allocate_sequence(seq)
            if success:
                seq.access_count = i + 1
                seq.last_access_time = i

            self.print_status(f"分配 seq_{i} ({num_tokens} tokens, {num_tokens//self.block_size} 块)")
            time.sleep(0.5)

        print("\n  ✅ 瀑布式换出演示完成！")
        print("     可以看到：新块优先分配到 GPU，冷数据逐层下沉到 CPU 和 DISK")

    def demo_swap_in(self):
        """演示逐级换入"""
        print("\n\n" + "#" * 70)
        print("#  演示 2: 逐级换入（DISK → CPU → GPU）")
        print("#  访问冷序列，观察数据如何从 DISK 逐层加载到 GPU")
        print("#" * 70)

        if len(self.sequences) < 3:
            print("  ⚠️  序列太少，先运行演示 1")
            return

        self.print_status("换入前状态（有数据在 DISK 上）")
        time.sleep(1)

        # 找到一个在 DISK 上的序列
        disk_seq = None
        for seq_id, seq in self.sequences.items():
            if any(loc == BlockLocation.DISK for loc in seq.block_locations):
                disk_seq = seq
                break

        if disk_seq is None:
            print("  ⚠️  没有 DISK 上的序列，先运行演示 1")
            return

        print(f"\n  🎯 目标序列: seq_{disk_seq.seq_id}")
        disk_blocks = sum(1 for loc in disk_seq.block_locations if loc == BlockLocation.DISK)
        cpu_blocks = sum(1 for loc in disk_seq.block_locations if loc == BlockLocation.CPU)
        gpu_blocks = sum(1 for loc in disk_seq.block_locations if loc == BlockLocation.GPU)
        print(f"     当前位置: GPU={gpu_blocks}, CPU={cpu_blocks}, DISK={disk_blocks}")
        time.sleep(1)

        # 执行换入
        print("\n  ⬆️  开始逐级换入...")
        self.bm.ensure_blocks_in_gpu(disk_seq, 0, len(disk_seq.block_table))
        disk_seq.access_count += 1
        disk_seq.last_access_time = 100  # 标记为最近访问

        self.print_status("换入后状态（数据已加载到 GPU）")

        print("\n  ✅ 逐级换入演示完成！")
        print("     可以看到：DISK 上的数据先加载到 CPU，再加载到 GPU")

    def demo_prefix_caching(self):
        """演示前缀缓存"""
        print("\n\n" + "#" * 70)
        print("#  演示 3: 前缀缓存（Prefix Caching）")
        print("#  相同前缀的序列可以复用已计算的 KV Cache")
        print("#" * 70)

        # 清空
        for seq_id in list(self.sequences.keys()):
            self.free_sequence(seq_id)

        # 创建第一个序列
        prompt1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        seq1 = self.create_sequence(prompt1, prefix="common_prefix_")
        self.allocate_sequence(seq1)
        seq1.access_count = 1

        self.print_status("第一个序列分配完成")
        time.sleep(1)

        # 创建第二个序列，有相同的前缀
        prompt2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        seq2 = self.create_sequence(prompt2, prefix="common_prefix_")

        print(f"\n  📝 创建第二个序列，前 16 个 token 与第一个相同")
        print(f"     预期: 前缀缓存命中 1 块 (16 tokens = 1 block)")
        time.sleep(1)

        self.allocate_sequence(seq2)
        seq2.access_count = 1

        self.print_status("第二个序列分配完成（前缀缓存命中）")

        stats = self.bm.get_stats()
        print(f"\n  ✅ 前缀缓存演示完成！")
        print(f"     前缀缓存命中: {stats.prefix_cache_hits} 次")
        print(f"     节省了重新计算的时间！")

    def demo_lru_eviction(self):
        """演示 LRU 淘汰策略"""
        print("\n\n" + "#" * 70)
        print("#  演示 4: LRU 淘汰策略")
        print("#  最近最少使用的块优先被换出")
        print("#" * 70)

        # 清空
        for seq_id in list(self.sequences.keys()):
            self.free_sequence(seq_id)

        self.print_status("初始状态")
        time.sleep(1)

        # 分配 5 个序列
        print("\n  📝 分配 5 个序列，每个 2 块（共 10 块，GPU=8 块）")
        print("     访问顺序: seq_0 → seq_1 → seq_2 → seq_3 → seq_4")
        print("     预期: seq_0（最久未访问）最先被换出")
        time.sleep(1)

        for i in range(5):
            tokens = list(range(i * 32, (i + 1) * 32))
            seq = self.create_sequence(tokens)
            self.allocate_sequence(seq)
            seq.access_count = 1
            seq.last_access_time = i  # seq_0 最早访问

        self.print_status("5 个序列分配完成")
        time.sleep(1)

        # 查看热度图
        heat_map = self.bm.get_block_heat_map()
        print("\n  🔥 块热度分布（热度越高越不容易被换出）:")
        for location in [BlockLocation.GPU, BlockLocation.CPU, BlockLocation.DISK]:
            blocks = [(bid, heat) for bid, heat in heat_map.items()
                     if self.bm.get_block_location(bid) == location]
            if blocks:
                blocks.sort(key=lambda x: x[1], reverse=True)
                loc_name = str(location).split('.')[-1]
                print(f"     {loc_name}: ", end="")
                for bid, heat in blocks[:6]:
                    print(f"blk_{bid}({heat:.2f}) ", end="")
                print()

        print("\n  ✅ LRU 淘汰策略演示完成！")
        print("     可以看到：最近访问的块热度更高，更不容易被换出")

    def run_full_demo(self):
        """运行完整演示"""
        print("\n" + "╔" + "═" * 68 + "╗")
        print("║" + " " * 10 + "🚀 Nano-vLLM 三级 KV Cache 演示 🚀" + " " * 15 + "║")
        print("║" + " " * 8 + "GPU HBM + CPU DRAM + SSD/NVMe 三级缓存" + " " * 12 + "║")
        print("╚" + "═" * 68 + "╝")

        time.sleep(2)

        # 演示 1: 瀑布式换出
        self.demo_waterfall_swap_out()
        time.sleep(2)

        # 演示 2: 逐级换入
        self.demo_swap_in()
        time.sleep(2)

        # 演示 3: 前缀缓存
        self.demo_prefix_caching()
        time.sleep(2)

        # 演示 4: LRU 淘汰
        self.demo_lru_eviction()
        time.sleep(2)

        # 总结
        print("\n\n" + "╔" + "═" * 68 + "╗")
        print("║" + " " * 20 + "🎉 演示完成！ 🎉" + " " * 25 + "║")
        print("╠" + "═" * 68 + "╣")
        print("║  核心特性总结:" + " " * 51 + "║")
        print("║    ✅ 瀑布式换出: GPU → CPU → DISK" + " " * 28 + "║")
        print("║    ✅ 逐级换入: DISK → CPU → GPU" + " " * 30 + "║")
        print("║    ✅ LRU 淘汰策略: 冷数据优先下沉" + " " * 28 + "║")
        print("║    ✅ 前缀缓存: 复用相同前缀的 KV Cache" + " " * 25 + "║")
        print("║    ✅ 三级容量扩展: 显存不够用内存，内存不够用磁盘" + " " * 16 + "║")
        print("╚" + "═" * 68 + "╝")


def main():
    """主函数"""
    # 创建演示器
    demo = KVCacheDemo(
        gpu_blocks=8,
        cpu_blocks=16,
        disk_blocks=32,
        block_size=16,
    )

    # 运行完整演示
    demo.run_full_demo()

    # 清理磁盘缓存
    print("\n  🧹 清理演示用的磁盘缓存...")
    if os.path.exists("./kv_disk_cache_demo"):
        import shutil
        shutil.rmtree("./kv_disk_cache_demo")
    print("  ✅ 清理完成")


if __name__ == "__main__":
    main()
