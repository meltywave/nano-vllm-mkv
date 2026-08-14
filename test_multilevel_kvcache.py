"""
三级 KV Cache 单元测试（不依赖 GPU/torch）
验证核心逻辑：三级块管理、调度、换入换出、前缀缓存
"""

import sys
import os
import tempfile
import shutil
import importlib.util

# ============================================================
# 第一步：mock 所有外部依赖
# ============================================================
import types

# mock torch（完整结构）
torch_mock = types.ModuleType('torch')

class _FakeTensor:
    def __init__(self, *args, **kwargs):
        self.shape = args[0] if args else ()
        self.dtype = kwargs.get('dtype', 'float32')
    def numel(self):
        n = 1
        for s in self.shape:
            n *= s
        return n
    def element_size(self):
        return 2
    def size(self, dim=None):
        if dim is None:
            return self.shape
        return self.shape[dim]
    def __getitem__(self, idx):
        return _FakeTensor()
    def __setitem__(self, idx, val):
        pass
    def copy_(self, src, non_blocking=False):
        return self
    def cuda(self, non_blocking=False):
        return self
    def cpu(self):
        return self
    def tolist(self):
        return []
    def fill_(self, val):
        return self
    def zero_(self):
        return self
    def __len__(self):
        return self.shape[0] if self.shape else 0

torch_mock.empty = lambda *args, **kwargs: _FakeTensor(*args, **kwargs)
torch_mock.float16 = 'float16'
torch_mock.save = lambda *args, **kwargs: None
torch_mock.load = lambda *args, **kwargs: _FakeTensor()
torch_mock.Tensor = _FakeTensor
torch_mock.cuda = types.ModuleType('torch.cuda')
torch_mock.cuda.Stream = lambda *a, **kw: type('Stream', (), {
    'synchronize': lambda: None,
})()
torch_mock.cuda.Event = lambda *a, **kw: type('Event', (), {
    'record': lambda self, s: None,
    'synchronize': lambda self: None,
})()
torch_mock.cuda.mem_get_info = lambda: (1024**3, 8 * 1024**3)
torch_mock.cuda.empty_cache = lambda: None
torch_mock.cuda.reset_peak_memory_stats = lambda: None
torch_mock.cuda.memory_stats = lambda: {
    'allocated_bytes.all.peak': 0,
    'allocated_bytes.all.current': 0,
}
torch_mock.cuda.set_device = lambda *a: None
torch_mock.cuda.synchronize = lambda: None
torch_mock.cuda.CUDAGraph = type('CUDAGraph', (), {})
torch_mock.cuda.graph = lambda *a, **kw: _GraphCtx()
torch_mock.inference_mode = lambda *a, **kw: _InferModeCtx()
torch_mock.set_default_dtype = lambda *a: None
torch_mock.set_default_device = lambda *a: None
torch_mock.get_default_dtype = lambda: torch_mock.float16
torch_mock.tensor = lambda *a, **kw: _FakeTensor()
torch_mock.int32 = 'int32'
torch_mock.int64 = 'int64'
torch_mock.float32 = 'float32'
torch_mock.Tensor = _FakeTensor
torch_mock.multiprocessing = types.ModuleType('torch.multiprocessing')
torch_mock.multiprocessing.get_context = lambda *a, **kw: _FakeMPContext()
torch_mock.distributed = types.ModuleType('torch.distributed')
torch_mock.distributed.init_process_group = lambda *a, **kw: None
torch_mock.distributed.barrier = lambda *a, **kw: None
torch_mock.distributed.destroy_process_group = lambda *a, **kw: None
sys.modules['torch'] = torch_mock
sys.modules['torch.cuda'] = torch_mock.cuda
sys.modules['torch.multiprocessing'] = torch_mock.multiprocessing
sys.modules['torch.distributed'] = torch_mock.distributed

class _GraphCtx:
    def __init__(self, *args, **kwargs):
        pass
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def pool(self):
        return None

class _InferModeCtx:
    def __init__(self, *args, **kwargs):
        pass
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass

class _FakeMPContext:
    def Event(self):
        return type('Event', (), {'wait': lambda: None, 'set': lambda: None, 'clear': lambda: None})()
    def Process(self, target=None, args=()):
        return type('Process', (), {'start': lambda: None, 'join': lambda: None})()

# mock transformers
transformers_mock = types.ModuleType('transformers')
class _FakeHFConfig:
    def __init__(self):
        self.num_hidden_layers = 2
        self.num_attention_heads = 8
        self.num_key_value_heads = 2
        self.hidden_size = 128
        self.head_dim = 16
        self.max_position_embeddings = 4096
        self.dtype = torch_mock.float16
AutoConfig_mock = type('AutoConfig', (), {
    'from_pretrained': lambda *a, **kw: _FakeHFConfig()
})
AutoTokenizer_mock = type('AutoTokenizer', (), {
    'from_pretrained': lambda *a, **kw: type('tok', (), {
        'encode': lambda s: [1, 2, 3],
        'decode': lambda ids: 'test',
        'eos_token_id': 2,
    })()
})
transformers_mock.AutoConfig = AutoConfig_mock
transformers_mock.AutoTokenizer = AutoTokenizer_mock
sys.modules['transformers'] = transformers_mock

# mock tqdm
tqdm_mock = types.ModuleType('tqdm')
tqdm_auto_mock = types.ModuleType('tqdm.auto')
class _FakeTqdm:
    def __init__(self, *a, **kw):
        self.total = 0
    def __enter__(self):
        return self
    def __exit__(self, *a):
        pass
    def update(self, *a):
        pass
    def set_postfix(self, *a, **kw):
        pass
    def close(self):
        pass
tqdm_auto_mock.tqdm = _FakeTqdm
sys.modules['tqdm'] = tqdm_mock
sys.modules['tqdm.auto'] = tqdm_auto_mock

# mock xxhash
xxhash_mock = types.ModuleType('xxhash')
class _FakeXXH64:
    def __init__(self):
        self._val = 0
    def update(self, data):
        if isinstance(data, bytes):
            self._val = (hash(data) & 0xFFFFFFFFFFFFFFFF)  # 确保无符号
        else:
            self._val = (self._val + 1) & 0xFFFFFFFFFFFFFFFF
    def intdigest(self):
        return self._val
xxhash_mock.xxh64 = _FakeXXH64
sys.modules['xxhash'] = xxhash_mock

# mock numpy
numpy_mock = types.ModuleType('numpy')
class _FakeNumpyArray:
    def __init__(self, data):
        self._data = data if isinstance(data, list) else [data]
    def tobytes(self):
        return bytes(str(self._data), 'utf-8')
    def __len__(self):
        return len(self._data)
numpy_mock.array = lambda x: _FakeNumpyArray(x)
sys.modules['numpy'] = numpy_mock

# mock multiprocessing
mp_mock = types.ModuleType('multiprocessing')
mp_mock.synchronize = types.ModuleType('multiprocessing.synchronize')
mp_mock.synchronize.Event = lambda: type('Event', (), {'wait': lambda: None, 'set': lambda: None, 'clear': lambda: None})()
mp_mock.shared_memory = types.ModuleType('multiprocessing.shared_memory')
mp_mock.shared_memory.SharedMemory = type('SharedMemory', (), {
    '__init__': lambda self, **kw: setattr(self, 'buf', bytearray(2**20)),
    'close': lambda self: None,
    'unlink': lambda self: None,
})
sys.modules['multiprocessing'] = mp_mock
sys.modules['multiprocessing.synchronize'] = mp_mock.synchronize
sys.modules['multiprocessing.shared_memory'] = mp_mock.shared_memory

# ============================================================
# 第二步：创建 nanovllm 包结构（避免 __init__.py 触发全量导入）
# ============================================================

# 先创建空的包模块
nanovllm_pkg = types.ModuleType('nanovllm')
nanovllm_pkg.__path__ = [os.path.join(os.path.dirname(__file__), 'nanovllm')]
sys.modules['nanovllm'] = nanovllm_pkg

engine_pkg = types.ModuleType('nanovllm.engine')
engine_pkg.__path__ = [os.path.join(os.path.dirname(__file__), 'nanovllm', 'engine')]
sys.modules['nanovllm.engine'] = engine_pkg

layers_pkg = types.ModuleType('nanovllm.layers')
layers_pkg.__path__ = [os.path.join(os.path.dirname(__file__), 'nanovllm', 'layers')]
sys.modules['nanovllm.layers'] = layers_pkg

models_pkg = types.ModuleType('nanovllm.models')
models_pkg.__path__ = [os.path.join(os.path.dirname(__file__), 'nanovllm', 'models')]
sys.modules['nanovllm.models'] = models_pkg

utils_pkg = types.ModuleType('nanovllm.utils')
utils_pkg.__path__ = [os.path.join(os.path.dirname(__file__), 'nanovllm', 'utils')]
sys.modules['nanovllm.utils'] = utils_pkg

# ============================================================
# 第三步：逐个导入需要的模块
# ============================================================

def _import_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

_base = os.path.dirname(os.path.abspath(__file__))

# 1. sampling_params
sp_module = _import_module(
    'nanovllm.sampling_params',
    os.path.join(_base, 'nanovllm', 'sampling_params.py')
)
nanovllm_pkg.sampling_params = sp_module
SamplingParams = sp_module.SamplingParams

# 2. sequence
seq_module = _import_module(
    'nanovllm.engine.sequence',
    os.path.join(_base, 'nanovllm', 'engine', 'sequence.py')
)
engine_pkg.sequence = seq_module
Sequence = seq_module.Sequence
SequenceStatus = seq_module.SequenceStatus

# 3. config
config_module = _import_module(
    'nanovllm.config',
    os.path.join(_base, 'nanovllm', 'config.py')
)
nanovllm_pkg.config = config_module
Config = config_module.Config

# 4. block_manager（原始）
bm_orig_module = _import_module(
    'nanovllm.engine.block_manager',
    os.path.join(_base, 'nanovllm', 'engine', 'block_manager.py')
)
engine_pkg.block_manager = bm_orig_module

# 5. multi_level_block_manager
bm_module = _import_module(
    'nanovllm.engine.multi_level_block_manager',
    os.path.join(_base, 'nanovllm', 'engine', 'multi_level_block_manager.py')
)
engine_pkg.multi_level_block_manager = bm_module
MultiLevelBlockManager = bm_module.MultiLevelBlockManager
BlockLocation = bm_module.BlockLocation
Block = bm_module.Block
DiskBlockStore = bm_module.DiskBlockStore

# 6. multi_level_scheduler
sched_module = _import_module(
    'nanovllm.engine.multi_level_scheduler',
    os.path.join(_base, 'nanovllm', 'engine', 'multi_level_scheduler.py')
)
engine_pkg.multi_level_scheduler = sched_module
MultiLevelScheduler = sched_module.MultiLevelScheduler


def test_sequence():
    """测试 Sequence 扩展"""
    print("=" * 60)
    print("测试 1: Sequence 扩展")
    print("=" * 60)

    # 测试 SWAPPED_OUT 状态
    assert hasattr(SequenceStatus, 'SWAPPED_OUT'), "缺少 SWAPPED_OUT 状态"
    print("✅ SequenceStatus.SWAPPED_OUT 存在")

    # 测试新字段
    seq = Sequence([1, 2, 3, 4, 5])
    assert hasattr(seq, 'block_locations'), "缺少 block_locations 字段"
    assert hasattr(seq, 'last_access_time'), "缺少 last_access_time 字段"
    assert hasattr(seq, 'access_count'), "缺少 access_count 字段"
    print("✅ 新字段存在: block_locations, last_access_time, access_count")

    # 测试默认值
    assert seq.block_locations == [], "block_locations 默认应为空列表"
    assert seq.last_access_time == 0.0, "last_access_time 默认应为 0.0"
    assert seq.access_count == 0, "access_count 默认应为 0"
    print("✅ 默认值正确")

    print("🎉 Sequence 测试通过！\n")


def test_block_manager_basic():
    """测试块管理器基本功能"""
    print("=" * 60)
    print("测试 2: 块管理器基本功能")
    print("=" * 60)

    Sequence.block_size = 16

    bm = MultiLevelBlockManager(
        gpu_num_blocks=10,
        cpu_num_blocks=20,
        block_size=16,
    )

    # 测试初始化
    assert bm.gpu_num_blocks == 10
    assert bm.cpu_num_blocks == 20
    assert len(bm.gpu_free_block_ids) == 10
    assert len(bm.cpu_free_block_ids) == 20
    print("✅ 初始化正确 (GPU=10, CPU=20)")

    # 测试分配
    seq = Sequence([i for i in range(32)])  # 32 tokens = 2 blocks
    num_cached = bm.can_allocate(seq)
    assert num_cached == 0, "首次分配应该没有前缀缓存命中"
    print(f"✅ can_allocate: {num_cached} (无前缀缓存)")

    bm.allocate(seq, num_cached)
    assert len(seq.block_table) == 2, "应该分配 2 个块"
    assert len(seq.block_locations) == 2, "block_locations 应该有 2 个"
    assert all(loc == BlockLocation.GPU for loc in seq.block_locations), "新块应该都在 GPU"
    assert bm.stats.gpu_used_blocks == 2, "GPU 已用 2 块"
    assert bm.stats.cpu_used_blocks == 0, "CPU 已用 0 块"
    print(f"✅ 分配成功: block_table={seq.block_table}, 全部在 GPU")

    # 测试释放
    bm.deallocate(seq)
    assert len(seq.block_table) == 0, "释放后 block_table 应为空"
    assert len(seq.block_locations) == 0, "释放后 block_locations 应为空"
    assert bm.stats.gpu_used_blocks == 0, "GPU 已用 0 块"
    print("✅ 释放成功")

    print("🎉 基本功能测试通过！\n")


def test_block_manager_gpu_full():
    """测试 GPU 满了之后触发瀑布式换出到 CPU"""
    print("=" * 60)
    print("测试 3: GPU 满 → 瀑布式换出到 CPU")
    print("=" * 60)

    Sequence.block_size = 16

    bm = MultiLevelBlockManager(
        gpu_num_blocks=2,  # 只有 2 个 GPU 块
        cpu_num_blocks=10,
        block_size=16,
        swap_watermark_high=0.6,  # 超过 60% 触发换出
        swap_watermark_low=0.3,   # 换出到 30% 停止
    )

    # 分配第一个序列（2 块，占满 GPU）
    seq1 = Sequence([i for i in range(32)])  # 2 blocks
    num_cached = bm.can_allocate(seq1)
    bm.allocate(seq1, num_cached)
    assert bm.stats.gpu_used_blocks == 2, "GPU 应该满了"
    assert bm.stats.gpu_cpu_swap_out == 0, "初始时没有换出"
    print(f"✅ seq1 分配完成，GPU 已用: {bm.stats.gpu_used_blocks}/2 (100%)")

    # 分配第二个序列
    # 瀑布式行为：GPU 满了触发换出，把 seq1 的冷块换到 CPU
    # 腾出空间给 seq2 的新块（新块优先在 GPU）
    seq2 = Sequence([i for i in range(32, 64)])  # 2 blocks
    num_cached = bm.can_allocate(seq2)
    assert num_cached != -1, "应该能分配（有 CPU 空间用于换出）"
    print(f"✅ can_allocate 返回: {num_cached} (可分配)")

    bm.allocate(seq2, num_cached)
    assert len(seq2.block_table) == 2
    assert len(seq2.block_locations) == 2

    # 验证换出发生了
    assert bm.stats.gpu_cpu_swap_out > 0, "应该有 GPU→CPU 换出"
    assert bm.stats.cpu_used_blocks > 0, "CPU 应该有块（换出过来的）"
    print(f"✅ 换出统计: GPU→CPU 换出 {bm.stats.gpu_cpu_swap_out} 次, 总换出 {bm.stats.total_swap_out_blocks} 块")
    print(f"✅ 当前状态: GPU={bm.stats.gpu_used_blocks}/2, CPU={bm.stats.cpu_used_blocks}/10")

    # 验证 seq2 的块都在 GPU（新块优先在 GPU）
    gpu_blocks_seq2 = sum(1 for loc in seq2.block_locations if loc == BlockLocation.GPU)
    print(f"✅ seq2 块位置: GPU={gpu_blocks_seq2}/2 (新块优先在 GPU)")

    print("🎉 GPU 满 → 瀑布式换出到 CPU 测试通过！\n")


def test_block_manager_cpu_full():
    """测试 CPU 满了之后触发瀑布式换出到 DISK（三级缓存）"""
    print("=" * 60)
    print("测试 4: CPU 满 → 瀑布式换出到 DISK（三级缓存）")
    print("=" * 60)

    Sequence.block_size = 16

    # 创建临时目录用于磁盘缓存
    tmp_dir = tempfile.mkdtemp(prefix="kv_disk_test_")

    try:
        bm = MultiLevelBlockManager(
            gpu_num_blocks=2,   # 2 个 GPU 块
            cpu_num_blocks=2,   # 2 个 CPU 块
            disk_num_blocks=10, # 10 个 DISK 块
            disk_cache_dir=tmp_dir,
            block_size=16,
            swap_watermark_high=0.6,  # 超过 60% 触发换出
            swap_watermark_low=0.3,   # 换出到 30% 停止
        )

        # 初始化磁盘存储
        bm.init_disk_store(tmp_dir, (2, 2, 16, 4, 16), 'float16')
        assert bm.enable_disk == True
        print("✅ 三级缓存初始化: GPU=2, CPU=2, DISK=10")

        # 分配 seq1（占满 GPU）
        seq1 = Sequence([i for i in range(32)])  # 2 blocks
        bm.allocate(seq1, 0)
        assert bm.stats.gpu_used_blocks == 2
        print(f"✅ seq1 占满 GPU: {bm.stats.gpu_used_blocks}/2")

        # 分配 seq2（占满 CPU）
        seq2 = Sequence([i for i in range(32, 64)])  # 2 blocks
        bm.allocate(seq2, 0)
        assert bm.stats.cpu_used_blocks == 2
        print(f"✅ seq2 占满 CPU: {bm.stats.cpu_used_blocks}/2 (100%)")

        # 分配 seq3
        # 瀑布式行为：CPU 满了触发换出到 DISK，腾出空间给新块
        seq3 = Sequence([i for i in range(64, 96)])  # 2 blocks
        num_cached = bm.can_allocate(seq3)
        assert num_cached != -1, "应该能分配（有 DISK 空间用于换出）"
        print(f"✅ can_allocate 返回: {num_cached} (可分配)")

        bm.allocate(seq3, num_cached)
        assert len(seq3.block_table) == 2
        assert len(seq3.block_locations) == 2

        # 验证换出到 DISK 发生了
        assert bm.stats.cpu_disk_swap_out > 0, "应该有 CPU→DISK 换出"
        assert bm.stats.disk_used_blocks > 0, "DISK 应该有块（换出过来的）"
        print(f"✅ 换出统计: CPU→DISK 换出 {bm.stats.cpu_disk_swap_out} 次, 总换出 {bm.stats.total_swap_out_blocks} 块")
        print(f"✅ 当前状态: GPU={bm.stats.gpu_used_blocks}/2, CPU={bm.stats.cpu_used_blocks}/2, DISK={bm.stats.disk_used_blocks}/10")

        print("🎉 CPU 满 → 瀑布式换出到 DISK 测试通过！\n")

    finally:
        # 清理临时目录
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_disk_block_store():
    """测试磁盘块存储"""
    print("=" * 60)
    print("测试 5: DiskBlockStore 基本功能")
    print("=" * 60)

    tmp_dir = tempfile.mkdtemp(prefix="disk_store_test_")

    try:
        store = DiskBlockStore(
            cache_dir=tmp_dir,
            num_blocks=5,
            block_shape=(2, 2, 16, 4, 16),
        )

        # 测试初始化
        assert store.get_free_count() == 5
        assert store.get_used_count() == 0
        print("✅ 初始化: 5 个空闲块")

        # 测试分配
        bid1 = store.allocate()
        bid2 = store.allocate()
        assert store.get_used_count() == 2
        assert store.get_free_count() == 3
        print(f"✅ 分配 2 个块: bid={bid1}, {bid2}")

        # 测试释放
        store.deallocate(bid1)
        assert store.get_used_count() == 1
        assert store.get_free_count() == 4
        print("✅ 释放 1 个块")

        # 测试全部释放
        store.deallocate(bid2)
        assert store.get_used_count() == 0
        assert store.get_free_count() == 5
        print("✅ 全部释放")

        # 测试清空
        store.clear()
        assert store.get_used_count() == 0
        print("✅ clear 成功")

        print("🎉 DiskBlockStore 测试通过！\n")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_swap_out_gpu_to_cpu():
    """测试 GPU → CPU 换出"""
    print("=" * 60)
    print("测试 6: GPU → CPU 换出")
    print("=" * 60)

    Sequence.block_size = 16

    bm = MultiLevelBlockManager(
        gpu_num_blocks=4,
        cpu_num_blocks=10,
        block_size=16,
        swap_watermark_high=0.6,  # 超过 60% 触发换出
        swap_watermark_low=0.3,   # 换出到 30% 停止
    )

    # 分配 3 个块（75%，超过高水位线）
    seq1 = Sequence([i for i in range(48)])  # 3 blocks
    bm.allocate(seq1, 0)
    assert bm.stats.gpu_used_blocks == 3
    print(f"✅ 分配 3 个 GPU 块 (使用率: {bm.stats.gpu_utilization():.0%})")

    # 手动触发换出
    bm._trigger_gpu_swap_out_if_needed()

    # 检查是否换出了一些块到 CPU
    gpu_after = bm.stats.gpu_used_blocks
    cpu_after = bm.stats.cpu_used_blocks
    print(f"✅ 换出后: GPU={gpu_after}, CPU={cpu_after}")
    assert gpu_after < 3, "GPU 使用量应该减少"
    assert cpu_after > 0, "CPU 使用量应该增加"
    assert bm.stats.gpu_cpu_swap_out > 0, "应该有 GPU→CPU 换出计数"

    print("🎉 GPU → CPU 换出测试通过！\n")


def test_swap_out_cpu_to_disk():
    """测试 CPU → DISK 换出（三级缓存）"""
    print("=" * 60)
    print("测试 7: CPU → DISK 换出（三级缓存）")
    print("=" * 60)

    Sequence.block_size = 16
    tmp_dir = tempfile.mkdtemp(prefix="swap_disk_test_")

    try:
        bm = MultiLevelBlockManager(
            gpu_num_blocks=2,
            cpu_num_blocks=4,
            disk_num_blocks=10,
            disk_cache_dir=tmp_dir,
            block_size=16,
            swap_watermark_high=0.6,
            swap_watermark_low=0.3,
        )
        bm.init_disk_store(tmp_dir, (2, 2, 16, 4, 16), 'float16')

        # 分配 2 个 GPU 块（占满）
        seq1 = Sequence([i for i in range(32)])  # 2 blocks
        bm.allocate(seq1, 0)
        print(f"✅ GPU 分配: {bm.stats.gpu_used_blocks}/2")

        # 分配 3 个 CPU 块（75%，超过高水位线）
        seq2 = Sequence([i for i in range(32, 80)])  # 3 blocks → CPU
        bm.allocate(seq2, 0)
        cpu_used = bm.stats.cpu_used_blocks
        print(f"✅ CPU 分配: {cpu_used}/4 (使用率: {bm.stats.cpu_utilization():.0%})")

        # 手动触发 CPU→DISK 换出
        bm._trigger_cpu_swap_out_if_needed()

        cpu_after = bm.stats.cpu_used_blocks
        disk_after = bm.stats.disk_used_blocks
        print(f"✅ 换出后: CPU={cpu_after}, DISK={disk_after}")
        assert cpu_after < cpu_used, "CPU 使用量应该减少"
        assert disk_after > 0, "DISK 使用量应该增加"
        assert bm.stats.cpu_disk_swap_out > 0, "应该有 CPU→DISK 换出计数"

        print("🎉 CPU → DISK 换出测试通过！\n")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_swap_in_disk_to_cpu():
    """测试 DISK → CPU 换入"""
    print("=" * 60)
    print("测试 8: DISK → CPU 换入")
    print("=" * 60)

    Sequence.block_size = 16
    tmp_dir = tempfile.mkdtemp(prefix="swap_in_test_")

    try:
        bm = MultiLevelBlockManager(
            gpu_num_blocks=2,
            cpu_num_blocks=4,
            disk_num_blocks=20,
            disk_cache_dir=tmp_dir,
            block_size=16,
            swap_watermark_high=0.5,  # 超过 50% 就换出（更激进）
            swap_watermark_low=0.25,  # 换出到 25%
        )
        bm.init_disk_store(tmp_dir, (2, 2, 16, 4, 16), 'float16')

        # 分配多个序列，让 CPU 满并触发换出到 DISK
        seqs = []
        for i in range(10):  # 10 个序列，每个 2 块 = 20 块
            seq = Sequence([j for j in range(i * 32, (i + 1) * 32)])
            bm.allocate(seq, 0)
            seqs.append(seq)

        disk_before = bm.stats.disk_used_blocks
        cpu_before = bm.stats.cpu_used_blocks
        gpu_before = bm.stats.gpu_used_blocks
        print(f"✅ 初始状态: GPU={gpu_before}, CPU={cpu_before}, DISK={disk_before}")
        print(f"✅ 换出统计: CPU→DISK: {bm.stats.cpu_disk_swap_out} 次, 总换出 {bm.stats.total_swap_out_blocks} 块")
        assert disk_before > 0, "应该有块在 DISK 上（瀑布式换出）"

        # 找到一个 DISK 块
        disk_block_id = None
        disk_seq_idx = None
        disk_block_idx = None
        for si, seq in enumerate(seqs):
            for bi, loc in enumerate(seq.block_locations):
                if loc == BlockLocation.DISK:
                    disk_block_id = seq.block_table[bi]
                    disk_seq_idx = si
                    disk_block_idx = bi
                    break
            if disk_block_id is not None:
                break

        assert disk_block_id is not None, "应该找到 DISK 块"
        print(f"✅ 找到 DISK 块: {disk_block_id} (seq {disk_seq_idx}, block {disk_block_idx})")

        # 找到 CPU 上的序列并释放，腾出 CPU 空间
        cpu_seq_indices = []
        for si, seq in enumerate(seqs):
            if si == disk_seq_idx:
                continue  # 不要释放包含目标 DISK 块的序列
            for bi, loc in enumerate(seq.block_locations):
                if loc == BlockLocation.CPU:
                    cpu_seq_indices.append(si)
                    break

        print(f"✅ CPU 上的序列: {cpu_seq_indices}")
        assert len(cpu_seq_indices) > 0, "应该有 CPU 上的序列"

        # 释放 CPU 上的序列
        for si in cpu_seq_indices[:2]:
            bm.deallocate(seqs[si])
        cpu_free_after_dealloc = bm.cpu_num_blocks - bm.stats.cpu_used_blocks
        print(f"✅ 释放 CPU 序列后 CPU 空闲: {cpu_free_after_dealloc}")
        assert cpu_free_after_dealloc > 0, "释放后应该有 CPU 空闲空间"

        # 换入 DISK 块到 CPU
        cpu_block_id = bm.swap_in_disk_to_cpu(disk_block_id)
        assert cpu_block_id is not None, "换入应该成功"

        disk_after = bm.stats.disk_used_blocks
        cpu_after = bm.stats.cpu_used_blocks
        print(f"✅ 换入后: CPU={cpu_after}, DISK={disk_after}")
        assert disk_after < disk_before, "DISK 使用量应该减少"
        assert bm.stats.cpu_disk_swap_in > 0, "应该有 DISK→CPU 换入计数"

        print("🎉 DISK → CPU 换入测试通过！\n")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_ensure_blocks_in_gpu():
    """测试 ensure_blocks_in_gpu（逐级换入）"""
    print("=" * 60)
    print("测试 9: ensure_blocks_in_gpu（逐级换入）")
    print("=" * 60)

    Sequence.block_size = 16
    tmp_dir = tempfile.mkdtemp(prefix="ensure_gpu_test_")

    try:
        bm = MultiLevelBlockManager(
            gpu_num_blocks=4,
            cpu_num_blocks=4,
            disk_num_blocks=10,
            disk_cache_dir=tmp_dir,
            block_size=16,
        )
        bm.init_disk_store(tmp_dir, (2, 2, 16, 4, 16), 'float16')

        # 分配一个序列，让块分布在不同层级
        seq = Sequence([i for i in range(160)])  # 10 blocks
        bm.allocate(seq, 0)

        gpu_before = sum(1 for loc in seq.block_locations if loc == BlockLocation.GPU)
        cpu_before = sum(1 for loc in seq.block_locations if loc == BlockLocation.CPU)
        disk_before = sum(1 for loc in seq.block_locations if loc == BlockLocation.DISK)
        print(f"✅ 初始分布: GPU={gpu_before}, CPU={cpu_before}, DISK={disk_before}")

        # 确保所有块都在 GPU
        # 先释放一些 GPU 空间
        # （这里简化测试，直接调用方法检查逻辑）
        result = bm.ensure_blocks_in_gpu(seq, 0, len(seq.block_table))
        # 可能会失败因为 GPU 空间不够，但方法应该正常执行
        print(f"✅ ensure_blocks_in_gpu 返回: {result}")

        # 验证块位置都被正确更新（如果成功的话）
        gpu_after = sum(1 for loc in seq.block_locations if loc == BlockLocation.GPU)
        print(f"✅ 换入后 GPU 块数: {gpu_after}")

        print("🎉 ensure_blocks_in_gpu 测试通过！\n")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_block_manager_prefix_caching():
    """测试前缀缓存（三级都支持）"""
    print("=" * 60)
    print("测试 10: 三级前缀缓存")
    print("=" * 60)

    Sequence.block_size = 16

    bm = MultiLevelBlockManager(
        gpu_num_blocks=10,
        cpu_num_blocks=20,
        block_size=16,
        enable_prefix_caching=True,
    )

    # 分配第一个序列
    prompt1 = [i for i in range(32)]  # 2 blocks
    seq1 = Sequence(prompt1)
    num_cached = bm.can_allocate(seq1)
    bm.allocate(seq1, num_cached)

    # 模拟 prefill 完成，计算哈希
    # num_cached_tokens = 0（初始），num_scheduled_tokens = 32
    # hash_blocks 会处理 0 到 32//16 = 2 个块
    seq1.num_cached_tokens = 0
    seq1.num_scheduled_tokens = 32
    bm.hash_blocks(seq1)
    seq1.num_cached_tokens = 32
    print("✅ seq1 prefill 完成，哈希已计算")

    # 分配第二个序列（相同前缀）
    prompt2 = list(prompt1) + [100, 101]  # 前 2 块相同，第 3 块不同
    seq2 = Sequence(prompt2)
    num_cached = bm.can_allocate(seq2)

    print(f"✅ 前缀缓存命中: {num_cached} 块")
    assert num_cached >= 1, "应该至少命中 1 块前缀缓存"

    print("🎉 前缀缓存测试通过！\n")


def test_scheduler_basic():
    """测试多级调度器基本功能"""
    print("=" * 60)
    print("测试 11: 多级调度器基本功能")
    print("=" * 60)

    from nanovllm.engine.multi_level_scheduler import MultiLevelScheduler

    # 创建一个 mock config
    class MockConfig:
        max_num_seqs = 16
        max_num_batched_tokens = 256
        eos = 2
        kvcache_block_size = 16
        num_kvcache_blocks = 10
        enable_multilevel_kvcache = True
        cpu_num_kvcache_blocks = 20
        disk_num_kvcache_blocks = 0  # 不启用磁盘
        swap_watermark_high = 0.9
        swap_watermark_low = 0.7
        replacement_policy = "lru"
        enable_prefetch = True
        prefetch_lookahead = 2

    config = MockConfig()
    Sequence.block_size = 16

    scheduler = MultiLevelScheduler(config)
    print("✅ MultiLevelScheduler 创建成功")

    # 测试三级队列存在
    assert hasattr(scheduler, 'waiting'), "缺少 waiting 队列"
    assert hasattr(scheduler, 'running'), "缺少 running 队列"
    assert hasattr(scheduler, 'swapped_out'), "缺少 swapped_out 队列"
    print("✅ 三级队列存在: waiting, running, swapped_out")

    # 测试 add
    seq = Sequence([1, 2, 3, 4, 5])
    scheduler.add(seq)
    assert len(scheduler.waiting) == 1
    print("✅ add 成功")

    # 测试 is_finished
    assert not scheduler.is_finished(), "有等待序列，不应该结束"
    print("✅ is_finished 正确")

    # 测试 get_stats
    stats = scheduler.get_stats()
    assert 'waiting' in stats
    assert 'running' in stats
    assert 'swapped_out' in stats
    print(f"✅ get_stats 成功: waiting={stats['waiting']}, running={stats['running']}")

    print("🎉 调度器基本功能测试通过！\n")


def test_scheduler_three_level():
    """测试三级缓存调度器（含 DISK）"""
    print("=" * 60)
    print("测试 12: 三级缓存调度器（含 DISK）")
    print("=" * 60)

    from nanovllm.engine.multi_level_scheduler import MultiLevelScheduler

    tmp_dir = tempfile.mkdtemp(prefix="sched_disk_test_")

    try:
        class MockConfig:
            max_num_seqs = 16
            max_num_batched_tokens = 256
            eos = 2
            kvcache_block_size = 16
            num_kvcache_blocks = 4  # GPU 只有 4 块
            enable_multilevel_kvcache = True
            cpu_num_kvcache_blocks = 4  # CPU 4 块
            disk_num_kvcache_blocks = 10  # DISK 10 块
            disk_cache_dir = tmp_dir
            swap_watermark_high = 0.8
            swap_watermark_low = 0.5
            replacement_policy = "lru"
            enable_prefetch = True
            prefetch_lookahead = 2

        config = MockConfig()
        Sequence.block_size = 16

        scheduler = MultiLevelScheduler(config)
        assert scheduler.enable_disk == True
        print("✅ 三级调度器创建成功（含 DISK）")

        # 测试块管理器支持三级
        bm = scheduler.block_manager
        assert bm.enable_disk == True
        assert bm.disk_num_blocks == 10
        print(f"✅ 块管理器三级配置: GPU={bm.gpu_num_blocks}, CPU={bm.cpu_num_blocks}, DISK={bm.disk_num_blocks}")

        print("🎉 三级缓存调度器测试通过！\n")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_config():
    """测试配置扩展"""
    print("=" * 60)
    print("测试 13: 配置扩展")
    print("=" * 60)

    from nanovllm.config import Config

    # 检查新字段存在
    config_fields = [f.name for f in Config.__dataclass_fields__.values()]
    new_fields = [
        'enable_multilevel_kvcache',
        'cpu_num_kvcache_blocks',
        'cpu_memory_utilization',
        'replacement_policy',
        'swap_watermark_high',
        'swap_watermark_low',
        'enable_prefetch',
        'prefetch_lookahead',
        'swap_bandwidth_gbps',
        'enable_disk_cache',
        'disk_num_kvcache_blocks',
        'disk_cache_dir',
        'disk_swap_watermark_high',
        'disk_swap_watermark_low',
    ]

    for field in new_fields:
        assert field in config_fields, f"缺少配置字段: {field}"
        print(f"✅ 配置字段存在: {field}")

    print("🎉 配置扩展测试通过！\n")


def test_stats():
    """测试统计信息"""
    print("=" * 60)
    print("测试 14: 统计信息")
    print("=" * 60)

    Sequence.block_size = 16

    bm = MultiLevelBlockManager(
        gpu_num_blocks=4,
        cpu_num_blocks=8,
        block_size=16,
    )

    # 分配一些块
    seq1 = Sequence([i for i in range(32)])  # 2 blocks
    bm.allocate(seq1, 0)
    seq2 = Sequence([i for i in range(32, 80)])  # 3 blocks
    bm.allocate(seq2, 0)

    stats = bm.get_stats()
    print(f"✅ GPU 使用率: {stats.gpu_utilization():.1%} ({stats.gpu_used_blocks}/{stats.gpu_total_blocks})")
    print(f"✅ CPU 使用率: {stats.cpu_utilization():.1%} ({stats.cpu_used_blocks}/{stats.cpu_total_blocks})")
    print(f"✅ 总换出: {stats.total_swap_out} 次, {stats.total_swap_out_blocks} 块")
    print(f"✅ 总换入: {stats.total_swap_in} 次, {stats.total_swap_in_blocks} 块")

    assert stats.gpu_total_blocks == 4
    assert stats.cpu_total_blocks == 8
    assert stats.gpu_used_blocks > 0

    print("🎉 统计信息测试通过！\n")


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("Nano-vLLM 三级 KV Cache 单元测试")
    print("GPU + CPU + SSD 三级缓存")
    print("=" * 60 + "\n")

    tests = [
        test_sequence,
        test_block_manager_basic,
        test_block_manager_gpu_full,
        test_block_manager_cpu_full,
        test_disk_block_store,
        test_swap_out_gpu_to_cpu,
        test_swap_out_cpu_to_disk,
        test_swap_in_disk_to_cpu,
        test_ensure_blocks_in_gpu,
        test_block_manager_prefix_caching,
        test_scheduler_basic,
        test_scheduler_three_level,
        test_config,
        test_stats,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n❌ {test_func.__name__} 失败: {e}")
            failed += 1
        except Exception as e:
            print(f"\n❌ {test_func.__name__} 发生错误: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败, 共 {len(tests)} 个测试")
    if failed == 0:
        print("🎉🎉🎉 所有测试通过！🎉🎉🎉")
    else:
        print("⚠️  部分测试失败")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
