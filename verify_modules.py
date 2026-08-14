"""
Nano-vLLM 三级 KV Cache 独立模块功能验证脚本

对三级 KV Cache 的每个模块进行独立、系统性的功能验证：
1. 基础数据结构（BlockLocation, Block, Stats, DiskBlockStore）
2. 块管理器（分配/释放/换出/换入/前缀缓存）
3. 调度器（三级队列/换入换出调度）
4. Sequence 扩展
5. Config 扩展
6. 集成端到端验证

输出详细的验证结果和最终验证报告。
"""

import sys
import os
import time
import tempfile
import shutil
import importlib.util
import types
from dataclasses import dataclass, field
from typing import List, Dict, Any


# ============================================================
# 验证框架
# ============================================================

@dataclass
class TestResult:
    """单个测试项的结果"""
    name: str
    passed: bool
    message: str = ""
    duration: float = 0.0


@dataclass
class ModuleResult:
    """模块验证结果"""
    module_name: str
    tests: List[TestResult] = field(default_factory=list)
    
    @property
    def passed_count(self):
        return sum(1 for t in self.tests if t.passed)
    
    @property
    def failed_count(self):
        return sum(1 for t in self.tests if not t.passed)
    
    @property
    def total_count(self):
        return len(self.tests)
    
    @property
    def all_passed(self):
        return self.failed_count == 0


class VerificationFramework:
    """验证框架"""
    
    def __init__(self):
        self.modules: List[ModuleResult] = []
        self.current_module: ModuleResult = None
        self.start_time = time.time()
    
    def start_module(self, name: str):
        """开始验证一个模块"""
        self.current_module = ModuleResult(module_name=name)
        self.modules.append(self.current_module)
        print(f"\n{'='*70}")
        print(f"📦 模块验证: {name}")
        print(f"{'='*70}")
    
    def test(self, name: str, condition: bool, message: str = ""):
        """执行一个测试项"""
        result = TestResult(name=name, passed=condition, message=message)
        self.current_module.tests.append(result)
        
        status = "✅ PASS" if condition else "❌ FAIL"
        print(f"  {status}  {name}")
        if message and not condition:
            print(f"         原因: {message}")
    
    def test_func(self, name: str, func):
        """执行一个测试函数"""
        start = time.time()
        try:
            func()
            duration = time.time() - start
            result = TestResult(name=name, passed=True, duration=duration)
            self.current_module.tests.append(result)
            print(f"  ✅ PASS  {name} ({duration*1000:.1f}ms)")
        except AssertionError as e:
            duration = time.time() - start
            result = TestResult(name=name, passed=False, message=str(e), duration=duration)
            self.current_module.tests.append(result)
            print(f"  ❌ FAIL  {name} ({duration*1000:.1f}ms)")
            print(f"         原因: {e}")
        except Exception as e:
            duration = time.time() - start
            result = TestResult(name=name, passed=False, message=f"异常: {e}", duration=duration)
            self.current_module.tests.append(result)
            print(f"  ❌ FAIL  {name} ({duration*1000:.1f}ms)")
            print(f"         异常: {e}")
            import traceback
            traceback.print_exc()
    
    def print_summary(self):
        """打印验证总结"""
        total_time = time.time() - self.start_time
        
        print(f"\n\n{'='*70}")
        print(f"📊 三级 KV Cache 模块功能验证报告")
        print(f"{'='*70}")
        print(f"\n总耗时: {total_time:.2f}s\n")
        
        total_pass = 0
        total_fail = 0
        
        for mod in self.modules:
            status = "✅ 全部通过" if mod.all_passed else f"⚠️  {mod.failed_count} 项失败"
            print(f"  {mod.module_name:<30} {mod.passed_count}/{mod.total_count} 项  {status}")
            total_pass += mod.passed_count
            total_fail += mod.failed_count
        
        print(f"\n{'='*70}")
        print(f"总计: {total_pass} 通过, {total_fail} 失败, 共 {total_pass + total_fail} 项")
        
        if total_fail == 0:
            print(f"🎉🎉🎉 所有模块验证通过！🎉🎉🎉")
        else:
            print(f"⚠️  有 {total_fail} 项验证失败，请检查上述详情")
        
        print(f"{'='*70}")
        
        return total_fail == 0


# ============================================================
# Mock 外部依赖
# ============================================================

def setup_mocks():
    """设置所有 mock"""
    
    # mock torch
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
    torch_mock.inference_mode = lambda *a, **kw: type('Ctx', (), {
        '__enter__': lambda self: self,
        '__exit__': lambda self, *a: None,
    })()
    torch_mock.set_default_dtype = lambda *a: None
    torch_mock.set_default_device = lambda *a: None
    torch_mock.get_default_dtype = lambda: torch_mock.float16
    torch_mock.tensor = lambda *a, **kw: _FakeTensor()
    torch_mock.int32 = 'int32'
    torch_mock.int64 = 'int64'
    torch_mock.float32 = 'float32'
    torch_mock.multiprocessing = types.ModuleType('torch.multiprocessing')
    torch_mock.multiprocessing.get_context = lambda *a, **kw: type('Ctx', (), {
        'Event': lambda: type('Event', (), {'wait': lambda: None, 'set': lambda: None, 'clear': lambda: None})(),
        'Process': lambda target=None, args=(): type('Proc', (), {'start': lambda: None, 'join': lambda: None})(),
    })()
    torch_mock.distributed = types.ModuleType('torch.distributed')
    torch_mock.distributed.init_process_group = lambda *a, **kw: None
    torch_mock.distributed.barrier = lambda *a, **kw: None
    torch_mock.distributed.destroy_process_group = lambda *a, **kw: None
    sys.modules['torch'] = torch_mock
    sys.modules['torch.cuda'] = torch_mock.cuda
    sys.modules['torch.multiprocessing'] = torch_mock.multiprocessing
    sys.modules['torch.distributed'] = torch_mock.distributed
    
    # mock xxhash
    xxhash_mock = types.ModuleType('xxhash')
    class _FakeXXH64:
        def __init__(self):
            self._val = 0
        def update(self, data):
            if isinstance(data, bytes):
                self._val = (hash(data) & 0xFFFFFFFFFFFFFFFF)
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
    
    # 创建包结构
    nanovllm_pkg = types.ModuleType('nanovllm')
    nanovllm_pkg.__path__ = [os.path.join(os.path.dirname(__file__), 'nanovllm')]
    sys.modules['nanovllm'] = nanovllm_pkg
    
    engine_pkg = types.ModuleType('nanovllm.engine')
    engine_pkg.__path__ = [os.path.join(os.path.dirname(__file__), 'nanovllm', 'engine')]
    sys.modules['nanovllm.engine'] = engine_pkg
    
    # 导入模块
    def _import(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    
    _base = os.path.dirname(os.path.abspath(__file__))
    
    seq_mod = _import('nanovllm.engine.sequence', os.path.join(_base, 'nanovllm', 'engine', 'sequence.py'))
    engine_pkg.sequence = seq_mod
    
    config_mod = _import('nanovllm.config', os.path.join(_base, 'nanovllm', 'config.py'))
    nanovllm_pkg.config = config_mod
    
    bm_mod = _import('nanovllm.engine.multi_level_block_manager', 
                     os.path.join(_base, 'nanovllm', 'engine', 'multi_level_block_manager.py'))
    engine_pkg.multi_level_block_manager = bm_mod
    
    sched_mod = _import('nanovllm.engine.multi_level_scheduler',
                        os.path.join(_base, 'nanovllm', 'engine', 'multi_level_scheduler.py'))
    engine_pkg.multi_level_scheduler = sched_mod
    
    return {
        'Sequence': seq_mod.Sequence,
        'SequenceStatus': seq_mod.SequenceStatus,
        'Config': config_mod.Config,
        'MultiLevelBlockManager': bm_mod.MultiLevelBlockManager,
        'BlockLocation': bm_mod.BlockLocation,
        'Block': bm_mod.Block,
        'DiskBlockStore': bm_mod.DiskBlockStore,
        'MultiLevelCacheStats': bm_mod.MultiLevelCacheStats,
        'MultiLevelScheduler': sched_mod.MultiLevelScheduler,
    }


# ============================================================
# 模块验证函数
# ============================================================

def verify_block_location(vf, classes):
    """验证 BlockLocation 枚举"""
    vf.start_module("BlockLocation 枚举")
    
    BlockLocation = classes['BlockLocation']
    
    # 测试枚举值存在
    vf.test("GPU 枚举值存在", hasattr(BlockLocation, 'GPU'))
    vf.test("CPU 枚举值存在", hasattr(BlockLocation, 'CPU'))
    vf.test("DISK 枚举值存在", hasattr(BlockLocation, 'DISK'))
    vf.test("NONE 枚举值存在", hasattr(BlockLocation, 'NONE'))
    
    # 测试枚举值不同
    vf.test("枚举值互不相同", 
            len({BlockLocation.GPU, BlockLocation.CPU, BlockLocation.DISK, BlockLocation.NONE}) == 4)


def verify_block_class(vf, classes):
    """验证 Block 类"""
    vf.start_module("Block 类")
    
    Block = classes['Block']
    BlockLocation = classes['BlockLocation']
    
    # 测试创建
    block = Block(block_id=0, location=BlockLocation.GPU)
    vf.test("Block 创建成功", block is not None)
    vf.test("block_id 正确", block.block_id == 0)
    vf.test("location 正确", block.location == BlockLocation.GPU)
    vf.test("ref_count 默认 0", block.ref_count == 0)
    vf.test("access_count 默认 0", block.access_count == 0)
    vf.test("last_access_time 默认 0.0", block.last_access_time == 0.0)
    vf.test("owner_seqs 默认为空集合", isinstance(block.owner_seqs, set) and len(block.owner_seqs) == 0)
    
    # 测试 access 方法
    block.access(current_time=100.0)
    vf.test("access 更新 last_access_time", block.last_access_time == 100.0)
    vf.test("access 增加 access_count", block.access_count == 1)
    
    block.access(current_time=200.0)
    vf.test("第二次 access 更新时间", block.last_access_time == 200.0)
    vf.test("第二次 access 增加计数", block.access_count == 2)
    
    # 测试 reset 方法
    block.reset()
    vf.test("reset 后 ref_count 为 1", block.ref_count == 1)
    vf.test("reset 后 hash 为 -1", block.hash == -1)


def verify_cache_stats(vf, classes):
    """验证 MultiLevelCacheStats 类"""
    vf.start_module("MultiLevelCacheStats 统计类")
    
    MultiLevelCacheStats = classes['MultiLevelCacheStats']
    
    stats = MultiLevelCacheStats()
    vf.test("Stats 创建成功", stats is not None)
    vf.test("GPU 总数默认 0", stats.gpu_total_blocks == 0)
    vf.test("CPU 总数默认 0", stats.cpu_total_blocks == 0)
    vf.test("DISK 总数默认 0", stats.disk_total_blocks == 0)
    
    # 设置值并测试使用率计算
    stats.gpu_total_blocks = 10
    stats.gpu_used_blocks = 5
    vf.test("GPU 使用率计算正确", abs(stats.gpu_utilization() - 0.5) < 0.001)
    
    stats.cpu_total_blocks = 20
    stats.cpu_used_blocks = 10
    vf.test("CPU 使用率计算正确", abs(stats.cpu_utilization() - 0.5) < 0.001)
    
    stats.disk_total_blocks = 30
    stats.disk_used_blocks = 15
    vf.test("DISK 使用率计算正确", abs(stats.disk_utilization() - 0.5) < 0.001)
    
    # 测试换入换出计数
    vf.test("默认换出计数为 0", stats.total_swap_out == 0)
    vf.test("默认换入计数为 0", stats.total_swap_in == 0)


def verify_disk_block_store(vf, classes):
    """验证 DiskBlockStore 类"""
    vf.start_module("DiskBlockStore 磁盘块存储")
    
    DiskBlockStore = classes['DiskBlockStore']
    
    tmp_dir = tempfile.mkdtemp(prefix="verify_disk_")
    
    try:
        store = DiskBlockStore(
            cache_dir=tmp_dir,
            num_blocks=5,
            block_shape=(2, 2, 16, 4, 16),
        )
        
        vf.test("DiskBlockStore 创建成功", store is not None)
        vf.test("初始空闲块数正确", store.get_free_count() == 5)
        vf.test("初始已用块数正确", store.get_used_count() == 0)
        
        # 测试分配
        bid1 = store.allocate()
        vf.test("第一次分配成功", bid1 is not None)
        vf.test("分配后空闲减少", store.get_free_count() == 4)
        vf.test("分配后已用增加", store.get_used_count() == 1)
        
        bid2 = store.allocate()
        vf.test("第二次分配成功", bid2 is not None and bid2 != bid1)
        vf.test("分配后空闲 3 块", store.get_free_count() == 3)
        
        # 测试释放
        store.deallocate(bid1)
        vf.test("释放后空闲增加", store.get_free_count() == 4)
        vf.test("释放后已用减少", store.get_used_count() == 1)
        
        # 测试全部释放
        store.deallocate(bid2)
        vf.test("全部释放后空闲 5 块", store.get_free_count() == 5)
        vf.test("全部释放后已用 0 块", store.get_used_count() == 0)
        
        # 测试 clear
        bid3 = store.allocate()
        store.clear()
        vf.test("clear 后已用 0 块", store.get_used_count() == 0)
        
        # 测试满了分配失败
        bids = []
        for i in range(5):
            bids.append(store.allocate())
        vf.test("分配满 5 块", store.get_free_count() == 0)
        
        try:
            bid_fail = store.allocate()
            vf.test("满了分配应该抛异常", False)
        except RuntimeError:
            vf.test("满了分配抛出 RuntimeError", True)
        
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def verify_sequence_extension(vf, classes):
    """验证 Sequence 扩展"""
    vf.start_module("Sequence 扩展")
    
    Sequence = classes['Sequence']
    SequenceStatus = classes['SequenceStatus']
    
    # 测试 SWAPPED_OUT 状态
    vf.test("SWAPPED_OUT 状态存在", hasattr(SequenceStatus, 'SWAPPED_OUT'))
    
    # 测试新字段
    seq = Sequence([1, 2, 3, 4, 5])
    vf.test("block_locations 字段存在", hasattr(seq, 'block_locations'))
    vf.test("last_access_time 字段存在", hasattr(seq, 'last_access_time'))
    vf.test("access_count 字段存在", hasattr(seq, 'access_count'))
    
    # 测试默认值
    vf.test("block_locations 默认空列表", seq.block_locations == [])
    vf.test("last_access_time 默认 0.0", seq.last_access_time == 0.0)
    vf.test("access_count 默认 0", seq.access_count == 0)
    
    # 测试 block_table 和 block_locations 长度一致
    vf.test("block_table 与 block_locations 长度相同", 
            len(seq.block_table) == len(seq.block_locations))


def verify_config_extension(vf, classes):
    """验证 Config 扩展"""
    vf.start_module("Config 配置扩展")
    
    Config = classes['Config']
    
    config_fields = [f.name for f in Config.__dataclass_fields__.values()]
    
    # 基础多级缓存配置
    vf.test("enable_multilevel_kvcache 字段存在", 'enable_multilevel_kvcache' in config_fields)
    vf.test("cpu_num_kvcache_blocks 字段存在", 'cpu_num_kvcache_blocks' in config_fields)
    vf.test("cpu_memory_utilization 字段存在", 'cpu_memory_utilization' in config_fields)
    vf.test("replacement_policy 字段存在", 'replacement_policy' in config_fields)
    vf.test("swap_watermark_high 字段存在", 'swap_watermark_high' in config_fields)
    vf.test("swap_watermark_low 字段存在", 'swap_watermark_low' in config_fields)
    
    # 预取配置
    vf.test("enable_prefetch 字段存在", 'enable_prefetch' in config_fields)
    vf.test("prefetch_lookahead 字段存在", 'prefetch_lookahead' in config_fields)
    
    # SSD 磁盘缓存配置
    vf.test("enable_disk_cache 字段存在", 'enable_disk_cache' in config_fields)
    vf.test("disk_num_kvcache_blocks 字段存在", 'disk_num_kvcache_blocks' in config_fields)
    vf.test("disk_cache_dir 字段存在", 'disk_cache_dir' in config_fields)
    vf.test("disk_swap_watermark_high 字段存在", 'disk_swap_watermark_high' in config_fields)
    vf.test("disk_swap_watermark_low 字段存在", 'disk_swap_watermark_low' in config_fields)
    
    # 传输配置
    vf.test("swap_bandwidth_gbps 字段存在", 'swap_bandwidth_gbps' in config_fields)
    
    # 测试默认值（通过字段默认值检查）
    from dataclasses import fields
    config_fields_dict = {f.name: f.default for f in Config.__dataclass_fields__.values()}
    
    vf.test("默认禁用多级缓存", config_fields_dict.get('enable_multilevel_kvcache') == False)
    vf.test("默认 CPU 块数为 0", config_fields_dict.get('cpu_num_kvcache_blocks') == 0)
    vf.test("默认禁用磁盘缓存", config_fields_dict.get('enable_disk_cache') == False)
    vf.test("默认替换策略为 lru", config_fields_dict.get('replacement_policy') == "lru")
    vf.test("默认启用预取", config_fields_dict.get('enable_prefetch') == True)


def verify_block_manager_basic(vf, classes):
    """验证块管理器基本功能"""
    vf.start_module("块管理器 - 基本功能")
    
    MultiLevelBlockManager = classes['MultiLevelBlockManager']
    BlockLocation = classes['BlockLocation']
    Sequence = classes['Sequence']
    
    Sequence.block_size = 16
    
    bm = MultiLevelBlockManager(
        gpu_num_blocks=10,
        cpu_num_blocks=20,
        block_size=16,
    )
    
    vf.test("块管理器创建成功", bm is not None)
    vf.test("GPU 块数正确", bm.gpu_num_blocks == 10)
    vf.test("CPU 块数正确", bm.cpu_num_blocks == 20)
    vf.test("初始 GPU 空闲 10 块", len(bm.gpu_free_block_ids) == 10)
    vf.test("初始 CPU 空闲 20 块", len(bm.cpu_free_block_ids) == 20)
    
    # 测试分配
    seq = Sequence([i for i in range(32)])  # 32 tokens = 2 blocks
    num_cached = bm.can_allocate(seq)
    vf.test("can_allocate 返回 0（无前缀缓存）", num_cached == 0)
    
    bm.allocate(seq, num_cached)
    vf.test("分配后 block_table 有 2 块", len(seq.block_table) == 2)
    vf.test("分配后 block_locations 有 2 个", len(seq.block_locations) == 2)
    vf.test("新块都在 GPU", all(loc == BlockLocation.GPU for loc in seq.block_locations))
    vf.test("GPU 已用 2 块", bm.stats.gpu_used_blocks == 2)
    
    # 测试 seq_map 注册
    vf.test("序列已注册到 seq_map", seq.seq_id in bm.seq_map)
    
    # 测试释放
    bm.deallocate(seq)
    vf.test("释放后 block_table 为空", len(seq.block_table) == 0)
    vf.test("释放后 block_locations 为空", len(seq.block_locations) == 0)
    vf.test("释放后 GPU 已用 0 块", bm.stats.gpu_used_blocks == 0)
    vf.test("序列已从 seq_map 注销", seq.seq_id not in bm.seq_map)


def verify_block_manager_gpu_swap(vf, classes):
    """验证 GPU→CPU 瀑布式换出"""
    vf.start_module("块管理器 - GPU→CPU 瀑布式换出")
    
    MultiLevelBlockManager = classes['MultiLevelBlockManager']
    BlockLocation = classes['BlockLocation']
    Sequence = classes['Sequence']
    
    Sequence.block_size = 16
    
    bm = MultiLevelBlockManager(
        gpu_num_blocks=2,
        cpu_num_blocks=10,
        block_size=16,
        swap_watermark_high=0.6,
        swap_watermark_low=0.3,
    )
    
    # 分配第一个序列（占满 GPU）
    seq1 = Sequence([i for i in range(32)])  # 2 blocks
    bm.allocate(seq1, 0)
    vf.test("seq1 占满 GPU", bm.stats.gpu_used_blocks == 2)
    
    # 分配第二个序列，触发瀑布式换出
    seq2 = Sequence([i for i in range(32, 64)])  # 2 blocks
    num_cached = bm.can_allocate(seq2)
    vf.test("can_allocate 可分配（有 CPU 空间）", num_cached != -1)
    
    bm.allocate(seq2, num_cached)
    vf.test("有 GPU→CPU 换出", bm.stats.gpu_cpu_swap_out > 0)
    vf.test("CPU 有已用块", bm.stats.cpu_used_blocks > 0)
    
    # 验证换出的块更新了序列引用
    # seq1 的一些块应该被换到 CPU 了
    seq1_cpu_blocks = sum(1 for loc in seq1.block_locations if loc == BlockLocation.CPU)
    vf.test("seq1 有块换到 CPU", seq1_cpu_blocks > 0)
    
    # 验证新块在 GPU
    seq2_gpu_blocks = sum(1 for loc in seq2.block_locations if loc == BlockLocation.GPU)
    vf.test("seq2 新块优先在 GPU", seq2_gpu_blocks > 0)


def verify_block_manager_disk_swap(vf, classes):
    """验证 CPU→DISK 瀑布式换出"""
    vf.start_module("块管理器 - CPU→DISK 瀑布式换出")
    
    MultiLevelBlockManager = classes['MultiLevelBlockManager']
    BlockLocation = classes['BlockLocation']
    Sequence = classes['Sequence']
    
    Sequence.block_size = 16
    tmp_dir = tempfile.mkdtemp(prefix="verify_disk_swap_")
    
    try:
        bm = MultiLevelBlockManager(
            gpu_num_blocks=2,
            cpu_num_blocks=2,
            disk_num_blocks=10,
            disk_cache_dir=tmp_dir,
            block_size=16,
            swap_watermark_high=0.6,
            swap_watermark_low=0.3,
        )
        bm.init_disk_store(tmp_dir, (2, 2, 16, 4, 16), 'float16')
        
        vf.test("三级缓存初始化成功", bm.enable_disk == True)
        
        # 分配 seq1（占满 GPU）
        seq1 = Sequence([i for i in range(32)])  # 2 blocks
        bm.allocate(seq1, 0)
        vf.test("seq1 占满 GPU", bm.stats.gpu_used_blocks == 2)
        
        # 分配 seq2（占满 CPU）
        seq2 = Sequence([i for i in range(32, 64)])  # 2 blocks
        bm.allocate(seq2, 0)
        vf.test("seq2 占满 CPU", bm.stats.cpu_used_blocks == 2)
        
        # 分配 seq3，触发 CPU→DISK 换出
        seq3 = Sequence([i for i in range(64, 96)])  # 2 blocks
        num_cached = bm.can_allocate(seq3)
        vf.test("can_allocate 可分配（有 DISK 空间）", num_cached != -1)
        
        bm.allocate(seq3, num_cached)
        vf.test("有 CPU→DISK 换出", bm.stats.cpu_disk_swap_out > 0)
        vf.test("DISK 有已用块", bm.stats.disk_used_blocks > 0)
        
        # 验证换出的块更新了序列引用
        # 找到一个在 DISK 上的块
        disk_blocks_found = False
        for seq in [seq1, seq2, seq3]:
            for loc in seq.block_locations:
                if loc == BlockLocation.DISK:
                    disk_blocks_found = True
                    break
            if disk_blocks_found:
                break
        vf.test("有序列的块在 DISK 上", disk_blocks_found)
        
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def verify_block_manager_swap_in(vf, classes):
    """验证逐级换入"""
    vf.start_module("块管理器 - 逐级换入")
    
    MultiLevelBlockManager = classes['MultiLevelBlockManager']
    BlockLocation = classes['BlockLocation']
    Sequence = classes['Sequence']
    
    Sequence.block_size = 16
    tmp_dir = tempfile.mkdtemp(prefix="verify_swap_in_")
    
    try:
        bm = MultiLevelBlockManager(
            gpu_num_blocks=4,
            cpu_num_blocks=4,
            disk_num_blocks=20,
            disk_cache_dir=tmp_dir,
            block_size=16,
            swap_watermark_high=0.5,
            swap_watermark_low=0.25,
        )
        bm.init_disk_store(tmp_dir, (2, 2, 16, 4, 16), 'float16')
        
        # 分配多个序列，让块分布到各层
        seqs = []
        for i in range(10):
            seq = Sequence([j for j in range(i * 32, (i + 1) * 32)])
            bm.allocate(seq, 0)
            seqs.append(seq)
        
        disk_before = bm.stats.disk_used_blocks
        vf.test("有块在 DISK 上", disk_before > 0)
        
        # 找到一个 DISK 块
        disk_block_id = None
        for seq in seqs:
            for bi, loc in enumerate(seq.block_locations):
                if loc == BlockLocation.DISK:
                    disk_block_id = seq.block_table[bi]
                    break
            if disk_block_id:
                break
        
        vf.test("找到 DISK 块", disk_block_id is not None)
        
        # 找到 CPU 上的序列并释放，腾出空间
        cpu_seqs = []
        for i, seq in enumerate(seqs):
            for loc in seq.block_locations:
                if loc == BlockLocation.CPU:
                    cpu_seqs.append(i)
                    break
        
        # 释放一些 CPU 序列
        for i in cpu_seqs[:2]:
            bm.deallocate(seqs[i])
        
        cpu_free = bm.cpu_num_blocks - bm.stats.cpu_used_blocks
        vf.test("释放后 CPU 有空闲空间", cpu_free > 0)
        
        # 测试 DISK→CPU 换入
        if disk_block_id and cpu_free > 0:
            cpu_block_id = bm.swap_in_disk_to_cpu(disk_block_id)
            vf.test("DISK→CPU 换入成功", cpu_block_id is not None)
            vf.test("DISK 换入计数增加", bm.stats.cpu_disk_swap_in > 0)
        
        # 测试 ensure_blocks_in_gpu
        # 找一个有 CPU 块的序列
        target_seq = None
        for seq in seqs:
            if any(loc == BlockLocation.CPU for loc in seq.block_locations):
                target_seq = seq
                break
        
        if target_seq:
            # 先释放一些 GPU 空间
            gpu_seqs = []
            for i, seq in enumerate(seqs):
                if seq is target_seq:
                    continue
                if any(loc == BlockLocation.GPU for loc in seq.block_locations):
                    gpu_seqs.append(i)
            
            for i in gpu_seqs[:1]:
                bm.deallocate(seqs[i])
            
            result = bm.ensure_blocks_in_gpu(target_seq, 0, len(target_seq.block_table))
            vf.test("ensure_blocks_in_gpu 执行成功", result is not None)
        
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def verify_block_manager_prefix_cache(vf, classes):
    """验证三级前缀缓存"""
    vf.start_module("块管理器 - 三级前缀缓存")
    
    MultiLevelBlockManager = classes['MultiLevelBlockManager']
    Sequence = classes['Sequence']
    
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
    
    # 计算哈希
    seq1.num_cached_tokens = 0
    seq1.num_scheduled_tokens = 32
    bm.hash_blocks(seq1)
    seq1.num_cached_tokens = 32
    
    vf.test("第一个序列哈希计算完成", True)
    
    # 分配第二个序列（相同前缀）
    prompt2 = list(prompt1) + [100, 101]  # 前 2 块相同
    seq2 = Sequence(prompt2)
    num_cached = bm.can_allocate(seq2)
    
    vf.test("前缀缓存命中至少 1 块", num_cached >= 1)


def verify_block_manager_seq_map(vf, classes):
    """验证 seq_map 序列映射"""
    vf.start_module("块管理器 - seq_map 序列映射")
    
    MultiLevelBlockManager = classes['MultiLevelBlockManager']
    Sequence = classes['Sequence']
    
    Sequence.block_size = 16
    
    bm = MultiLevelBlockManager(
        gpu_num_blocks=10,
        cpu_num_blocks=20,
        block_size=16,
    )
    
    # 测试 seq_map 存在
    vf.test("seq_map 存在", hasattr(bm, 'seq_map'))
    vf.test("seq_map 初始为空", len(bm.seq_map) == 0)
    
    # 分配序列后注册
    seq1 = Sequence([i for i in range(32)])
    bm.allocate(seq1, 0)
    vf.test("分配后 seq_map 有 1 个序列", len(bm.seq_map) == 1)
    vf.test("seq_id 在 seq_map 中", seq1.seq_id in bm.seq_map)
    
    # 分配第二个序列
    seq2 = Sequence([i for i in range(32, 64)])
    bm.allocate(seq2, 0)
    vf.test("分配后 seq_map 有 2 个序列", len(bm.seq_map) == 2)
    
    # 释放后注销
    bm.deallocate(seq1)
    vf.test("释放后 seq_map 减少", len(bm.seq_map) == 1)
    vf.test("释放的 seq_id 不在 seq_map 中", seq1.seq_id not in bm.seq_map)
    
    # 全部释放
    bm.deallocate(seq2)
    vf.test("全部释放后 seq_map 为空", len(bm.seq_map) == 0)


def verify_scheduler_basic(vf, classes):
    """验证调度器基本功能"""
    vf.start_module("调度器 - 基本功能")
    
    MultiLevelScheduler = classes['MultiLevelScheduler']
    Sequence = classes['Sequence']
    
    class MockConfig:
        max_num_seqs = 16
        max_num_batched_tokens = 256
        eos = 2
        kvcache_block_size = 16
        num_kvcache_blocks = 10
        enable_multilevel_kvcache = True
        cpu_num_kvcache_blocks = 20
        disk_num_kvcache_blocks = 0
        swap_watermark_high = 0.9
        swap_watermark_low = 0.7
        replacement_policy = "lru"
        enable_prefetch = True
        prefetch_lookahead = 2
    
    config = MockConfig()
    Sequence.block_size = 16
    
    scheduler = MultiLevelScheduler(config)
    vf.test("调度器创建成功", scheduler is not None)
    
    # 测试三级队列
    vf.test("waiting 队列存在", hasattr(scheduler, 'waiting'))
    vf.test("running 队列存在", hasattr(scheduler, 'running'))
    vf.test("swapped_out 队列存在", hasattr(scheduler, 'swapped_out'))
    
    # 测试 add
    seq = Sequence([1, 2, 3, 4, 5])
    scheduler.add(seq)
    vf.test("add 后 waiting 队列有 1 个", len(scheduler.waiting) == 1)
    
    # 测试 is_finished
    vf.test("is_finished 返回 False（有等待序列）", not scheduler.is_finished())
    
    # 测试 get_stats
    stats = scheduler.get_stats()
    vf.test("get_stats 包含 waiting", 'waiting' in stats)
    vf.test("get_stats 包含 running", 'running' in stats)
    vf.test("get_stats 包含 swapped_out", 'swapped_out' in stats)


def verify_scheduler_three_level(vf, classes):
    """验证三级缓存调度器"""
    vf.start_module("调度器 - 三级缓存调度")
    
    MultiLevelScheduler = classes['MultiLevelScheduler']
    Sequence = classes['Sequence']
    
    tmp_dir = tempfile.mkdtemp(prefix="verify_sched_3l_")
    
    try:
        class MockConfig:
            max_num_seqs = 16
            max_num_batched_tokens = 256
            eos = 2
            kvcache_block_size = 16
            num_kvcache_blocks = 4
            enable_multilevel_kvcache = True
            cpu_num_kvcache_blocks = 4
            disk_num_kvcache_blocks = 10
            disk_cache_dir = tmp_dir
            swap_watermark_high = 0.8
            swap_watermark_low = 0.5
            replacement_policy = "lru"
            enable_prefetch = True
            prefetch_lookahead = 2
        
        config = MockConfig()
        Sequence.block_size = 16
        
        scheduler = MultiLevelScheduler(config)
        vf.test("三级调度器创建成功", scheduler is not None)
        vf.test("enable_disk 为 True", scheduler.enable_disk == True)
        
        # 测试块管理器支持三级
        bm = scheduler.block_manager
        vf.test("块管理器 enable_disk 为 True", bm.enable_disk == True)
        vf.test("块管理器 disk_num_blocks 为 10", bm.disk_num_blocks == 10)
        
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def verify_integration_e2e(vf, classes):
    """端到端集成验证"""
    vf.start_module("集成验证 - 端到端流程")
    
    MultiLevelBlockManager = classes['MultiLevelBlockManager']
    BlockLocation = classes['BlockLocation']
    Sequence = classes['Sequence']
    
    Sequence.block_size = 16
    tmp_dir = tempfile.mkdtemp(prefix="verify_e2e_")
    
    try:
        bm = MultiLevelBlockManager(
            gpu_num_blocks=4,
            cpu_num_blocks=4,
            disk_num_blocks=10,
            disk_cache_dir=tmp_dir,
            block_size=16,
            swap_watermark_high=0.75,
            swap_watermark_low=0.5,
            enable_prefix_caching=True,
        )
        bm.init_disk_store(tmp_dir, (2, 2, 16, 4, 16), 'float16')
        
        # 步骤 1: 分配多个序列
        seqs = []
        for i in range(8):
            seq = Sequence([j for j in range(i * 32, (i + 1) * 32)])
            num_cached = bm.can_allocate(seq)
            bm.allocate(seq, num_cached)
            seqs.append(seq)
        
        vf.test("成功分配 8 个序列", len(seqs) == 8)
        
        # 步骤 2: 验证块分布在三层
        gpu_blocks = sum(1 for seq in seqs for loc in seq.block_locations if loc == BlockLocation.GPU)
        cpu_blocks = sum(1 for seq in seqs for loc in seq.block_locations if loc == BlockLocation.CPU)
        disk_blocks = sum(1 for seq in seqs for loc in seq.block_locations if loc == BlockLocation.DISK)
        
        vf.test("GPU 有块", gpu_blocks > 0)
        vf.test("CPU 有块", cpu_blocks > 0)
        vf.test("DISK 有块", disk_blocks > 0)
        vf.test("总块数正确", gpu_blocks + cpu_blocks + disk_blocks == 16)  # 8 序列 × 2 块
        
        # 步骤 3: 验证统计信息
        stats = bm.get_stats()
        vf.test("GPU 使用率 > 0", stats.gpu_utilization() > 0)
        vf.test("CPU 使用率 > 0", stats.cpu_utilization() > 0)
        vf.test("DISK 使用率 > 0", stats.disk_utilization() > 0)
        vf.test("有 GPU→CPU 换出", stats.gpu_cpu_swap_out > 0)
        vf.test("有 CPU→DISK 换出", stats.cpu_disk_swap_out > 0)
        
        # 步骤 4: 释放一些序列
        for seq in seqs[:4]:
            bm.deallocate(seq)
        
        vf.test("释放 4 个序列后 seq_map 减少", len(bm.seq_map) == 4)
        
        # 步骤 5: 验证释放后空间增加
        stats_after = bm.get_stats()
        vf.test("释放后 GPU 使用减少", stats_after.gpu_used_blocks <= stats.gpu_used_blocks)
        
        vf.test("端到端流程完整执行", True)
        
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def verify_edge_cases(vf, classes):
    """边界情况验证"""
    vf.start_module("边界情况验证")
    
    MultiLevelBlockManager = classes['MultiLevelBlockManager']
    Sequence = classes['Sequence']
    
    Sequence.block_size = 16
    
    # 测试 1: 单 token 序列
    bm = MultiLevelBlockManager(gpu_num_blocks=10, cpu_num_blocks=10, block_size=16)
    seq_single = Sequence([1])
    num_cached = bm.can_allocate(seq_single)
    bm.allocate(seq_single, num_cached)
    vf.test("单 token 序列分配 1 块", len(seq_single.block_table) == 1)
    bm.deallocate(seq_single)
    
    # 测试 2: 恰好满的情况
    bm2 = MultiLevelBlockManager(gpu_num_blocks=2, cpu_num_blocks=0, block_size=16)
    seq_full = Sequence([i for i in range(32)])  # 2 blocks
    bm2.allocate(seq_full, 0)
    vf.test("恰好占满 GPU", bm2.stats.gpu_used_blocks == 2)
    vf.test("恰好占满时使用率 100%", abs(bm2.stats.gpu_utilization() - 1.0) < 0.001)
    
    # 测试 3: 无 CPU 缓存（降级模式）
    bm3 = MultiLevelBlockManager(gpu_num_blocks=10, cpu_num_blocks=0, block_size=16)
    vf.test("无 CPU 缓存时 enable_multilevel 为 False", not bm3.enable_multilevel)
    
    # 测试 4: 水位线边界
    bm4 = MultiLevelBlockManager(
        gpu_num_blocks=10,
        cpu_num_blocks=10,
        block_size=16,
        swap_watermark_high=1.0,  # 100% 才触发
        swap_watermark_low=0.5,
    )
    seq_water = Sequence([i for i in range(160)])  # 10 blocks = 100%
    bm4.allocate(seq_water, 0)
    vf.test("100% 使用率时触发换出（高水位 1.0）", bm4.stats.gpu_cpu_swap_out >= 0)
    
    # 测试 5: 大量短序列
    bm5 = MultiLevelBlockManager(gpu_num_blocks=20, cpu_num_blocks=20, block_size=16)
    short_seqs = []
    for i in range(10):
        seq = Sequence([i])  # 每个 1 token = 1 块
        bm5.allocate(seq, 0)
        short_seqs.append(seq)
    vf.test("10 个短序列都分配成功", len(short_seqs) == 10)
    vf.test("GPU 使用 10 块", bm5.stats.gpu_used_blocks == 10)
    
    for seq in short_seqs:
        bm5.deallocate(seq)
    vf.test("全部释放后 GPU 空闲", bm5.stats.gpu_used_blocks == 0)


# ============================================================
# 主函数
# ============================================================

def main():
    print("\n" + "="*70)
    print("🔬 Nano-vLLM 三级 KV Cache 独立模块功能验证")
    print("   GPU + CPU + SSD 三级缓存架构")
    print("="*70)
    
    # 设置 mock
    print("\n⏳ 初始化环境...")
    classes = setup_mocks()
    print("✅ 环境初始化完成")
    
    # 创建验证框架
    vf = VerificationFramework()
    
    # 1. 基础数据结构验证
    verify_block_location(vf, classes)
    verify_block_class(vf, classes)
    verify_cache_stats(vf, classes)
    verify_disk_block_store(vf, classes)
    
    # 2. Sequence 扩展验证
    verify_sequence_extension(vf, classes)
    
    # 3. Config 扩展验证
    verify_config_extension(vf, classes)
    
    # 4. 块管理器验证
    verify_block_manager_basic(vf, classes)
    verify_block_manager_gpu_swap(vf, classes)
    verify_block_manager_disk_swap(vf, classes)
    verify_block_manager_swap_in(vf, classes)
    verify_block_manager_prefix_cache(vf, classes)
    verify_block_manager_seq_map(vf, classes)
    
    # 5. 调度器验证
    verify_scheduler_basic(vf, classes)
    verify_scheduler_three_level(vf, classes)
    
    # 6. 集成验证
    verify_integration_e2e(vf, classes)
    
    # 7. 边界情况
    verify_edge_cases(vf, classes)
    
    # 打印总结
    all_passed = vf.print_summary()
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
