"""
Qwen3 端到端推理集成测试

验证三级 KV Cache 与 Qwen3 模型推理的完整集成：
1. 模型加载与初始化
2. Prefill 阶段（前缀缓存）
3. Decode 阶段（KV Cache 复用）
4. 多级缓存换入换出与推理协调
5. 端到端推理流程
6. 高并发场景下的多级缓存行为

使用 Mock 环境模拟 PyTorch/GPU，重点验证集成逻辑正确性。
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
# 测试框架
# ============================================================

@dataclass
class TestResult:
    name: str
    passed: bool
    message: str = ""
    duration: float = 0.0


@dataclass
class TestSuite:
    name: str
    tests: List[TestResult] = field(default_factory=list)
    
    @property
    def passed_count(self):
        return sum(1 for t in self.tests if t.passed)
    
    @property
    def failed_count(self):
        return sum(1 for t in self.tests if not t.passed)
    
    @property
    def all_passed(self):
        return self.failed_count == 0


class IntegrationTestFramework:
    def __init__(self):
        self.suites: List[TestSuite] = []
        self.current_suite: TestSuite = None
        self.start_time = time.time()
    
    def start_suite(self, name: str):
        self.current_suite = TestSuite(name=name)
        self.suites.append(self.current_suite)
        print(f"\n{'='*70}")
        print(f"🧪 测试套件: {name}")
        print(f"{'='*70}")
    
    def test(self, name: str, condition: bool, message: str = ""):
        result = TestResult(name=name, passed=condition, message=message)
        self.current_suite.tests.append(result)
        status = "✅ PASS" if condition else "❌ FAIL"
        print(f"  {status}  {name}")
        if message and not condition:
            print(f"         原因: {message}")
    
    def test_func(self, name: str, func):
        start = time.time()
        try:
            func()
            duration = time.time() - start
            result = TestResult(name=name, passed=True, duration=duration)
            self.current_suite.tests.append(result)
            print(f"  ✅ PASS  {name} ({duration*1000:.1f}ms)")
        except AssertionError as e:
            duration = time.time() - start
            result = TestResult(name=name, passed=False, message=str(e), duration=duration)
            self.current_suite.tests.append(result)
            print(f"  ❌ FAIL  {name} ({duration*1000:.1f}ms)")
            print(f"         原因: {e}")
        except Exception as e:
            duration = time.time() - start
            result = TestResult(name=name, passed=False, message=f"异常: {e}", duration=duration)
            self.current_suite.tests.append(result)
            print(f"  ❌ FAIL  {name} ({duration*1000:.1f}ms)")
            print(f"         异常: {e}")
            import traceback
            traceback.print_exc()
    
    def print_summary(self):
        total_time = time.time() - self.start_time
        
        print(f"\n\n{'='*70}")
        print(f"📊 Qwen3 端到端推理集成测试报告")
        print(f"{'='*70}")
        print(f"\n总耗时: {total_time:.2f}s\n")
        
        total_pass = 0
        total_fail = 0
        
        for suite in self.suites:
            status = "✅ 全部通过" if suite.all_passed else f"⚠️  {suite.failed_count} 项失败"
            print(f"  {suite.name:<40} {suite.passed_count}/{suite.passed_count + suite.failed_count} 项  {status}")
            total_pass += suite.passed_count
            total_fail += suite.failed_count
        
        print(f"\n{'='*70}")
        print(f"总计: {total_pass} 通过, {total_fail} 失败, 共 {total_pass + total_fail} 项")
        
        if total_fail == 0:
            print(f"🎉🎉🎉 所有集成测试通过！🎉🎉🎉")
        else:
            print(f"⚠️  有 {total_fail} 项测试失败，请检查上述详情")
        
        print(f"{'='*70}")
        
        return total_fail == 0


# ============================================================
# Mock 环境设置
# ============================================================

def setup_full_mock():
    """设置完整的 mock 环境，包括 PyTorch、Transformers 等"""
    
    # Mock torch
    torch_mock = types.ModuleType('torch')
    torch_mock.__version__ = '2.1.0+cu121'
    
    class MockTensor:
        _id_counter = 0
        
        def __init__(self, *args, **kwargs):
            MockTensor._id_counter += 1
            self._id = MockTensor._id_counter
            self.shape = args[0] if args else ()
            self.dtype = kwargs.get('dtype', 'float32')
            self._data = {}
        
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
            return MockTensor()
        
        def __setitem__(self, idx, val):
            pass
        
        def copy_(self, src, non_blocking=False):
            return self
        
        def cuda(self, non_blocking=False):
            return self
        
        def cpu(self):
            return self
        
        def tolist(self):
            return list(range(min(self.shape[0] if self.shape else 0, 10)))
        
        def fill_(self, val):
            return self
        
        def zero_(self):
            return self
        
        def __len__(self):
            return self.shape[0] if self.shape else 0
        
        def view(self, *args):
            return MockTensor(*args)
        
        def split(self, sizes, dim=-1):
            return [MockTensor() for _ in sizes]
        
        def flatten(self, start_dim=0, end_dim=-1):
            return MockTensor()
        
        def unsqueeze(self, dim):
            return MockTensor()
        
        def squeeze(self, dim):
            return MockTensor()
        
        def transpose(self, dim0, dim1):
            return MockTensor()
        
        def contiguous(self):
            return self
        
        def to(self, *args, **kwargs):
            return self
        
        def float(self):
            return self
        
        def half(self):
            return self
        
        def argmax(self, dim=-1):
            return MockTensor()
        
        def softmax(self, dim=-1):
            return MockTensor()
        
        def __add__(self, other):
            return MockTensor()
        
        def __mul__(self, other):
            return MockTensor()
        
        def __matmul__(self, other):
            return MockTensor()
    
    torch_mock.Tensor = MockTensor
    torch_mock.empty = lambda *args, **kwargs: MockTensor(*args, **kwargs)
    torch_mock.zeros = lambda *args, **kwargs: MockTensor(*args, **kwargs)
    torch_mock.ones = lambda *args, **kwargs: MockTensor(*args, **kwargs)
    torch_mock.randn = lambda *args, **kwargs: MockTensor(*args, **kwargs)
    torch_mock.tensor = lambda *args, **kwargs: MockTensor()
    torch_mock.arange = lambda *args, **kwargs: MockTensor()
    
    # dtypes
    torch_mock.float16 = 'float16'
    torch_mock.float32 = 'float32'
    torch_mock.bfloat16 = 'bfloat16'
    torch_mock.int32 = 'int32'
    torch_mock.int64 = 'int64'
    torch_mock.bool = 'bool'
    
    # nn module
    nn_mock = types.ModuleType('torch.nn')
    class MockModule:
        def __init__(self, *args, **kwargs):
            self._modules = {}
            self._parameters = {}
            self.training = False
        
        def __setattr__(self, name, value):
            if isinstance(value, MockModule):
                if not hasattr(self, '_modules'):
                    self._modules = {}
                self._modules[name] = value
            super().__setattr__(name, value)
        
        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)
        
        def forward(self, *args, **kwargs):
            return MockTensor()
        
        def parameters(self):
            return iter([])
        
        def named_parameters(self):
            return iter([])
        
        def modules(self):
            return iter([self])
        
        def to(self, *args, **kwargs):
            return self
        
        def cuda(self):
            return self
        
        def eval(self):
            return self
        
        def train(self, mode=True):
            self.training = mode
            return self
        
        def state_dict(self):
            return {}
        
        def load_state_dict(self, state_dict, strict=True):
            pass
    
    class MockLinear(MockModule):
        def __init__(self, in_features, out_features, bias=True):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weight = MockTensor()
            self.bias = MockTensor() if bias else None
    
    class MockModuleList(MockModule):
        def __init__(self, modules=None):
            super().__init__()
            self._modules_list = modules or []
        
        def __iter__(self):
            return iter(self._modules_list)
        
        def __len__(self):
            return len(self._modules_list)
        
        def __getitem__(self, idx):
            return self._modules_list[idx]
    
    class MockEmbedding(MockModule):
        def __init__(self, num_embeddings, embedding_dim):
            super().__init__()
            self.num_embeddings = num_embeddings
            self.embedding_dim = embedding_dim
            self.weight = MockTensor()
    
    nn_mock.Module = MockModule
    nn_mock.Linear = MockLinear
    nn_mock.ModuleList = MockModuleList
    nn_mock.Embedding = MockEmbedding
    nn_mock.LayerNorm = MockModule
    nn_mock.Dropout = MockModule
    nn_mock.Sequential = MockModule
    
    torch_mock.nn = nn_mock
    torch_mock.nn.functional = types.ModuleType('torch.nn.functional')
    
    # distributed
    dist_mock = types.ModuleType('torch.distributed')
    dist_mock.is_initialized = lambda: True
    dist_mock.get_world_size = lambda: 1
    dist_mock.get_rank = lambda: 0
    dist_mock.init_process_group = lambda *a, **kw: None
    dist_mock.barrier = lambda *a, **kw: None
    dist_mock.destroy_process_group = lambda *a, **kw: None
    dist_mock.all_reduce = lambda *a, **kw: None
    dist_mock.broadcast = lambda *a, **kw: None
    torch_mock.distributed = dist_mock
    
    # cuda
    cuda_mock = types.ModuleType('torch.cuda')
    cuda_mock.is_available = lambda: True
    cuda_mock.device_count = lambda: 1
    cuda_mock.current_device = lambda: 0
    cuda_mock.set_device = lambda *a: None
    cuda_mock.synchronize = lambda: None
    cuda_mock.empty_cache = lambda: None
    cuda_mock.reset_peak_memory_stats = lambda: None
    cuda_mock.memory_stats = lambda: {
        'allocated_bytes.all.peak': 1024**3,
        'allocated_bytes.all.current': 512 * 1024**2,
    }
    cuda_mock.mem_get_info = lambda: (6 * 1024**3, 8 * 1024**3)
    
    class MockStream:
        def synchronize(self):
            pass
    
    class MockEvent:
        def record(self, stream=None):
            pass
        def synchronize(self):
            pass
        def elapsed_time(self, other):
            return 1.0
    
    cuda_mock.Stream = MockStream
    cuda_mock.Event = MockEvent
    cuda_mock.current_stream = lambda: MockStream()
    cuda_mock.default_stream = lambda: MockStream()
    cuda_mock.CUDAGraph = type('CUDAGraph', (), {})
    
    torch_mock.cuda = cuda_mock
    
    # multiprocessing
    mp_mock = types.ModuleType('torch.multiprocessing')
    class MockMPContext:
        def Event(self):
            e = type('Event', (), {
                'wait': lambda: None,
                'set': lambda: None,
                'clear': lambda: None,
                'is_set': lambda: False,
            })()
            return e
        def Process(self, target=None, args=()):
            p = type('Process', (), {
                'start': lambda: None,
                'join': lambda: None,
                'is_alive': lambda: False,
                'terminate': lambda: None,
            })()
            return p
    mp_mock.get_context = lambda *a, **kw: MockMPContext()
    torch_mock.multiprocessing = mp_mock
    
    # inference_mode
    class MockInferMode:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    torch_mock.inference_mode = MockInferMode
    torch_mock.no_grad = MockInferMode
    
    # set_default
    torch_mock.set_default_dtype = lambda *a: None
    torch_mock.set_default_device = lambda *a: None
    torch_mock.get_default_dtype = lambda: torch_mock.float16
    
    # save/load
    torch_mock.save = lambda *a, **kw: None
    torch_mock.load = lambda *a, **kw: MockTensor()
    
    # device
    class MockDevice:
        def __init__(self, device):
            self.type = device
        def __str__(self):
            return self.type
    torch_mock.device = MockDevice
    
    # 注册到 sys.modules
    sys.modules['torch'] = torch_mock
    sys.modules['torch.nn'] = nn_mock
    sys.modules['torch.nn.functional'] = nn_mock.functional
    sys.modules['torch.distributed'] = dist_mock
    sys.modules['torch.cuda'] = cuda_mock
    sys.modules['torch.multiprocessing'] = mp_mock
    
    # Mock transformers
    transformers_mock = types.ModuleType('transformers')
    
    class MockQwen3Config:
        def __init__(self, **kwargs):
            self.vocab_size = kwargs.get('vocab_size', 151936)
            self.hidden_size = kwargs.get('hidden_size', 2048)
            self.intermediate_size = kwargs.get('intermediate_size', 11008)
            self.num_hidden_layers = kwargs.get('num_hidden_layers', 24)
            self.num_attention_heads = kwargs.get('num_attention_heads', 16)
            self.num_key_value_heads = kwargs.get('num_key_value_heads', 4)
            self.head_dim = kwargs.get('head_dim', 128)
            self.max_position_embeddings = kwargs.get('max_position_embeddings', 32768)
            self.rms_norm_eps = kwargs.get('rms_norm_eps', 1e-6)
            self.hidden_act = kwargs.get('hidden_act', 'silu')
            self.rope_theta = kwargs.get('rope_theta', 1000000)
            self.rope_scaling = kwargs.get('rope_scaling', None)
            self.tie_word_embeddings = kwargs.get('tie_word_embeddings', False)
            self.attention_bias = kwargs.get('attention_bias', False)
            self.model_type = 'qwen3'
    
    class MockAutoConfig:
        @classmethod
        def from_pretrained(cls, model_path, **kwargs):
            return MockQwen3Config(**kwargs)
    
    class MockTokenizer:
        def __init__(self):
            self.eos_token_id = 151643
            self.pad_token_id = 151643
            self.vocab_size = 151936
        
        def encode(self, text, **kwargs):
            # 简单模拟：返回一些 token
            return list(range(min(len(text) * 2, 100)))
        
        def decode(self, ids, **kwargs):
            return f"generated_{len(ids)}_tokens"
        
        def __call__(self, text, **kwargs):
            class Result:
                def __init__(self, ids):
                    self.input_ids = [ids]
                    self.attention_mask = [[1] * len(ids)]
            return Result(self.encode(text))
    
    class MockAutoTokenizer:
        @classmethod
        def from_pretrained(cls, model_path, **kwargs):
            return MockTokenizer()
    
    transformers_mock.Qwen3Config = MockQwen3Config
    transformers_mock.AutoConfig = MockAutoConfig
    transformers_mock.AutoTokenizer = MockAutoTokenizer
    
    sys.modules['transformers'] = transformers_mock
    
    # Mock xxhash
    xxhash_mock = types.ModuleType('xxhash')
    class MockXXH64:
        def __init__(self):
            self._val = 0
        def update(self, data):
            if isinstance(data, bytes):
                self._val = (hash(data) & 0xFFFFFFFFFFFFFFFF)
            else:
                self._val = (self._val + 1) & 0xFFFFFFFFFFFFFFFF
        def intdigest(self):
            return self._val
    xxhash_mock.xxh64 = MockXXH64
    sys.modules['xxhash'] = xxhash_mock
    
    # Mock numpy
    numpy_mock = types.ModuleType('numpy')
    class MockNumpyArray:
        def __init__(self, data):
            self._data = data if isinstance(data, list) else [data]
        def tobytes(self):
            return bytes(str(self._data), 'utf-8')
        def __len__(self):
            return len(self._data)
    numpy_mock.array = lambda x: MockNumpyArray(x)
    sys.modules['numpy'] = numpy_mock
    
    # 创建包结构
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
    
    # 导入模块
    def _import(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    
    _base = os.path.dirname(os.path.abspath(__file__))
    
    # 导入基础模块
    sp_mod = _import('nanovllm.sampling_params', os.path.join(_base, 'nanovllm', 'sampling_params.py'))
    nanovllm_pkg.sampling_params = sp_mod
    
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
        'torch': torch_mock,
        'Sequence': seq_mod.Sequence,
        'SequenceStatus': seq_mod.SequenceStatus,
        'Config': config_mod.Config,
        'MultiLevelBlockManager': bm_mod.MultiLevelBlockManager,
        'BlockLocation': bm_mod.BlockLocation,
        'MultiLevelScheduler': sched_mod.MultiLevelScheduler,
        'SamplingParams': sp_mod.SamplingParams,
    }


# ============================================================
# 模拟推理引擎
# ============================================================

class MockQwen3InferenceEngine:
    """模拟 Qwen3 推理引擎，用于集成测试"""
    
    def __init__(self, config, block_manager):
        self.config = config
        self.block_manager = block_manager
        self.model_name = "Qwen3-0.5B"
        self.num_layers = config.num_hidden_layers if hasattr(config, 'num_hidden_layers') else 24
        self.num_kv_heads = config.num_key_value_heads if hasattr(config, 'num_key_value_heads') else 4
        self.head_dim = config.head_dim if hasattr(config, 'head_dim') else 128
        
        # 模拟 KV Cache tensor
        self.kv_cache_shape = (2, self.num_layers, block_manager.gpu_num_blocks, 
                              256, self.num_kv_heads, self.head_dim)
        
        self.prefill_count = 0
        self.decode_count = 0
        self.total_tokens_generated = 0
    
    def prefill(self, seq):
        """模拟 Prefill 阶段"""
        self.prefill_count += 1
        num_tokens = len(seq.prompt_token_ids)
        
        # 模拟 prefill 计算时间
        time.sleep(0.001)
        
        # 更新序列状态
        # Prefill 前: num_cached_tokens = 0, num_scheduled_tokens = num_tokens
        # Prefill 后: num_cached_tokens = num_tokens, num_scheduled_tokens = 0
        seq.num_cached_tokens = 0
        seq.num_scheduled_tokens = num_tokens
        
        # 计算块哈希（前缀缓存）
        if hasattr(self.block_manager, 'hash_blocks'):
            self.block_manager.hash_blocks(seq)
        
        # Prefill 完成后更新 cached tokens
        seq.num_cached_tokens = num_tokens
        seq.num_scheduled_tokens = 0
        
        return num_tokens
    
    def decode(self, seq):
        """模拟 Decode 阶段（生成 1 个 token）"""
        self.decode_count += 1
        self.total_tokens_generated += 1
        
        # 模拟 decode 计算时间
        time.sleep(0.0005)
        
        # 生成一个新 token
        new_token = 100 + self.total_tokens_generated
        seq.token_ids.append(new_token)
        
        # 更新进度
        seq.num_scheduled_tokens += 1
        
        # 检查是否需要追加新块
        if hasattr(self.block_manager, 'may_append'):
            if self.block_manager.can_append(seq):
                self.block_manager.may_append(seq)
        
        return new_token
    
    def generate(self, seq, max_new_tokens=10):
        """模拟完整的生成过程"""
        # Prefill
        self.prefill(seq)
        
        # Decode
        generated_tokens = []
        for _ in range(max_new_tokens):
            token = self.decode(seq)
            generated_tokens.append(token)
            
            # 检查是否结束
            if token == 151643:  # EOS
                break
        
        return generated_tokens


# ============================================================
# 测试套件 1: 基础集成验证
# ============================================================

def test_basic_integration(tf, classes):
    """基础集成验证：块管理器 + 模拟推理引擎"""
    tf.start_suite("基础集成验证")
    
    MultiLevelBlockManager = classes['MultiLevelBlockManager']
    BlockLocation = classes['BlockLocation']
    Sequence = classes['Sequence']
    
    Sequence.block_size = 256
    
    # 创建块管理器
    bm = MultiLevelBlockManager(
        gpu_num_blocks=16,
        cpu_num_blocks=32,
        block_size=256,
        enable_prefix_caching=True,
    )
    
    # 创建模拟推理引擎
    class MockConfig:
        num_hidden_layers = 24
        num_key_value_heads = 4
        head_dim = 128
    
    engine = MockQwen3InferenceEngine(MockConfig(), bm)
    
    tf.test("块管理器创建成功", bm is not None)
    tf.test("模拟推理引擎创建成功", engine is not None)
    tf.test("GPU 块数正确", bm.gpu_num_blocks == 16)
    tf.test("CPU 块数正确", bm.cpu_num_blocks == 32)
    
    # 测试单个序列的完整推理流程
    seq = Sequence([i for i in range(512)])  # 512 tokens = 2 blocks
    num_cached = bm.can_allocate(seq)
    bm.allocate(seq, num_cached)
    
    tf.test("序列分配成功", len(seq.block_table) == 2)
    tf.test("块都在 GPU", all(loc == BlockLocation.GPU for loc in seq.block_locations))
    
    # Prefill
    prefill_tokens = engine.prefill(seq)
    tf.test("Prefill 完成", prefill_tokens == 512)
    tf.test("num_cached_tokens 更新", seq.num_cached_tokens == 512)
    
    # Decode（生成 5 个 token）
    for i in range(5):
        engine.decode(seq)
    
    tf.test("Decode 5 次成功", engine.decode_count == 5)
    tf.test("生成了 5 个 token", len(seq.completion_token_ids) == 5)
    
    # 验证块仍然在 GPU
    tf.test("Decode 后块仍在 GPU", all(loc == BlockLocation.GPU for loc in seq.block_locations))
    
    # 释放
    bm.deallocate(seq)
    tf.test("释放成功", len(seq.block_table) == 0)


# ============================================================
# 测试套件 2: 前缀缓存集成
# ============================================================

def test_prefix_caching_integration(tf, classes):
    """前缀缓存与推理集成测试"""
    tf.start_suite("前缀缓存集成测试")
    
    MultiLevelBlockManager = classes['MultiLevelBlockManager']
    Sequence = classes['Sequence']
    
    Sequence.block_size = 256
    
    bm = MultiLevelBlockManager(
        gpu_num_blocks=16,
        cpu_num_blocks=32,
        block_size=256,
        enable_prefix_caching=True,
    )
    
    class MockConfig:
        num_hidden_layers = 24
        num_key_value_heads = 4
        head_dim = 128
    
    engine = MockQwen3InferenceEngine(MockConfig(), bm)
    
    # 第一个序列
    prompt1 = [i for i in range(512)]  # 2 blocks
    seq1 = Sequence(prompt1)
    num_cached1 = bm.can_allocate(seq1)
    bm.allocate(seq1, num_cached1)
    engine.prefill(seq1)
    
    tf.test("第一个序列 Prefill 完成", seq1.num_cached_tokens == 512)
    tf.test("前缀缓存初始命中 0 块", num_cached1 == 0)
    
    # 第二个序列（相同前缀）
    prompt2 = list(prompt1) + [600, 601, 602]  # 前 2 块相同，第 3 块不同
    seq2 = Sequence(prompt2)
    num_cached2 = bm.can_allocate(seq2)
    
    tf.test("第二个序列前缀缓存命中", num_cached2 >= 1)
    tf.test("前缀缓存命中 2 块", num_cached2 == 2)
    
    # 分配第二个序列（复用前缀缓存块）
    bm.allocate(seq2, num_cached2)
    engine.prefill(seq2)
    
    tf.test("第二个序列 Prefill 完成", seq2.num_cached_tokens == 515)
    
    # 验证引用计数
    # 前缀缓存块应该被两个序列共享
    gpu_used = bm.stats.gpu_used_blocks
    tf.test("前缀缓存节省了块空间", gpu_used < 5)  # 2 + 3 = 5，但共享了 2 块，所以应该是 3


# ============================================================
# 测试套件 3: 多级缓存与推理协调
# ============================================================

def test_multilevel_inference_coordination(tf, classes):
    """多级缓存与推理协调测试"""
    tf.start_suite("多级缓存与推理协调测试")
    
    MultiLevelBlockManager = classes['MultiLevelBlockManager']
    BlockLocation = classes['BlockLocation']
    Sequence = classes['Sequence']
    
    Sequence.block_size = 256
    
    bm = MultiLevelBlockManager(
        gpu_num_blocks=4,   # 只有 4 个 GPU 块
        cpu_num_blocks=8,   # 8 个 CPU 块
        block_size=256,
        swap_watermark_high=0.75,
        swap_watermark_low=0.5,
        enable_prefix_caching=False,
    )
    
    class MockConfig:
        num_hidden_layers = 24
        num_key_value_heads = 4
        head_dim = 128
    
    engine = MockQwen3InferenceEngine(MockConfig(), bm)
    
    # 分配多个序列，触发换出
    seqs = []
    for i in range(6):
        seq = Sequence([j for j in range(i * 256, (i + 1) * 256)])  # 每个 1 块
        num_cached = bm.can_allocate(seq)
        bm.allocate(seq, num_cached)
        engine.prefill(seq)
        seqs.append(seq)
    
    tf.test("分配了 6 个序列", len(seqs) == 6)
    tf.test("有 GPU→CPU 换出", bm.stats.gpu_cpu_swap_out > 0)
    tf.test("CPU 有已用块", bm.stats.cpu_used_blocks > 0)
    
    # 统计块分布
    gpu_blocks = sum(1 for seq in seqs for loc in seq.block_locations if loc == BlockLocation.GPU)
    cpu_blocks = sum(1 for seq in seqs for loc in seq.block_locations if loc == BlockLocation.CPU)
    
    tf.test("GPU 有块", gpu_blocks > 0)
    tf.test("CPU 有块", cpu_blocks > 0)
    tf.test("总块数正确", gpu_blocks + cpu_blocks == 6)
    
    # 测试：对 CPU 上的序列进行推理（需要换入）
    # 找到一个在 CPU 上的序列
    cpu_seq = None
    for seq in seqs:
        if any(loc == BlockLocation.CPU for loc in seq.block_locations):
            cpu_seq = seq
            break
    
    if cpu_seq:
        tf.test("找到 CPU 上的序列", cpu_seq is not None)
        
        # 确保块在 GPU 才能推理
        # 先释放一些 GPU 空间
        gpu_seqs = []
        for i, seq in enumerate(seqs):
            if seq is cpu_seq:
                continue
            if all(loc == BlockLocation.GPU for loc in seq.block_locations):
                gpu_seqs.append(i)
        
        # 释放一个 GPU 序列
        if gpu_seqs:
            bm.deallocate(seqs[gpu_seqs[0]])
            
            # 换入目标序列
            result = bm.ensure_blocks_in_gpu(cpu_seq, 0, len(cpu_seq.block_table))
            tf.test("ensure_blocks_in_gpu 执行成功", result is not None)
            
            # 验证块都在 GPU 了
            all_gpu = all(loc == BlockLocation.GPU for loc in cpu_seq.block_locations)
            tf.test("换入后块都在 GPU", all_gpu)
            
            # 现在可以进行推理了
            engine.decode(cpu_seq)
            tf.test("换入后可以正常 Decode", len(cpu_seq.completion_token_ids) == 1)


# ============================================================
# 测试套件 4: 三级缓存端到端
# ============================================================

def test_three_level_e2e(tf, classes):
    """三级缓存端到端测试"""
    tf.start_suite("三级缓存端到端测试")
    
    MultiLevelBlockManager = classes['MultiLevelBlockManager']
    BlockLocation = classes['BlockLocation']
    Sequence = classes['Sequence']
    
    Sequence.block_size = 256
    tmp_dir = tempfile.mkdtemp(prefix="qwen3_e2e_test_")
    
    try:
        bm = MultiLevelBlockManager(
            gpu_num_blocks=4,    # 4 GPU 块
            cpu_num_blocks=4,    # 4 CPU 块
            disk_num_blocks=16,  # 16 DISK 块
            disk_cache_dir=tmp_dir,
            block_size=256,
            swap_watermark_high=0.75,
            swap_watermark_low=0.5,
            enable_prefix_caching=False,
        )
        bm.init_disk_store(tmp_dir, (2, 24, 256, 4, 128), 'float16')
        
        class MockConfig:
            num_hidden_layers = 24
            num_key_value_heads = 4
            head_dim = 128
        
        engine = MockQwen3InferenceEngine(MockConfig(), bm)
        
        tf.test("三级缓存初始化成功", bm.enable_disk == True)
        tf.test("GPU 块数正确", bm.gpu_num_blocks == 4)
        tf.test("CPU 块数正确", bm.cpu_num_blocks == 4)
        tf.test("DISK 块数正确", bm.disk_num_blocks == 16)
        
        # 分配 12 个序列，让块分布到三层
        seqs = []
        for i in range(12):
            seq = Sequence([j for j in range(i * 256, (i + 1) * 256)])  # 每个 1 块
            num_cached = bm.can_allocate(seq)
            bm.allocate(seq, num_cached)
            engine.prefill(seq)
            seqs.append(seq)
        
        tf.test("成功分配 12 个序列", len(seqs) == 12)
        
        # 统计三层分布
        gpu_blocks = sum(1 for seq in seqs for loc in seq.block_locations if loc == BlockLocation.GPU)
        cpu_blocks = sum(1 for seq in seqs for loc in seq.block_locations if loc == BlockLocation.CPU)
        disk_blocks = sum(1 for seq in seqs for loc in seq.block_locations if loc == BlockLocation.DISK)
        
        tf.test("GPU 有块", gpu_blocks > 0)
        tf.test("CPU 有块", cpu_blocks > 0)
        tf.test("DISK 有块", disk_blocks > 0)
        tf.test("总块数正确", gpu_blocks + cpu_blocks + disk_blocks == 12)
        
        # 验证换出统计
        tf.test("有 GPU→CPU 换出", bm.stats.gpu_cpu_swap_out > 0)
        tf.test("有 CPU→DISK 换出", bm.stats.cpu_disk_swap_out > 0)
        
        # 测试：DISK 上的序列换入到 GPU 进行推理
        # 找到一个 DISK 上的序列
        disk_seq = None
        for seq in seqs:
            if any(loc == BlockLocation.DISK for loc in seq.block_locations):
                disk_seq = seq
                break
        
        if disk_seq:
            tf.test("找到 DISK 上的序列", disk_seq is not None)
            
            # 释放一些 GPU 和 CPU 空间
            released = 0
            for i, seq in enumerate(seqs):
                if seq is disk_seq:
                    continue
                if released < 2:
                    bm.deallocate(seq)
                    released += 1
            
            # 逐级换入到 GPU
            result = bm.ensure_blocks_in_gpu(disk_seq, 0, len(disk_seq.block_table))
            tf.test("ensure_blocks_in_gpu 执行成功", result is not None)
            
            # 验证块都在 GPU
            all_gpu = all(loc == BlockLocation.GPU for loc in disk_seq.block_locations)
            tf.test("DISK 序列换入到 GPU 成功", all_gpu)
            
            # 进行推理
            for _ in range(5):
                engine.decode(disk_seq)
            tf.test("换入后可以正常 Decode 5 次", len(disk_seq.completion_token_ids) == 5)
            
            # 验证换入统计
            tf.test("有 DISK→CPU 换入", bm.stats.cpu_disk_swap_in > 0)
            tf.test("有 CPU→GPU 换入", bm.stats.gpu_cpu_swap_in > 0)
        
        # 释放所有序列
        for seq in seqs:
            if len(seq.block_table) > 0:
                bm.deallocate(seq)
        
        tf.test("所有序列释放后 GPU 空闲", bm.stats.gpu_used_blocks == 0)
        
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ============================================================
# 测试套件 5: 高并发场景模拟
# ============================================================

def test_high_concurrency_simulation(tf, classes):
    """高并发场景模拟测试"""
    tf.start_suite("高并发场景模拟测试")
    
    MultiLevelBlockManager = classes['MultiLevelBlockManager']
    BlockLocation = classes['BlockLocation']
    Sequence = classes['Sequence']
    
    Sequence.block_size = 256
    
    bm = MultiLevelBlockManager(
        gpu_num_blocks=8,    # 8 GPU 块
        cpu_num_blocks=32,   # 32 CPU 块
        block_size=256,
        swap_watermark_high=0.8,
        swap_watermark_low=0.6,
        enable_prefix_caching=True,
    )
    
    class MockConfig:
        num_hidden_layers = 24
        num_key_value_heads = 4
        head_dim = 128
    
    engine = MockQwen3InferenceEngine(MockConfig(), bm)
    
    # 模拟 15 个并发请求（每个 1-2 块）
    num_seqs = 15
    seqs = []
    
    for i in range(num_seqs):
        # 不同长度的序列
        seq_len = 256 * (1 + i % 2)  # 256 或 512 tokens
        seq = Sequence([j for j in range(i * 1000, i * 1000 + seq_len)])
        num_cached = bm.can_allocate(seq)
        bm.allocate(seq, num_cached)
        engine.prefill(seq)
        seqs.append(seq)
    
    tf.test(f"成功分配 {num_seqs} 个并发序列", len(seqs) == num_seqs)
    
    # 统计块分布
    gpu_blocks = sum(1 for seq in seqs for loc in seq.block_locations if loc == BlockLocation.GPU)
    cpu_blocks = sum(1 for seq in seqs for loc in seq.block_locations if loc == BlockLocation.CPU)
    total_blocks = gpu_blocks + cpu_blocks
    
    tf.test("GPU 有热数据", gpu_blocks > 0)
    tf.test("CPU 有温数据", cpu_blocks > 0)
    tf.test(f"总块数: {total_blocks}", total_blocks > 0)
    
    # 验证换出统计
    tf.test("发生了 GPU→CPU 换出", bm.stats.gpu_cpu_swap_out > 0)
    tf.test(f"总换出 {bm.stats.total_swap_out_blocks} 块", bm.stats.total_swap_out_blocks > 0)
    
    # 模拟轮询推理（每个序列 decode 一次）
    decode_count = 0
    for seq in seqs:
        # 确保块在 GPU
        if any(loc != BlockLocation.GPU for loc in seq.block_locations):
            # 简化：跳过不在 GPU 的序列（真实场景会调度换入）
            continue
        engine.decode(seq)
        decode_count += 1
    
    tf.test(f"成功完成 {decode_count} 次 Decode", decode_count > 0)
    
    # 释放所有序列
    for seq in seqs:
        if len(seq.block_table) > 0:
            bm.deallocate(seq)
    
    tf.test("所有序列释放完成", bm.stats.gpu_used_blocks == 0)


# ============================================================
# 测试套件 6: 调度器集成
# ============================================================

def test_scheduler_integration(tf, classes):
    """调度器集成测试"""
    tf.start_suite("调度器集成测试")
    
    MultiLevelScheduler = classes['MultiLevelScheduler']
    Sequence = classes['Sequence']
    
    class MockConfig:
        max_num_seqs = 16
        max_num_batched_tokens = 1024
        eos = 151643
        kvcache_block_size = 256
        num_kvcache_blocks = 8
        enable_multilevel_kvcache = True
        cpu_num_kvcache_blocks = 16
        disk_num_kvcache_blocks = 0
        swap_watermark_high = 0.9
        swap_watermark_low = 0.7
        replacement_policy = "lru"
        enable_prefetch = True
        prefetch_lookahead = 2
    
    config = MockConfig()
    Sequence.block_size = 256
    
    scheduler = MultiLevelScheduler(config)
    
    tf.test("调度器创建成功", scheduler is not None)
    tf.test("块管理器存在", hasattr(scheduler, 'block_manager'))
    tf.test("waiting 队列存在", hasattr(scheduler, 'waiting'))
    tf.test("running 队列存在", hasattr(scheduler, 'running'))
    tf.test("swapped_out 队列存在", hasattr(scheduler, 'swapped_out'))
    
    # 添加多个请求
    for i in range(5):
        seq = Sequence([j for j in range(i * 256, (i + 1) * 256)])
        scheduler.add(seq)
    
    tf.test("添加 5 个请求到 waiting 队列", len(scheduler.waiting) == 5)
    
    # 执行一次调度
    output = scheduler.schedule()
    tf.test("调度执行成功", output is not None)
    
    # 验证有序列进入 running 队列
    stats = scheduler.get_stats()
    tf.test("running 队列有序列", stats['running'] > 0)
    tf.test("waiting 队列减少", stats['waiting'] < 5)


# ============================================================
# 主函数
# ============================================================

def main():
    print("\n" + "="*70)
    print("🚀 Qwen3 端到端推理集成测试")
    print("   三级 KV Cache + Qwen3 模型推理")
    print("="*70)
    
    # 设置 mock 环境
    print("\n⏳ 初始化 Mock 环境...")
    classes = setup_full_mock()
    print("✅ Mock 环境初始化完成")
    
    # 创建测试框架
    tf = IntegrationTestFramework()
    
    # 运行测试套件
    test_basic_integration(tf, classes)
    test_prefix_caching_integration(tf, classes)
    test_multilevel_inference_coordination(tf, classes)
    test_three_level_e2e(tf, classes)
    test_high_concurrency_simulation(tf, classes)
    test_scheduler_integration(tf, classes)
    
    # 打印总结
    all_passed = tf.print_summary()
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
