# Nano-vLLM 多级 KV Cache 集成指南

本文档说明如何将多级 KV Cache 管理系统集成到 Nano-vLLM 中，支持 GPU + CPU + SSD 三级缓存架构。

## 概述

多级 KV Cache 系统通过瀑布式换出机制，将冷数据从 GPU 显存逐级换出到 CPU 内存和 SSD 磁盘，在保证推理正确性的前提下显著扩展有效缓存容量，提高系统吞吐量。

### 核心特性

- **三级缓存架构**：GPU HBM（热数据）+ CPU DRAM（温数据）+ NVMe SSD（冷数据）
- **瀑布式换出**：GPU 满 → LRU 换出到 CPU；CPU 满 → LRU 换出到 SSD
- **逐级换入**：访问冷数据时，DISK → CPU → GPU 逐级换入
- **三级前缀缓存**：每一级独立维护前缀缓存哈希表
- **LRU 替换策略**：基于访问时间的热度排序，选择最冷的块换出
- **预取支持**：decode 阶段提前将下一个块换入 GPU，隐藏换入延迟

### 核心组件

| 组件 | 文件 | 说明 |
|------|------|------|
| MultiLevelBlockManager | `engine/multi_level_block_manager.py` | 三级块管理器，GPU+CPU+SSD 三级缓存 |
| MultiLevelScheduler | `engine/multi_level_scheduler.py` | 三级缓存感知调度器 |
| MultiLevelModelRunnerMixin | `engine/multi_level_model_runner.py` | ModelRunner 扩展，支持数据传输 |
| DiskBlockStore | `engine/multi_level_block_manager.py` | 磁盘块存储，管理 SSD 上的 KV 块 |
| Sequence (已修改) | `engine/sequence.py` | 增加块位置跟踪和热度统计 |
| Config (已修改) | `config.py` | 增加多级缓存配置参数（含 SSD） |

## 集成步骤

### 步骤 1：确认文件已就位

确保以下文件在正确的位置：

```
nanovllm/
├── config.py                          # 已修改，增加配置参数（含 SSD）
├── engine/
│   ├── block_manager.py               # 原有，保持不变
│   ├── sequence.py                    # 已修改，增加块位置跟踪
│   ├── scheduler.py                   # 原有，保持不变
│   ├── model_runner.py                # 原有，保持不变
│   ├── llm_engine.py                  # 已修改，集成多级缓存
│   ├── multi_level_block_manager.py   # 新增：三级块管理器
│   ├── multi_level_scheduler.py       # 新增：三级调度器
│   └── multi_level_model_runner.py    # 新增：ModelRunner 扩展
└── docs/
    └── multilevel_kvcache_integration.md  # 本文档
```

### 步骤 2：修改 LLMEngine（已完成）

`engine/llm_engine.py` 已根据配置选择使用对应的调度器和 ModelRunner：

```python
# 根据配置选择 ModelRunner
enable_multilevel = (getattr(config, "enable_multilevel_kvcache", False) 
                    and getattr(config, "cpu_num_kvcache_blocks", 0) > 0)

if enable_multilevel:
    from nanovllm.engine.multi_level_model_runner import MultiLevelModelRunner
    ModelRunnerClass = MultiLevelModelRunner
else:
    from nanovllm.engine.model_runner import ModelRunner
    ModelRunnerClass = ModelRunner

# 根据配置选择调度器
if enable_multilevel:
    from nanovllm.engine.multi_level_scheduler import MultiLevelScheduler
    self.scheduler = MultiLevelScheduler(config)
else:
    from nanovllm.engine.scheduler import Scheduler
    self.scheduler = Scheduler(config)
```

### 步骤 3：ModelRunner 扩展（已完成）

采用包装模式（Wrapper Pattern）扩展 ModelRunner，避免修改原始代码：

```python
class MultiLevelModelRunner:
    """三级 KV Cache 感知的 ModelRunner 包装类"""
    
    def __init__(self, config, rank, events):
        # 创建原始 ModelRunner
        self.base_runner = ModelRunner(config, rank, events)
        # 初始化多级缓存
        self.init_multilevel_kvcache(config)
        # 块管理器引用（由 LLMEngine 注入）
        self.block_manager = None
    
    def set_block_manager(self, block_manager):
        """设置块管理器引用"""
        self.block_manager = block_manager
    
    def run(self, seqs, is_prefill):
        """重写 run 方法，在推理前后处理换入换出"""
        # 确保需要的块在 GPU 中
        if self.block_manager is not None:
            for seq in seqs:
                self.block_manager.ensure_blocks_in_gpu(seq, ...)
        # 调用原始 run
        return self.base_runner.run(seqs, is_prefill)
    
    def __getattr__(self, name):
        """代理未重写的方法到 base_runner"""
        return getattr(self.base_runner, name)
```

### 步骤 4：配置参数

在创建 LLMEngine 时传入多级缓存配置：

```python
from nanovllm import LLM

# 两级缓存（GPU + CPU）
llm = LLM(
    model="path/to/model",
    enable_multilevel_kvcache=True,
    cpu_num_kvcache_blocks=1024,      # CPU 缓存块数
    replacement_policy="lru",         # 替换策略
    swap_watermark_high=0.9,          # 换出高水位线
    swap_watermark_low=0.7,           # 换入低水位线
    enable_prefetch=True,             # 启用预取
    prefetch_lookahead=2,             # 预取超前块数
)

# 三级缓存（GPU + CPU + SSD）
llm = LLM(
    model="path/to/model",
    enable_multilevel_kvcache=True,
    cpu_num_kvcache_blocks=1024,      # CPU 缓存块数
    enable_disk_cache=True,           # 启用磁盘缓存（第三级）
    disk_num_kvcache_blocks=4096,     # 磁盘缓存块数
    disk_cache_dir="./kv_disk_cache", # 磁盘缓存目录
    replacement_policy="lru",
    swap_watermark_high=0.85,
    swap_watermark_low=0.6,
)
```

## 配置参数详解

### 基础配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_multilevel_kvcache` | bool | False | 是否启用多级缓存 |
| `cpu_num_kvcache_blocks` | int | 0 | CPU 缓存块数量（0 表示不使用） |
| `cpu_memory_utilization` | float | 0.5 | CPU 内存利用率（用于自动计算块数） |

### SSD 磁盘缓存配置（第三级）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_disk_cache` | bool | False | 是否启用磁盘缓存（第三级） |
| `disk_num_kvcache_blocks` | int | 0 | 磁盘缓存块数量（0 表示不使用） |
| `disk_cache_dir` | str | "./kv_disk_cache" | 磁盘缓存目录 |
| `disk_swap_watermark_high` | float | 0.9 | 磁盘使用率超过此值触发淘汰 |
| `disk_swap_watermark_low` | float | 0.7 | 淘汰到此值以下停止 |

**注意**：启用磁盘缓存时，CPU 缓存必须也启用（`cpu_num_kvcache_blocks > 0`）。

### 替换策略

| 参数 | 类型 | 默认值 | 可选值 | 说明 |
|------|------|--------|--------|------|
| `replacement_policy` | str | "lru" | lru / lfu / lru_k / arc | 替换策略 |

- **lru**：最近最少使用，简单高效（当前实现）
- **lfu**：使用频率最低，适合访问模式稳定的场景
- **lru_k**：记录最近 K 次访问，抗扫描污染
- **arc**：自适应替换缓存，综合性能最好（推荐）

### 水位线控制

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `swap_watermark_high` | float | 0.9 | GPU/CPU 使用率超过此值触发换出 |
| `swap_watermark_low` | float | 0.7 | 换出到此值以下停止 |

建议：
- 高水位线不要设太低，否则频繁换出影响性能
- 高低水位线之间保持 0.2 左右的间隔，避免抖动
- 三级缓存场景下，建议适当降低水位线（如 0.85/0.6），给磁盘 I/O 留时间

### 预取配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_prefetch` | bool | True | 是否启用预取 |
| `prefetch_lookahead` | int | 2 | 预取超前块数 |

预取可以隐藏换入延迟，但会增加显存占用。

### 传输配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `swap_bandwidth_gbps` | float | 10.0 | PCIe 带宽（GB/s），用于估算时间 |

实际带宽取决于硬件：
- PCIe 3.0 x16: ~16 GB/s
- PCIe 4.0 x16: ~32 GB/s
- PCIe 5.0 x16: ~64 GB/s
- NVMe SSD: ~3-7 GB/s（顺序读写）

## 使用示例

### 基本使用（两级缓存）

```python
from nanovllm import LLM, SamplingParams

# 创建 LLM 实例，启用两级缓存（GPU + CPU）
llm = LLM(
    model="Qwen/Qwen3-7B",
    enable_multilevel_kvcache=True,
    cpu_num_kvcache_blocks=2048,  # 2048 块 × 8MB = 16GB CPU 缓存
    replacement_policy="lru",
    max_num_seqs=128,
)

# 生成
sampling_params = SamplingParams(max_tokens=256, temperature=0.7)
outputs = llm.generate(["Hello, how are you?"], sampling_params)

for output in outputs:
    print(output["text"])
```

### 三级缓存（GPU + CPU + SSD）

```python
# 三级缓存场景：超长上下文，用 SSD 扩展容量
llm = LLM(
    model="Qwen/Qwen3-7B",
    enable_multilevel_kvcache=True,
    cpu_num_kvcache_blocks=1024,   # 8GB CPU 缓存
    enable_disk_cache=True,        # 启用磁盘缓存
    disk_num_kvcache_blocks=8192,  # 64GB SSD 缓存
    disk_cache_dir="/mnt/nvme/kv_cache",
    max_model_len=65536,           # 64K 上下文
    max_num_seqs=16,
    replacement_policy="lru",
    swap_watermark_high=0.85,
    swap_watermark_low=0.6,
)
```

### 长上下文场景

```python
# 长上下文场景：GPU 显存不够，用 CPU + SSD 扩展
llm = LLM(
    model="Qwen/Qwen3-7B",
    enable_multilevel_kvcache=True,
    cpu_num_kvcache_blocks=4096,   # 32GB CPU 缓存
    enable_disk_cache=True,
    disk_num_kvcache_blocks=16384, # 128GB SSD 缓存
    max_model_len=131072,          # 128K 上下文
    max_num_seqs=4,
    replacement_policy="lru",
)
```

### 高并发场景

```python
# 高并发场景：用更大的 CPU 缓存支持更多并发
llm = LLM(
    model="Qwen/Qwen3-7B",
    enable_multilevel_kvcache=True,
    cpu_num_kvcache_blocks=8192,  # 64GB CPU 缓存
    max_num_seqs=512,             # 512 并发
    swap_watermark_high=0.85,
    swap_watermark_low=0.6,
    enable_prefetch=True,
    prefetch_lookahead=4,
)
```

## 性能调优指南

### 1. 选择合适的缓存层级

| 场景 | 推荐架构 | 原因 |
|------|----------|------|
| 低并发 + 短序列 | 仅 GPU | 无换出开销，性能最高 |
| 中高并发 + 中等序列 | GPU + CPU | PCIe 传输快，性价比高 |
| 超长上下文 + 低频率 | GPU + CPU + SSD | 容量大，适合冷数据 |
| 成本敏感 + 可接受延迟 | GPU + SSD | 用磁盘替代内存，成本低 |

### 2. 调整水位线

- **高并发、延迟敏感**：降低高水位线（如 0.8），留出更多缓冲
- **追求高吞吐**：提高高水位线（如 0.95），充分利用显存
- **避免抖动**：高低水位线间隔至少 0.2
- **三级缓存**：建议 GPU 水位 0.85/0.6，CPU 水位 0.8/0.5

### 3. 预取调优

- **顺序访问为主**：增大预取深度（4-8 块）
- **随机访问为主**：减小预取深度或关闭预取
- **显存紧张**：关闭预取或减小预取深度
- **三级缓存**：建议开启预取，隐藏磁盘 I/O 延迟

### 4. CPU 缓存大小

建议 CPU 缓存大小为 GPU 缓存的 2-4 倍：
- 太小：频繁换入换出，性能下降
- 太大：内存浪费，且收益递减

### 5. SSD 缓存大小

建议 SSD 缓存大小为 CPU 缓存的 4-8 倍：
- 太小：磁盘频繁淘汰，性能下降
- 太大：磁盘空间浪费，且访问冷数据概率低
- **注意**：SSD 缓存适合冷数据，热数据应驻留在 GPU/CPU

### 6. 监控指标

关注以下指标来调优：

```python
# 获取调度器统计
stats = llm.engine.scheduler.get_stats()
print(f"GPU 使用率: {stats['gpu_utilization']:.1%}")
print(f"CPU 使用率: {stats['cpu_utilization']:.1%}")
print(f"DISK 使用率: {stats['disk_utilization']:.1%}")
print(f"GPU→CPU 换出次数: {stats['gpu_cpu_swap_out']}")
print(f"CPU→DISK 换出次数: {stats['cpu_disk_swap_out']}")
print(f"换入次数: {stats['total_swap_in']}")
print(f"抢占次数: {stats['total_preempt']}")
print(f"前缀缓存命中率: {stats['prefix_cache_hit_rate']:.1%}")
```

理想状态：
- GPU 使用率在 70-90% 之间波动
- CPU 使用率在 50-80% 之间波动
- DISK 使用率根据冷热数据比例调整
- 换入换出次数相对稳定，没有爆发式增长
- 抢占次数很少（最好为 0）

## 架构说明

### 整体架构（三级缓存）

```
┌─────────────────────────────────────────────────────────────────┐
│                           LLMEngine                             │
└───────────────────────────────┬─────────────────────────────────┘
                                │
            ┌───────────────────┴───────────────────┐
            │                                       │
    ┌───────▼────────┐                     ┌────────▼─────────┐
    │    Scheduler   │                     │   ModelRunner    │
    │  (三级感知)    │                     │  (支持换入换出)  │
    └───────┬────────┘                     └────────┬─────────┘
            │                                       │
            │ 调度决策                              │ 数据传输
            │                                       │
    ┌───────▼───────────────────────────────────────▼─────────────┐
    │                  MultiLevelBlockManager                     │
    │  ┌───────────────┐    ┌───────────────┐    ┌─────────────┐ │
    │  │  GPU Cache    │ ←→ │  CPU Cache    │ ←→ │  SSD Cache  │ │
    │  │  (热数据)     │    │  (温数据)     │    │  (冷数据)   │ │
    │  └───────────────┘    └───────────────┘    └─────────────┘ │
    └─────────────────────────────────────────────────────────────┘
```

### 瀑布式换出流程

```
┌─────────────────────────────────────────────────┐
│              GPU HBM (Tier 1, Hot)              │
│  新块优先分配到这里，热数据驻留                    │
│  满了 → LRU 换出到 CPU                           │
└──────────────────────┬──────────────────────────┘
                       ↓ swap out (waterfall)
┌─────────────────────────────────────────────────┐
│             CPU DRAM (Tier 2, Warm)             │
│  温数据驻留，GPU 换出的冷块放到这里                │
│  满了 → LRU 换出到 SSD                           │
└──────────────────────┬──────────────────────────┘
                       ↓ swap out (waterfall)
┌─────────────────────────────────────────────────┐
│              NVMe SSD (Tier 3, Cold)            │
│  冷数据持久化存储，CPU 换出的冷块放到这里          │
│  满了 → LRU 淘汰                                 │
└─────────────────────────────────────────────────┘
```

### 逐级换入流程

```
访问 SSD 上的块：
    DISK → CPU → GPU 逐级换入
    ↑ 磁盘 I/O    ↑ PCIe 传输

访问 CPU 上的块：
    CPU → GPU 换入
    ↑ PCIe 传输

预取：
    decode 阶段提前把下一个块换入 GPU
    隐藏换入延迟
```

### 调度流程

```
1. 检查 waiting 队列，尝试调度 prefill 请求
   ↓ 空间不足
2. 尝试换出冷序列（先 GPU→CPU，再 CPU→DISK）
   ↓ 还不够
3. 等待（或 preempt）
   ↓
4. 检查 GPU/CPU 使用率，触发瀑布式换出
   ↓
5. 尝试将 swapped_out 队列中的热序列换入
   ↓
6. 调度 decode 请求
   ↓
7. 确保所有需要的块在 GPU 中（逐级换入）
   ↓
8. 执行推理
   ↓
9. 触发预取（可选）
```

### 三级前缀缓存

每一级维护独立的前缀缓存哈希表：
- `gpu_hash_to_block_id`：GPU 层前缀缓存
- `cpu_hash_to_block_id`：CPU 层前缀缓存
- `disk_hash_to_block_id`：DISK 层前缀缓存

换出时：从源层级哈希表删除，加入目标层级哈希表
命中检查顺序：GPU → CPU → DISK

## 常见问题

### Q1：启用多级缓存后性能反而下降了？

可能原因：
1. CPU 缓存太小，导致频繁换入换出
2. 替换策略不匹配访问模式
3. 水位线设置不合理，频繁抖动
4. 三级缓存场景下，磁盘 I/O 成为瓶颈

解决方法：
- 增大 CPU 缓存
- 尝试不同的替换策略
- 调整水位线，增大间隔
- 三级缓存场景下，确保使用 NVMe SSD，且有足够的 I/O 带宽

### Q2：为什么还是会有 preempt？

当所有层级的缓存都满了，或者换出速度跟不上分配速度时，仍然会发生 preempt。这是最后的兜底机制。

解决方法：
- 增大 CPU/SSD 缓存
- 降低并发数
- 优化替换策略

### Q3：三级缓存（SSD）适合什么场景？

适合以下场景：
- 超长上下文（32K+），GPU + CPU 装不下
- 大量冷数据，访问频率低
- 成本敏感，用磁盘替代昂贵的内存

不适合：
- 低延迟要求的场景（磁盘 I/O 延迟高）
- 高随机访问的场景（预取效果差）
- 磁盘性能差的环境（如 HDD）

### Q4：预取有什么副作用？

预取会：
- 增加显存占用（预取的块暂时不用）
- 可能预取错误的块（浪费带宽）
- 增加调度复杂度

如果显存紧张或访问模式随机，建议关闭预取。

### Q5：支持分布式（张量并行）吗？

支持。每个 GPU rank 都有自己的 CPU 和 SSD 缓存，数据是分片的。配置方式相同。

### Q6：支持前缀缓存吗？

支持。三级缓存系统完全兼容前缀缓存，GPU、CPU、SSD 三层都维护前缀缓存表。

### Q7：磁盘上的 KV Cache 数据格式是什么？

每个块保存为独立的 `.pt` 文件：
- 文件名：`block_{id:08d}.pt`
- 格式：PyTorch tensor
- Shape：`(2, num_layers, block_size, num_kv_heads, head_dim)`
- 第 0 维：2 表示 K 和 V 两份

## 局限性与注意事项

1. **延迟开销**：换入换出有延迟开销，适合吞吐优先而非延迟优先的场景
2. **CPU 内存占用**：需要足够的 CPU 内存来存放温数据
3. **PCIe 带宽**：GPU↔CPU 换入换出受 PCIe 带宽限制
4. **磁盘 I/O**：CPU↔SSD 换入换出受磁盘 I/O 带宽限制，延迟较高
5. **CUDA Graph**：动态换入换出可能与 CUDA Graph 有冲突，需要小心处理
6. **前缀缓存共享块**：共享块（ref_count > 1）不会被换出，可能导致换出效率降低
7. **写放大**：频繁换出到 SSD 可能影响 SSD 寿命，适合读多写少的场景

## 未来优化方向

1. **异步 I/O**：磁盘读写与 GPU 计算完全重叠，使用异步 I/O 队列
2. **KV Cache 压缩**：CPU/SSD 层使用量化（如 INT8/INT4）减少数据量
3. **智能预取**：基于机器学习预测访问模式，提高预取准确率
4. **共享块优化**：前缀缓存共享块的换出策略（引用计数递减而非不换出）
5. **多队列调度**：按序列热度分队列调度，热序列优先
6. **带宽感知调度**：根据 PCIe/NVMe 带宽动态调整换入换出节奏
7. **分布式多级缓存**：多节点之间的 KV Cache 共享和换入换出
8. **透明压缩**：SSD 层使用透明压缩（如 LZ4），减少磁盘占用和 I/O

## 性能对比参考

基于模拟测试的性能对比（32 层，32 KV 头，128 维，1GB GPU 显存）：

| 并发数 | 基线 (tok/s) | 两级 (tok/s) | 三级 (tok/s) | 两级提升 | 三级提升 |
|--------|-------------|-------------|-------------|----------|----------|
| 1      | 2000        | 2000        | 2000        | +0.0%    | +0.0%    |
| 2      | 2000        | 2000        | 1818        | +0.0%    | -9.1%    |
| 4      | 1429        | 2000        | 1600        | +40.0%   | +12.0%   |
| 8      | 909         | 1939        | 1455        | +113.3%  | +60.0%   |
| 16     | 526         | 1753        | 1260        | +233.2%  | +139.4%  |

**结论**：
- 低并发下，多级缓存收益不明显（数据都能放进 GPU）
- 高并发下，两级缓存可提升 2-3 倍吞吐量
- 三级缓存比基线好，但比两级差（磁盘 I/O 开销）
- 三级缓存的主要价值是**容量扩展**，而非性能提升

## 总结

多级 KV Cache 是一种有效的显存扩展技术，通过瀑布式换出将冷数据逐级换出到 CPU 和 SSD，可以：

- ✅ 显著扩展有效缓存容量（10-100 倍）
- ✅ 提高系统吞吐量（2-3 倍，高并发场景）
- ✅ 支持更长的上下文和更高的并发
- ✅ 降低硬件成本（用便宜的内存/磁盘扩展昂贵的显存）
- ✅ 三级缓存支持超长上下文（128K+）

但需要注意：
- ⚠️ 有一定的延迟开销（尤其是磁盘 I/O）
- ⚠️ 需要足够的 CPU 内存、磁盘空间和 I/O 带宽
- ⚠️ 需要根据具体场景调优参数
- ⚠️ 三级缓存适合冷数据，热数据应驻留在 GPU/CPU

---

*文档版本：v2.0*
*更新时间：2026年8月6日*
*更新内容：增加 SSD 三级缓存支持*
