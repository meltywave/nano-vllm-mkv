# 三级 KV Cache 代码审查报告

**审查日期**：2026-08-06  
**审查范围**：全部三级 KV Cache 相关代码  
**审查文件**：6 个核心文件

---

## 一、总体评价

代码整体架构清晰，三级缓存（GPU + CPU + SSD）的设计思路正确，单元测试覆盖率较高（14/14 通过），端到端集成测试也通过了（63/63）。

但是，**测试通过的主要原因是：测试主要验证了块管理器的逻辑，而调度器和 ModelRunner 的换出/换入逻辑存在严重的设计问题，测试中没有真正触发这些路径。**

**核心问题**：调度器、块管理器、ModelRunner 三者之间的职责划分不清，换出/换入逻辑存在多处不一致，可能导致在真实运行时出现严重 bug。

---

## 二、问题清单

### 🔴 严重问题（必须修复）

---

#### 问题 1：调度器的序列换出没有实际换出块

**位置**：`multi_level_scheduler.py`，`_swap_out_sequence` 方法（第 342-356 行）

**问题描述**：
```python
def _swap_out_sequence(self, seq: Sequence):
    if seq.status == SequenceStatus.SWAPPED_OUT:
        return
    if seq in self.running:
        self.running.remove(seq)
    seq.status = SequenceStatus.SWAPPED_OUT
    self.swapped_out.append(seq)
    self.stats["total_swap_out"] += 1
    # 注意：实际的数据拷贝由 ModelRunner 协调
    # 这里只更新状态
```

这个方法**只是修改了序列的状态和队列归属，完全没有调用块管理器的换出方法**。块还留在 GPU 上，GPU 显存没有释放。

**影响范围**：
- `_try_swap_out_for_new_seq`：为新序列腾出空间时，换出后 GPU 空间没有释放
- `_check_and_swap_out`：水位线触发的换出没有实际效果
- `_swap_out_gpu_to_cpu`：序列级换出没有实际换出块
- `_swap_out_or_preempt`：优先换出的策略失效

**根本原因**：
调度器的换出逻辑和块管理器的换出逻辑是两套独立的体系，没有协调好。
- 块管理器有自己的块级 LRU 换出（`_swap_out_gpu_to_cpu`）
- 调度器有自己的序列级 LRU 换出（`_swap_out_sequence`）
- 两者之间没有调用关系

**修复建议**：
调度器的 `_swap_out_sequence` 应该调用块管理器的换出方法，或者干脆删除调度器自己的换出逻辑，完全依赖块管理器的被动换出（在分配时触发）。

---

#### 问题 2：调度器的 `_swap_out_sequence_to_disk` 什么都没做

**位置**：`multi_level_scheduler.py`，`_swap_out_sequence_to_disk` 方法（第 358-362 行）

**问题描述**：
```python
def _swap_out_sequence_to_disk(self, seq: Sequence):
    # 序列已经在 swapped_out 队列中
    # 只需要更新块的位置状态（由 BlockManager 处理）
    self.stats["total_disk_swap_out"] += 1
```

这个方法**只是增加了一个统计计数，完全没有做任何实际的换出操作**。块还留在 CPU 上，CPU 内存没有释放。

**影响范围**：
- `_swap_out_cpu_to_disk`：CPU→DISK 的换出没有实际效果
- `_check_and_swap_out`：CPU 水位线触发的换出无效

**修复建议**：
调用块管理器的 `_swap_out_cpu_to_disk` 方法，或者删除这个空实现。

---

#### 问题 3：ModelRunner 的 `swap_out_sequence` 绕过块管理器

**位置**：`multi_level_model_runner.py`，`swap_out_sequence` 方法（第 261-300 行）

**问题描述**：
```python
def swap_out_sequence(self, seq: Sequence, block_manager: MultiLevelBlockManager):
    # ...
    for i, block_id in enumerate(seq.block_table):
        if seq.block_locations[i] == BlockLocation.GPU:
            try:
                cpu_block_id = block_manager._allocate_cpu_block()  # 直接调用私有方法！
            except RuntimeError:
                # ...
            gpu_block_ids.append(block_id)
            cpu_block_ids.append(cpu_block_id)
            # 直接更新序列的 block_table
            seq.block_table[i] = cpu_block_id
            seq.block_locations[i] = BlockLocation.CPU
    # 执行数据传输
    if gpu_block_ids:
        self.swap_out_blocks(gpu_block_ids, cpu_block_ids)
```

这个方法存在多个问题：
1. **直接调用私有方法**：`block_manager._allocate_cpu_block()` 违反封装原则
2. **没有更新块元数据**：新分配的 CPU 块的 hash、token_ids、owner_seqs 等都是默认值，没有从 GPU 块复制
3. **没有更新前缀缓存**：前缀缓存哈希表没有更新
4. **没有更新块管理器统计**：`gpu_used_blocks`、`cpu_used_blocks` 等统计虽然在 `_allocate_cpu_block` 中更新了，但是 GPU 块的释放是怎么处理的？

**等等，GPU 块没有被释放！**
- 代码中只分配了 CPU 块，拷贝了数据
- 但是 GPU 块没有被释放！
- GPU 块的 ref_count 还是 1
- GPU 空间没有释放

**这是一个严重的 bug！**

**影响范围**：
- 所有调用 `swap_out_sequence` 的地方
- 换出后 GPU 显存没有释放，等于白换了

**修复建议**：
1. 换出应该由块管理器主导，ModelRunner 只负责数据拷贝
2. 正确的流程应该是：
   - 块管理器：分配 CPU 块、复制元数据、更新前缀缓存、释放 GPU 块、更新序列引用
   - ModelRunner：执行实际的数据传输（GPU→CPU）
3. 两者需要协调好顺序

---

#### 问题 4：磁盘存储没有初始化

**位置**：`llm_engine.py` 和 `multi_level_model_runner.py`

**问题描述**：
块管理器有一个 `init_disk_store` 方法，用于初始化磁盘存储。但是在整个代码库中，**没有任何地方调用这个方法**。

- `LLMEngine.__init__` 中没有调用
- `MultiLevelModelRunner.__init__` 中没有调用
- `MultiLevelScheduler.__init__` 中也没有调用

如果用户启用了磁盘缓存（`enable_disk_cache=True`），块管理器的 `disk_store` 将是 `None`，第一次访问磁盘时就会崩溃。

**修复建议**：
在 LLMEngine 初始化时，创建调度器之后，调用 `self.scheduler.block_manager.init_disk_store(...)`，传入正确的块 shape 和 dtype。

---

#### 问题 5：共享块换入时，其他引用序列的 block_table 没有更新

**位置**：`multi_level_block_manager.py`，`swap_in_cpu_to_gpu` 和 `swap_in_disk_to_cpu` 方法

**问题描述**：
如果一个块被多个序列共享（`ref_count > 1`），当其中一个序列调用 `ensure_blocks_in_gpu` 时：
1. 块被换入到新的层级（新的 block_id）
2. 旧块被释放
3. **但是只有当前序列的 block_table 被更新了**
4. 其他引用这个块的序列的 block_table 还指向旧的 block_id！

这会导致：
- 其他序列访问块时，找到的是已经释放的旧块
- 或者找到的是错误位置的块
- 数据不一致，可能导致崩溃或错误的计算结果

**当前为什么测试没发现？**
因为换出时只换 `ref_count == 1` 的块（第 728 行），所以 CPU 和 DISK 上的块都是单引用的，换入时也只有一个序列引用。

**但是有一个场景会触发**：
- 序列 A 的块被换出到 CPU
- 序列 B 有相同前缀，前缀缓存命中了 CPU 上的块
- 这时候块的 ref_count 变成 2
- 序列 B 调用 `ensure_blocks_in_gpu`，块被换入到 GPU
- 序列 A 的 block_table 还指向 CPU 块 ID
- **bug 触发！**

**修复建议**：
换入时，和换出一样，遍历 `block.owner_seqs`，更新所有引用序列的 block_table 和 block_locations。

---

### 🟡 中等问题（建议修复）

---

#### 问题 6：块管理器和调度器都有 LRU 逻辑，可能不一致

**位置**：`multi_level_block_manager.py` 和 `multi_level_scheduler.py`

**问题描述**：
现在有两套独立的 LRU 机制：
1. **块级 LRU**（块管理器）：基于 `block.last_access_time`
2. **序列级 LRU**（调度器）：基于 `seq.last_access_time`

两者独立工作，可能导致：
- 调度器认为是热序列，但是它的块被块管理器换出了
- 块管理器认为是热块，但是它所属的序列被调度器换出了
- 换出决策不一致，影响性能

**修复建议**：
统一换出策略，建议：
- 方案 A：以块管理器的块级 LRU 为主，调度器不主动换出序列，只负责调度
- 方案 B：以调度器的序列级 LRU 为主，块管理器不主动换出块，只在调度器指令下换出

---

#### 问题 7：块管理器和 ModelRunner 职责划分不清

**位置**：整体架构

**问题描述**：
换出/换入的逻辑分散在三个地方：
1. 块管理器：块元数据管理、LRU 选择、引用计数
2. 调度器：序列状态管理、队列调度
3. ModelRunner：实际数据传输

但是现在的实现中：
- 块管理器有自己的换出逻辑（`_swap_out_gpu_to_cpu`）
- 调度器有自己的换出逻辑（`_swap_out_sequence`）
- ModelRunner 也有自己的换出逻辑（`swap_out_sequence`）

三者之间没有清晰的调用关系，容易导致状态不一致。

**修复建议**：
明确职责划分：
- **调度器**：决定哪个序列换出/换入（策略层）
- **块管理器**：管理块的元数据、位置、引用计数（元数据层）
- **ModelRunner**：执行实际的数据传输（数据层）

调用关系：
```
调度器 → 块管理器（换出/换入块）
       → ModelRunner（执行数据传输）
```

---

#### 问题 8：`can_allocate` 中前缀缓存命中的空闲块的 `num_new_blocks` 计算

**位置**：`multi_level_block_manager.py`，`can_allocate` 方法（第 456-457 行）

**问题描述**：
```python
if block_id in self.gpu_used_block_ids:
    num_new_blocks -= 1
```

这里只减去了 `used` 状态的前缀缓存块，但是 `free` 状态的前缀缓存块也可以复用啊！

空闲的前缀缓存块（ref_count=0，但是内容还在）也可以被复用，不需要分配新块。

所以 `num_new_blocks` 应该减去所有命中的块，不管它当前是 used 还是 free。

**影响**：
- `can_allocate` 可能错误地返回 -1（空间不足）
- 但实际上有空闲的前缀缓存块可以复用

**修复建议**：
```python
# 不管是 used 还是 free，只要命中了就可以复用
num_cached_blocks += 1
num_new_blocks -= 1
```

---

#### 问题 9：子进程的 ModelRunner 没有设置 block_manager

**位置**：`llm_engine.py`，第 37-56 行

**问题描述**：
```python
# 子进程
for i in range(1, config.tensor_parallel_size):
    event = ctx.Event()
    process = ctx.Process(target=ModelRunnerClass, args=(config, i, event))
    process.start()

# 主进程
self.model_runner = ModelRunnerClass(config, 0, self.events)

# 只有主进程设置了 block_manager
if hasattr(self.model_runner, 'set_block_manager'):
    self.model_runner.set_block_manager(self.scheduler.block_manager)
```

子进程的 ModelRunner 没有设置 block_manager，所以子进程的 `run` 方法中，`block_manager_ref` 是 None，不会执行多级缓存逻辑。

对于单 GPU（`tensor_parallel_size=1`），没有问题。
对于多 GPU，子进程的 KV Cache 换入换出可能有问题。

**修复建议**：
通过 IPC 机制将块管理器的状态同步给子进程，或者每个进程维护自己的块管理器（但需要同步决策）。

---

#### 问题 10：磁盘 I/O 统计没有更新

**位置**：`multi_level_block_manager.py`，`MultiLevelCacheStats` 类

**问题描述**：
`MultiLevelCacheStats` 中有这些字段：
```python
self.disk_read_bytes = 0
self.disk_write_bytes = 0
self.disk_read_time = 0.0
self.disk_write_time = 0.0
```

但是在整个代码库中，**这些字段从来没有被更新过**。
- `DiskBlockStore.write_block` 返回了字节数，但是没有更新统计
- `DiskBlockStore.read_block` 没有统计时间和字节数
- ModelRunner 的 `swap_out_cpu_to_disk` 和 `swap_in_disk_to_cpu` 有自己的统计，但是没有同步到块管理器的统计

**修复建议**：
在磁盘读写的地方更新这些统计字段，或者统一由 ModelRunner 统计。

---

### 🟢 轻微问题（可选优化）

---

#### 问题 11：释放块时没有从前缀缓存哈希表中移除

**位置**：`multi_level_block_manager.py`，`_deallocate_gpu_block` 和 `_deallocate_cpu_block`

**问题描述**：
释放 GPU/CPU 块时，没有从前缀缓存哈希表中删除对应的条目。
只有 DISK 块释放时做了这个检查。

不过，`_allocate_gpu_block` 和 `_allocate_cpu_block` 在分配时会检查并删除旧的哈希条目，所以功能上没问题。

只是不太干净，哈希表中可能存在一些已经释放的块的映射。

**修复建议**：
在 `_deallocate_gpu_block` 和 `_deallocate_cpu_block` 中，也加上前缀缓存的清理。

---

#### 问题 12：prefill 阶段两次调用 `can_allocate`

**位置**：`multi_level_scheduler.py`，`_schedule_multilevel` 方法（第 168 行和第 181 行）

**问题描述**：
```python
# 第一次调用
num_cached_blocks = self.block_manager.can_allocate(seq)
if num_cached_blocks == -1:
    if not self._try_swap_out_for_new_seq(seq):
        break
# ...
# 第二次调用
num_cached_blocks = self.block_manager.can_allocate(seq)
if num_cached_blocks == -1:
    break
self.block_manager.allocate(seq, num_cached_blocks)
```

两次调用 `can_allocate`，中间可能换出了一些序列，状态变了。
但是两次调用有点冗余，可以优化。

**修复建议**：
换出后直接尝试分配，失败再处理。

---

#### 问题 13：`get_block_heat_map` 中热度计算公式不太合理

**位置**：`multi_level_block_manager.py`，第 1001 行

**问题描述**：
```python
heat = block.access_count / (idle_time + 1)
```

这个公式有点奇怪：
- `access_count` 是总访问次数
- `idle_time` 是距离上次访问的时间
- 两者相除的物理意义不明确

热度应该是访问频率（单位时间内的访问次数），但这个公式不是。

**修复建议**：
改用更合理的热度公式，比如：
- 滑动窗口内的访问次数
- 或者指数移动平均的访问频率

---

## 三、最关键的问题总结

如果只能修复 3 个问题，我建议优先修复这 3 个：

### 1. 调度器的换出没有实际效果
**严重程度**：🔴 严重  
**影响**：换出策略完全失效，GPU 显存不会释放  
**修复难度**：中等

### 2. ModelRunner 的换出绕过块管理器
**严重程度**：🔴 严重  
**影响**：状态不一致，可能导致数据错误或崩溃  
**修复难度**：中等

### 3. 磁盘存储没有初始化
**严重程度**：🔴 严重  
**影响**：启用磁盘缓存就会崩溃  
**修复难度**：简单

---

## 四、架构改进建议

### 建议 1：统一换出/换入的控制权

**现状**：三个组件都有自己的换出逻辑，混乱不堪。

**建议**：
- **块管理器是唯一的块状态管理者**
- 调度器只负责序列调度，不直接操作块
- ModelRunner 只负责数据传输，不管理块状态

**调用关系**：
```
调度器（决定哪个序列换出）
    ↓
块管理器（执行块的换出/换入，管理元数据）
    ↓
ModelRunner（执行实际的数据传输）
```

### 建议 2：明确换出/换入的触发时机

**现状**：
- 块管理器：分配时被动触发换出
- 调度器：水位线主动触发换出
- 两者重复且不一致

**建议**：
选择一种触发方式，建议：
- **被动触发为主**：分配时如果空间不够，触发换出
- **主动触发为辅**：空闲时可以主动换出一些冷块，预留空间

### 建议 3：增加集成测试的覆盖范围

**现状**：
- 单元测试主要测块管理器
- 集成测试主要测块管理器 + 模拟推理
- 调度器的换出/换入逻辑没有真正测试

**建议**：
- 增加调度器级别的集成测试
- 测试完整的调度 → 换出 → 换入 → 推理流程
- 确保调度器、块管理器、ModelRunner 三者协调工作

---

## 五、文件审查总结

| 文件 | 严重问题 | 中等问题 | 轻微问题 | 总体评价 |
|------|----------|----------|----------|----------|
| `multi_level_block_manager.py` | 1 | 2 | 2 | 核心逻辑较完善，边界情况需注意 |
| `multi_level_scheduler.py` | 2 | 1 | 1 | 换出逻辑基本是空壳，需要重写 |
| `multi_level_model_runner.py` | 1 | 1 | 0 | 职责不清，需要重新设计 |
| `sequence.py` | 0 | 0 | 0 | 扩展正确，没有问题 |
| `config.py` | 0 | 0 | 0 | 配置完整，验证充分 |
| `llm_engine.py` | 1 | 1 | 0 | 集成有遗漏，需要补充 |

---

## 六、测试覆盖分析

**为什么测试都通过了，但还有这么多问题？**

因为测试主要验证的是**块管理器的逻辑正确性**，而没有真正触发**调度器和 ModelRunner 的换出/换入路径**。

具体来说：
1. 单元测试（14/14）：直接调用块管理器的方法，不经过调度器
2. 集成测试（63/63）：直接调用块管理器的方法，用模拟推理引擎
3. 调度器的 `_swap_out_sequence` 从来没有在测试中被验证过效果
4. ModelRunner 的 `swap_out_sequence` 从来没有在测试中被调用过

**建议增加的测试**：
1. 调度器级别的换出测试：验证换出后 GPU 块数减少
2. 调度器级别的换入测试：验证换入后 GPU 块数增加
3. 完整流程测试：调度 → 换出 → 换入 → 推理，验证数据一致性
4. 共享块换入测试：验证多序列共享块时的换入行为

---

**报告生成时间**：2026-08-06  
**审查人**：代码审查助手
