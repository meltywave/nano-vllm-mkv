# nano-vLLM-MKV

支持 GPU、CPU、SSD 和远端设备的多级 KV Cache 原型

## 实验参数

| 数值 | 缓存层级 |
| ---: | --- |
| `1` | GPU |
| `2` | GPU + CPU |
| `3` | GPU + CPU + SSD |
| `4` | GPU + CPU + SSD + Remote |

主要配置文件：

- `experiments/config/cache_levels.yaml`：各层容量、水位线和远端地址
- `experiments/config/benchmark.yaml`：预热和正式测量次数
- `experiments/workloads/*.json`：输入长度、输出长度、并发数及 prompt 文件
- `experiments/prompts/*.jsonl`：固定 prompt 样本

## 实验命令

```powershell
python bench.py --engine <1|2|3|4> --model models/Qwen3-0.6B --workload experiments/workloads/smoke.json
```

GPU baseline：

```powershell
python bench.py --engine 1 --model models/Qwen3-0.6B --workload experiments/workloads/smoke.json
```

GPU + CPU：

```powershell
python bench.py --engine 2 --model models/Qwen3-0.6B --workload experiments/workloads/smoke.json
```

GPU + CPU + SSD：

```powershell
python bench.py --engine 3 --model models/Qwen3-0.6B --workload experiments/workloads/smoke.json
```

GPU + CPU + SSD + Remote：

```powershell
# 终端 1：启动远端缓存服务
python -m nanovllm.engine.remote_cache_server --host 127.0.0.1 --port 19090 --max-gb 64

# 终端 2：运行实验
python bench.py --engine 4 --model models/Qwen3-0.6B --workload experiments/workloads/smoke.json
```

## 实验流程

1. 在 `cache_levels.yaml` 中设置各层容量和 high/low 水位线
2. 在 workload 文件中固定 prompt、输入长度、输出长度和并发数
3. 先运行 engine 1，得到原生 Nano-vLLM baseline
4. 依次运行 engine 2、3、4，完成层级消融实验
5. 主表使用 engine 1 和实验前已锁定的 Nano-vLLM-mkv 配置；vLLM 在仓库外使用相同 workload 运行
6. 在 result/ 中检查不同 engine 的输出 hash、吞吐、峰值显存及迁移统计

## 实验结果

```text
results/<workload>/engine-<level>/<model>/run-*/result.json
```

`result.json` 包含延迟、TTFT、吞吐、峰值显存、Swap 次数、各层迁移量和输出 hash

汇总全部结果，生成results/summary.csv：

```powershell
python -m experiments.aggregate
```

## 基础检查

```powershell
python -m unittest discover -s tests -v
python -m compileall -q nanovllm experiments tests bench.py
```