"""
Nano-vLLM 多级 KV Cache 性能可视化脚本

生成性能对比图表：
1. 不同并发数下的 Decode 吞吐量对比
2. 不同序列长度下的 Prefill 吞吐量对比
3. 三级缓存容量对比
4. 换出次数对比
"""

import json
import os
import argparse
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端，不需要 GUI
    import matplotlib.pyplot as plt
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️  未安装 matplotlib，将跳过图表生成")
    print("    安装命令: pip install matplotlib")


def load_results(json_path: str) -> list:
    """加载基准测试结果"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_decode_throughput_vs_concurrency(results: list, output_dir: str):
    """绘制不同并发数下的 Decode 吞吐量对比图"""
    if not HAS_MATPLOTLIB:
        return

    # 筛选 seq_len=1024 的数据
    data = defaultdict(lambda: defaultdict(float))
    for r in results:
        if r['seq_len'] == 1024:
            data[r['mode']][r['num_seqs']] = r['decode_throughput']

    fig, ax = plt.subplots(figsize=(10, 6))

    modes = ['baseline', 'two_level', 'three_level']
    labels = ['基线 (仅 GPU)', '两级 (GPU+CPU)', '三级 (GPU+CPU+SSD)']
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
    markers = ['o', 's', '^']

    for mode, label, color, marker in zip(modes, labels, colors, markers):
        if mode not in data:
            continue
        x = sorted(data[mode].keys())
        y = [data[mode][n] for n in x]
        ax.plot(x, y, marker=marker, label=label, color=color, linewidth=2, markersize=8)

    ax.set_xlabel('并发数', fontsize=12)
    ax.set_ylabel('Decode 吞吐量 (tokens/s)', fontsize=12)
    ax.set_title('不同并发数下的 Decode 吞吐量对比 (seq_len=1024)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([1, 2, 4, 8, 16])

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'decode_throughput_vs_concurrency.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 已生成: {output_path}")


def plot_prefill_throughput_vs_seqlen(results: list, output_dir: str):
    """绘制不同序列长度下的 Prefill 吞吐量对比图"""
    if not HAS_MATPLOTLIB:
        return

    # 筛选 num_seqs=4 的数据
    data = defaultdict(lambda: defaultdict(float))
    for r in results:
        if r['num_seqs'] == 4:
            data[r['mode']][r['seq_len']] = r['prefill_throughput']

    fig, ax = plt.subplots(figsize=(10, 6))

    modes = ['baseline', 'two_level', 'three_level']
    labels = ['基线 (仅 GPU)', '两级 (GPU+CPU)', '三级 (GPU+CPU+SSD)']
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
    markers = ['o', 's', '^']

    for mode, label, color, marker in zip(modes, labels, colors, markers):
        if mode not in data:
            continue
        x = sorted(data[mode].keys())
        y = [data[mode][n] for n in x]
        ax.plot(x, y, marker=marker, label=label, color=color, linewidth=2, markersize=8)

    ax.set_xlabel('序列长度 (tokens)', fontsize=12)
    ax.set_ylabel('Prefill 吞吐量 (tokens/s)', fontsize=12)
    ax.set_title('不同序列长度下的 Prefill 吞吐量对比 (num_seqs=4)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([256, 512, 1024, 2048])

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'prefill_throughput_vs_seqlen.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 已生成: {output_path}")


def plot_memory_capacity(results: list, output_dir: str):
    """绘制三级缓存容量对比图"""
    if not HAS_MATPLOTLIB:
        return

    # 筛选 num_seqs=8, seq_len=2048 的数据
    data = {}
    for r in results:
        if r['num_seqs'] == 8 and r['seq_len'] == 2048:
            data[r['mode']] = r

    fig, ax = plt.subplots(figsize=(10, 6))

    modes = ['baseline', 'two_level', 'three_level']
    labels = ['基线 (仅 GPU)', '两级 (GPU+CPU)', '三级 (GPU+CPU+SSD)']

    gpu_mem = [data[m]['gpu_memory_peak'] for m in modes if m in data]
    cpu_mem = [data[m]['cpu_memory_peak'] for m in modes if m in data]
    disk_mem = [data[m]['disk_usage_peak'] for m in modes if m in data]

    x = range(len(labels))
    width = 0.6

    ax.bar(x, gpu_mem, width, label='GPU 显存', color='#ff6b6b')
    ax.bar(x, cpu_mem, width, bottom=gpu_mem, label='CPU 内存', color='#4ecdc4')
    ax.bar(x, disk_mem, width, [g + c for g, c in zip(gpu_mem, cpu_mem)],
           label='SSD 磁盘', color='#45b7d1')

    ax.set_xlabel('缓存模式', fontsize=12)
    ax.set_ylabel('KV Cache 容量 (GB)', fontsize=12)
    ax.set_title('三级缓存容量对比 (num_seqs=8, seq_len=2048)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # 在柱子上标注总容量
    for i, (g, c, d) in enumerate(zip(gpu_mem, cpu_mem, disk_mem)):
        total = g + c + d
        ax.text(i, total + 0.1, f'{total:.2f} GB', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'memory_capacity_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 已生成: {output_path}")


def plot_swap_count(results: list, output_dir: str):
    """绘制换出次数对比图"""
    if not HAS_MATPLOTLIB:
        return

    # 按并发数统计换出次数（seq_len=1024）
    data = defaultdict(lambda: defaultdict(int))
    for r in results:
        if r['seq_len'] == 1024:
            data[r['mode']][r['num_seqs']] = r['total_swap_out']

    fig, ax = plt.subplots(figsize=(10, 6))

    modes = ['two_level', 'three_level']
    labels = ['两级 (GPU+CPU)', '三级 (GPU+CPU+SSD)']
    colors = ['#4ecdc4', '#45b7d1']

    x = sorted(set().union(*[data[m].keys() for m in modes if m in data]))
    width = 0.35

    for i, (mode, label, color) in enumerate(zip(modes, labels, colors)):
        if mode not in data:
            continue
        y = [data[mode].get(n, 0) for n in x]
        ax.bar([xi + (i - 0.5) * width for xi in range(len(x))], y, width,
               label=label, color=color)

    ax.set_xlabel('并发数', fontsize=12)
    ax.set_ylabel('换出次数 (块)', fontsize=12)
    ax.set_title('不同并发数下的换出次数对比 (seq_len=1024)', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(x)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'swap_count_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 已生成: {output_path}")


def plot_throughput_improvement(results: list, output_dir: str):
    """绘制吞吐量提升百分比图"""
    if not HAS_MATPLOTLIB:
        return

    # 计算相对于基线的提升百分比（seq_len=1024）
    baseline_data = {}
    two_level_data = {}
    three_level_data = {}

    for r in results:
        if r['seq_len'] == 1024:
            if r['mode'] == 'baseline':
                baseline_data[r['num_seqs']] = r['decode_throughput']
            elif r['mode'] == 'two_level':
                two_level_data[r['num_seqs']] = r['decode_throughput']
            elif r['mode'] == 'three_level':
                three_level_data[r['num_seqs']] = r['decode_throughput']

    x = sorted(baseline_data.keys())
    two_level_improvement = [(two_level_data[n] - baseline_data[n]) / baseline_data[n] * 100
                             for n in x]
    three_level_improvement = [(three_level_data[n] - baseline_data[n]) / baseline_data[n] * 100
                               for n in x]

    fig, ax = plt.subplots(figsize=(10, 6))

    width = 0.35
    ax.bar([i - width/2 for i in range(len(x))], two_level_improvement, width,
           label='两级缓存', color='#4ecdc4')
    ax.bar([i + width/2 for i in range(len(x))], three_level_improvement, width,
           label='三级缓存', color='#45b7d1')

    ax.set_xlabel('并发数', fontsize=12)
    ax.set_ylabel('吞吐量提升 (%)', fontsize=12)
    ax.set_title('相对于基线的吞吐量提升 (Decode, seq_len=1024)', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(x)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linewidth=0.5)

    # 在柱子上标注百分比
    for i, (t1, t2) in enumerate(zip(two_level_improvement, three_level_improvement)):
        ax.text(i - width/2, t1 + 5, f'{t1:+.1f}%', ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, t2 + 5, f'{t2:+.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'throughput_improvement.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 已生成: {output_path}")


def plot_architecture_diagram(output_dir: str):
    """绘制三级缓存架构图"""
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # 绘制三个层级
    levels = [
        {'name': 'GPU HBM', 'subtitle': '(Tier 1, 热数据)', 'y': 0.8, 'color': '#ff6b6b',
         'features': ['最低延迟', '最高带宽', '容量最小', '成本最高']},
        {'name': 'CPU DRAM', 'subtitle': '(Tier 2, 温数据)', 'y': 0.5, 'color': '#4ecdc4',
         'features': ['中等延迟', '中等带宽', '中等容量', '中等成本']},
        {'name': 'NVMe SSD', 'subtitle': '(Tier 3, 冷数据)', 'y': 0.2, 'color': '#45b7d1',
         'features': ['最高延迟', '最低带宽', '容量最大', '成本最低']},
    ]

    for level in levels:
        # 绘制矩形
        rect = plt.Rectangle((0.2, level['y'] - 0.1), 0.6, 0.15,
                            facecolor=level['color'], alpha=0.3,
                            edgecolor=level['color'], linewidth=2)
        ax.add_patch(rect)

        # 标题
        ax.text(0.5, level['y'] + 0.02, level['name'],
                ha='center', va='center', fontsize=16, fontweight='bold',
                color=level['color'])
        ax.text(0.5, level['y'] - 0.05, level['subtitle'],
                ha='center', va='center', fontsize=12, color='#555')

        # 特性列表
        for i, feat in enumerate(level['features']):
            ax.text(0.85, level['y'] + 0.04 - i * 0.03, f'• {feat}',
                    ha='left', va='center', fontsize=10, color='#333')

    # 绘制瀑布式换出箭头
    ax.annotate('', xy=(0.5, 0.7), xytext=(0.5, 0.78),
                arrowprops=dict(arrowstyle='->', color='#ff6b6b', lw=2))
    ax.text(0.55, 0.74, 'LRU 换出', fontsize=10, color='#ff6b6b')

    ax.annotate('', xy=(0.5, 0.4), xytext=(0.5, 0.48),
                arrowprops=dict(arrowstyle='->', color='#4ecdc4', lw=2))
    ax.text(0.55, 0.44, 'LRU 换出', fontsize=10, color='#4ecdc4')

    # 绘制换入箭头
    ax.annotate('', xy=(0.3, 0.78), xytext=(0.3, 0.7),
                arrowprops=dict(arrowstyle='->', color='#4ecdc4', lw=2, linestyle='--'))
    ax.text(0.15, 0.74, '换入', fontsize=10, color='#4ecdc4')

    ax.annotate('', xy=(0.3, 0.48), xytext=(0.3, 0.4),
                arrowprops=dict(arrowstyle='->', color='#45b7d1', lw=2, linestyle='--'))
    ax.text(0.15, 0.44, '换入', fontsize=10, color='#45b7d1')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('三级 KV Cache 瀑布式架构', fontsize=18, fontweight='bold', pad=20)
    ax.axis('off')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'architecture_diagram.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 已生成: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="多级 KV Cache 性能可视化")
    parser.add_argument("--input", type=str, default="./bench_results/benchmark_results.json",
                        help="基准测试结果 JSON 文件")
    parser.add_argument("--output-dir", type=str, default="./bench_results",
                        help="图表输出目录")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ 找不到结果文件: {args.input}")
        print("    请先运行 bench_multilevel.py 生成测试结果")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Nano-vLLM 多级 KV Cache 性能可视化")
    print("=" * 60)
    print()

    results = load_results(args.input)
    print(f"✅ 加载了 {len(results)} 组测试结果")
    print()

    # 生成各种图表
    print("正在生成图表...")
    print()

    plot_decode_throughput_vs_concurrency(results, args.output_dir)
    plot_prefill_throughput_vs_seqlen(results, args.output_dir)
    plot_memory_capacity(results, args.output_dir)
    plot_swap_count(results, args.output_dir)
    plot_throughput_improvement(results, args.output_dir)
    plot_architecture_diagram(args.output_dir)

    print()
    print("🎉 所有图表生成完成！")
    print(f"   输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()
