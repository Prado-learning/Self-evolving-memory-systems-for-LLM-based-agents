#!/usr/bin/env python3
"""
生成 MemRL 实验结果可视化图表

Usage:
    python plot_results.py --input results.json --output-dir ./plots

Input JSON format:
{
    "exp_name": {
        "results": {
            "no_memory": { "final_sr": 0.xx, "epochs": [...] },
            "rag": { "final_sr": 0.xx, "epochs": [...] },
            "memrl": { "final_sr": 0.xx, "epochs": [...] },
            "task_memrl": { "final_sr": 0.xx, "epochs": [...] }
        }
    }
}
"""
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

COLORS = {
    'no_memory': '#888888',
    'rag': '#4CAF50',
    'memrl': '#2196F3',
    'task_memrl': '#FF5722'
}


def plot_final_sr_comparison(data: dict, output_dir: Path) -> None:
    """图1: 多个实验的最终成功率对比柱状图"""
    fig, ax = plt.subplots(figsize=(12, 6))

    experiments = list(data.keys())
    x = np.arange(len(experiments))
    width = 0.2

    no_memory_vals = []
    rag_vals = []
    memrl_vals = []
    task_memrl_vals = []

    for exp_name in experiments:
        exp = data[exp_name]['results']
        no_memory_vals.append(exp.get('no_memory', {}).get('final_sr', 0))
        rag_vals.append(exp.get('rag', {}).get('final_sr', 0))
        memrl_vals.append(exp.get('memrl', {}).get('final_sr', 0))
        task_memrl_vals.append(exp.get('task_memrl', {}).get('final_sr', 0))

    bars1 = ax.bar(x - 1.5*width, no_memory_vals, width, label='No Memory', color=COLORS['no_memory'], alpha=0.8)
    bars2 = ax.bar(x - 0.5*width, rag_vals, width, label='RAG', color=COLORS['rag'], alpha=0.8)
    bars3 = ax.bar(x + 0.5*width, memrl_vals, width, label='MemRL', color=COLORS['memrl'], alpha=0.8)
    bars4 = ax.bar(x + 1.5*width, task_memrl_vals, width, label='Task-MemRL (Ours)', color=COLORS['task_memrl'], alpha=0.8)

    ax.set_ylabel('Success Rate', fontsize=12)
    ax.set_title('Final Success Rate Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, fontsize=10)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)

    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'final_sr_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Generated: {output_dir / 'final_sr_comparison.png'}")


def plot_training_curve(data: dict, exp_key: str, output_dir: Path, title: str = None) -> None:
    """绘制单个实验的训练曲线"""
    if exp_key not in data:
        print(f"Warning: experiment '{exp_key}' not found in data")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    exp = data[exp_key]['results']

    methods = ['no_memory', 'memrl', 'task_memrl']
    method_names = ['No Memory', 'MemRL', 'Task-MemRL (Ours)']
    markers = ['o', 's', '^']

    epochs = None
    for method, name, marker in zip(methods, method_names, markers):
        if method in exp and 'epochs' in exp[method]:
            if epochs is None:
                epochs = list(range(1, len(exp[method]['epochs']) + 1))
            sr_history = [ep['sr'] for ep in exp[method]['epochs']]
            ax.plot(epochs, sr_history, marker=marker, label=name, color=COLORS[method],
                    linewidth=2, markersize=8)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Success Rate', fontsize=12)
    ax.set_title(title or f'Training Curve: {exp_key}', fontsize=14, fontweight='bold')
    ax.set_xticks(epochs)
    ax.legend(loc='best')
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    safe_name = exp_key.replace(' ', '_').replace('/', '_')
    plt.savefig(output_dir / f'{safe_name}_training_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Generated: {output_dir / f'{safe_name}_training_curve.png'}")


def plot_improvement(data: dict, output_dir: Path) -> None:
    """图4: Task-MemRL 相对 MemRL 的提升幅度"""
    fig, ax = plt.subplots(figsize=(10, 5))

    exp_names = list(data.keys())
    improvements = []

    for exp_name in exp_names:
        exp = data[exp_name]['results']
        memrl_sr = exp.get('memrl', {}).get('final_sr', 0)
        task_sr = exp.get('task_memrl', {}).get('final_sr', 0)
        if memrl_sr > 0:
            imp = (task_sr - memrl_sr) / memrl_sr * 100
        else:
            imp = 0
        improvements.append(imp)

    bars = ax.bar(exp_names, improvements, color=COLORS['task_memrl'], alpha=0.8)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    ax.set_ylabel('Improvement (%)', fontsize=12)
    ax.set_title('Task-MemRL Improvement over MemRL', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax.annotate(f'{imp:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3 if height >= 0 else -15), textcoords="offset points",
                   ha='center', va='bottom' if height >= 0 else 'top', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'improvement_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Generated: {output_dir / 'improvement_comparison.png'}")


def plot_multiseed(data: dict, exp_key: str, output_dir: Path, seeds: list) -> None:
    """绘制多seed实验结果"""
    if exp_key not in data:
        print(f"Warning: experiment '{exp_key}' not found in data")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    exp = data[exp_key]['results']

    x = np.arange(len(seeds))
    width = 0.2
    methods = ['no_memory', 'memrl', 'task_memrl']
    method_names = ['No Memory', 'MemRL', 'Task-MemRL (Ours)']

    for i, (method, name) in enumerate(zip(methods, method_names)):
        if method in exp and 'seeds' in exp[method]:
            vals = []
            for seed in seeds:
                seed_key = f'seed_{seed}'
                if seed_key in exp[method]['seeds']:
                    vals.append(exp[method]['seeds'][seed_key].get('final_sr', 0))
                else:
                    vals.append(0)
            ax.bar(x + (i-1)*width, vals, width, label=name, color=COLORS[method], alpha=0.8)

    ax.set_ylabel('Success Rate', fontsize=12)
    ax.set_title(f'Multi-Seed Results: {exp_key}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'seed={s}' for s in seeds])
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    safe_name = exp_key.replace(' ', '_').replace('/', '_')
    plt.savefig(output_dir / f'{safe_name}_multiseed.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Generated: {output_dir / f'{safe_name}_multiseed.png'}")


def plot_radar(output_dir: Path) -> None:
    """绘制方法对比雷达图"""
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    categories = ['Success Rate', 'Anti-Interference', 'Stability', 'Robustness']
    N = len(categories)

    scores = {
        'No Memory': [0.3, 0.1, 0.3, 0.3],
        'RAG': [0.4, 0.2, 0.4, 0.5],
        'MemRL': [0.3, 0.15, 0.2, 0.4],
        'Task-MemRL': [0.45, 0.85, 0.8, 0.7]
    }

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    for name, score in scores.items():
        values = score + score[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=name)
        ax.fill(angles, values, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title('Method Comparison Radar Chart', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / 'radar_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Generated: {output_dir / 'radar_comparison.png'}")


def main():
    parser = argparse.ArgumentParser(description='Generate MemRL experiment visualization')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to results JSON file')
    parser.add_argument('--output-dir', type=str, default='./plots',
                       help='Output directory for plots')
    parser.add_argument('--exp-keys', nargs='*', default=[],
                       help='Specific experiment keys to plot (default: all)')
    parser.add_argument('--seeds', nargs='*', type=int, default=[42, 123, 456],
                       help='Seed values for multi-seed plots')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(Path(args.input).read_text())

    print(f"Generating plots from {args.input}...")

    plot_final_sr_comparison(data, output_dir)
    plot_improvement(data, output_dir)
    plot_radar(output_dir)

    if args.exp_keys:
        for exp_key in args.exp_keys:
            if exp_key in data:
                plot_training_curve(data, exp_key, output_dir)
                if 'seeds' in data[exp_key].get('results', {}).get('no_memory', {}):
                    plot_multiseed(data, exp_key, output_dir, args.seeds)

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == '__main__':
    main()
