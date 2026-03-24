#!/usr/bin/env python3
"""
MemRL 实验结果可视化 - 美化版

Usage:
    python plot_final_results.py --input results.json --output-dir ./plots

Input JSON format:
{
    "exp_key": {
        "results": {
            "no_memory": { "final_sr": 0.xx, "epochs": [...] },
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

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.6
plt.rcParams['legend.framealpha'] = 0.9
plt.rcParams['legend.edgecolor'] = 'none'

COLORS = {
    'no_memory': '#7F7F7F',
    'memrl':     '#4A90D9',
    'task_memrl': '#E07B39',
}

METHODS = ['no_memory', 'memrl', 'task_memrl']
METHOD_LABELS = ['No Memory', 'MemRL', 'Task-MemRL (Ours)']
MARKERS = ['o', 's', '^']


def load_data(input_path: str) -> dict:
    """Load experiment results from JSON file."""
    with open(input_path, 'r') as f:
        return json.load(f)


def get_final_sr(data: dict, exp_name: str, method: str) -> float:
    """Get final success rate for a method in an experiment."""
    if exp_name not in data:
        return 0.0
    exp = data[exp_name]['results']
    if method not in exp:
        return 0.0
    m = exp[method]
    if 'seeds' in m:
        seeds = list(m['seeds'].keys())
        return np.mean([m['seeds'][s].get('final_sr', 0) for s in seeds])
    return m.get('final_sr', 0.0)


def plot_comprehensive_results(data: dict, output_dir: Path, exp_labels: list = None) -> None:
    """综合对比图 (2×2): 最终成功率 + 提升幅度 + 两条训练曲线"""
    experiments = list(data.keys())
    if exp_labels is None:
        exp_labels = [f'exp{i+1}' for i in range(len(experiments))]

    fig = plt.figure(figsize=(17, 13))
    gs = plt.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)
    fig.suptitle('Memory RL Experiment Results', fontsize=18, fontweight='bold', y=0.98)

    # (a) 最终成功率柱状图
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(exp_labels))
    bw = 0.24
    vals_no = [get_final_sr(data, n, 'no_memory') for n in experiments]
    vals_mem = [get_final_sr(data, n, 'memrl') for n in experiments]
    vals_task = [get_final_sr(data, n, 'task_memrl') for n in experiments]

    bars_no  = ax1.bar(x - bw, vals_no,  bw, label=METHOD_LABELS[0], color=COLORS['no_memory'],  edgecolor='white', linewidth=1.2)
    bars_mem = ax1.bar(x,      vals_mem, bw, label=METHOD_LABELS[1], color=COLORS['memrl'],      edgecolor='white', linewidth=1.2)
    bars_task= ax1.bar(x + bw, vals_task, bw, label=METHOD_LABELS[2], color=COLORS['task_memrl'], edgecolor='white', linewidth=1.2)

    ax1.set_ylabel('Success Rate', fontsize=12, fontweight='medium')
    ax1.set_title('(a) Final Success Rate', fontsize=13, fontweight='bold', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(exp_labels, fontsize=10)
    ax1.legend(loc='upper right', fontsize=9, framealpha=0.85)
    ax1.set_ylim(0, 1.0)
    ax1.set_xlim(-0.6, len(exp_labels) - 0.4)

    for bars in [bars_no, bars_mem, bars_task]:
        for bar in bars:
            h = bar.get_height()
            ax1.annotate(f'{h:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='bold',
                        color='#333333')

    # (b) Task-MemRL 提升幅度
    ax2 = fig.add_subplot(gs[0, 1])
    imps = []
    for exp_name in experiments:
        m_sr = get_final_sr(data, exp_name, 'memrl')
        t_sr = get_final_sr(data, exp_name, 'task_memrl')
        imps.append((t_sr - m_sr) / m_sr * 100 if m_sr > 0 else 0)

    bars_imp = ax2.bar(exp_labels, imps, color=COLORS['task_memrl'], edgecolor='white', linewidth=1.2, alpha=0.88)
    ax2.axhline(50, color='#4CAF50', linestyle='--', linewidth=1.4, alpha=0.7, label='50% baseline')
    ax2.set_ylabel('Improvement over MemRL (%)', fontsize=12, fontweight='medium')
    ax2.set_title('(b) Task-MemRL Improvement', fontsize=13, fontweight='bold', pad=10)
    ax2.set_ylim(0, max(115, max(imps) * 1.2))
    ax2.legend(loc='upper right', fontsize=9)

    for bar, imp in zip(bars_imp, imps):
        ax2.annotate(f'+{imp:.0f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, imp),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color='#333333')

    # (c) & (d) 训练曲线 - 选择两个有代表性的实验
    for idx, exp_key in enumerate(experiments[:2]):
        ax = fig.add_subplot(gs[1, idx])
        exp = data[exp_key]['results']

        epochs = None
        for method, lbl, mkr in zip(METHODS, METHOD_LABELS, MARKERS):
            if method in exp and 'epochs' in exp[method]:
                if epochs is None:
                    epochs = list(range(1, len(exp[method]['epochs']) + 1))
                sr = [ep['sr'] for ep in exp[method]['epochs']]
                ax.plot(epochs, sr, marker=mkr, label=lbl, color=COLORS[method],
                       linewidth=2.2, markersize=8, markeredgecolor='white', markeredgewidth=0.8)

        ax.set_xlabel('Epoch', fontsize=12, fontweight='medium')
        ax.set_ylabel('Success Rate', fontsize=12, fontweight='medium')
        ax.set_title(f'(c) Training: {exp_labels[idx]}', fontsize=13, fontweight='bold', pad=10)
        ax.set_xticks(epochs)
        ax.legend(loc='upper right', fontsize=9, framealpha=0.85)
        ax.set_ylim(0.05, 0.9)
        ax.grid(True, linestyle='--', alpha=0.25)

    plt.savefig(output_dir / 'comprehensive_results.png', dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {output_dir / 'comprehensive_results.png'}")


def plot_all_training_curves(data: dict, output_dir: Path, exp_infos: list = None) -> None:
    """全部训练曲线 (2×3)"""
    experiments = list(data.keys())
    if exp_infos is None:
        exp_infos = [(exp, f'Exp{i+1}') for i, exp in enumerate(experiments)]

    n_plots = min(6, len(experiments))
    n_rows = (n_plots + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(19, 11))
    fig.suptitle('Training Curves Across All Experiments', fontsize=16, fontweight='bold', y=0.98)

    for idx in range(6):
        row, col = idx // 3, idx % 3
        if n_rows == 1:
            ax = axes[col] if n_rows == 1 else axes[row, col]
        else:
            ax = axes[row, col] if n_rows > 1 else axes[col]

        if idx < len(experiments):
            exp_key, title = exp_infos[idx]
            exp = data[exp_key]['results']

            epochs = None
            for method, lbl, mkr in zip(METHODS, METHOD_LABELS, MARKERS):
                if method in exp and 'epochs' in exp[method]:
                    if epochs is None:
                        if 'seeds' in exp[method]:
                            seed_keys = list(exp[method]['seeds'].keys())
                            n_epochs = len(exp[method]['seeds'][seed_keys[0]]['epochs'])
                        else:
                            n_epochs = len(exp[method]['epochs'])
                        epochs = list(range(1, n_epochs + 1))

                    if 'seeds' in exp[method]:
                        avg_sr = []
                        for e_i in range(n_epochs):
                            vals = [exp[method]['seeds'][s]['epochs'][e_i]['sr'] for s in seed_keys]
                            avg_sr.append(np.mean(vals))
                        ax.plot(epochs, avg_sr, marker=mkr, label=lbl, color=COLORS[method],
                               linewidth=2.2, markersize=7, markeredgecolor='white', markeredgewidth=0.8)
                    else:
                        sr = [ep['sr'] for ep in exp[method]['epochs']]
                        ax.plot(epochs, sr, marker=mkr, label=lbl, color=COLORS[method],
                               linewidth=2.2, markersize=7, markeredgecolor='white', markeredgewidth=0.8)

            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel('Success Rate', fontsize=10)
            ax.set_title(title, fontsize=11, fontweight='bold', pad=6)
            ax.set_xticks(epochs)
            ax.legend(loc='upper left', fontsize=7.5, framealpha=0.85)
            ax.set_ylim(0.05, 0.9)
            ax.grid(True, linestyle='--', alpha=0.25)

    plt.savefig(output_dir / 'all_training_curves.png', dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {output_dir / 'all_training_curves.png'}")


def plot_robustness(data: dict, output_dir: Path, exp_key: str, seeds: list, title: str = None) -> None:
    """多 Seed 鲁棒性验证"""
    if exp_key not in data:
        print(f"Warning: experiment '{exp_key}' not found")
        return

    exp = data[exp_key]['results']
    fig, ax = plt.subplots(figsize=(9, 6))

    x = np.arange(len(seeds))
    bw = 0.24

    bars_list = []
    for i, (method, lbl, col) in enumerate(zip(METHODS, METHOD_LABELS,
            [COLORS['no_memory'], COLORS['memrl'], COLORS['task_memrl']])):
        if method in exp and 'seeds' in exp[method]:
            vals = []
            for seed in seeds:
                seed_key = f'seed_{seed}'
                if seed_key in exp[method]['seeds']:
                    vals.append(exp[method]['seeds'][seed_key].get('final_sr', 0))
                else:
                    vals.append(0)
            b = ax.bar(x + (i - 1) * bw, vals, bw, label=lbl, color=col,
                      edgecolor='white', linewidth=1.2)
            bars_list.append((b, vals))

    ax.set_ylabel('Success Rate', fontsize=13, fontweight='medium')
    ax.set_xlabel('Random Seed', fontsize=13, fontweight='medium')
    ax.set_title(title or f'Multi-Seed Robustness', fontsize=14, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f'seed={s}' for s in seeds], fontsize=11)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.85)
    ax.set_ylim(0, 1.0)
    ax.grid(True, linestyle='--', alpha=0.25, axis='y')

    for bars, vals in bars_list:
        for bar, v in zip(bars, vals):
            ax.annotate(f'{v:.2f}', xy=(bar.get_x() + bar.get_width() / 2, v),
                       xytext=(0, 4), textcoords="offset points",
                       ha='center', va='bottom', fontsize=10, fontweight='bold', color='#333333')

    plt.tight_layout()
    plt.savefig(output_dir / 'robustness_comparison.png', dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Generated: {output_dir / 'robustness_comparison.png'}")


def main():
    parser = argparse.ArgumentParser(description='MemRL experiment visualization - enhanced version')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to results JSON file')
    parser.add_argument('--output-dir', type=str, default='./plots',
                       help='Output directory for plots')
    parser.add_argument('--exp-labels', nargs='*', default=None,
                       help='Labels for experiments (default: exp1, exp2, ...)')
    parser.add_argument('--seeds', nargs='*', type=int, default=[42, 123, 456],
                       help='Seed values for robustness plot')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(args.input)
    print(f"Loaded data from {args.input}")

    plot_comprehensive_results(data, output_dir, args.exp_labels)
    plot_all_training_curves(data, output_dir)

    # 找第一个多seed实验做鲁棒性图
    multiseed_exp = None
    for exp_key in data.keys():
        if 'seeds' in data[exp_key].get('results', {}).get('no_memory', {}):
            multiseed_exp = exp_key
            break

    if multiseed_exp:
        plot_robustness(data, output_dir, multiseed_exp, args.seeds)

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == '__main__':
    main()
