#!/usr/bin/env python3
"""Analyze and compare results from ALFWorld experiments."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load_results(output_dir: str = "outputs/alfworld") -> dict:
    results = {}
    for f in Path(output_dir).glob("*_results.json"):
        method = f.stem.replace("_results", "")
        with open(f) as fp:
            results[method] = json.load(fp)
    return results


def print_comparison(results: dict) -> None:
    methods = ["no_memory", "rag", "memrl", "task_memrl"]
    available = [m for m in methods if m in results]

    if not available:
        print("No results found.")
        return

    # Header
    print("\n" + "=" * 80)
    print("ALFWorld Experiment Results: TaskMemRL vs MemRL Comparison")
    print("=" * 80)

    # Summary table
    print(f"\n{'Method':<15} {'Final SR':>10} {'CSR':>10} {'Tokens':>12} {'Memory':>8}")
    print("-" * 60)
    for m in available:
        s = results[m]["summary"]
        print(f"{m:<15} {s['final_sr']:>10.3f} {s['csr']:>10.3f} {s['total_api_tokens']:>12,} {s['memory_size']:>8}")

    # Per-epoch SR
    print("\n--- Success Rate by Epoch ---")
    header = f"{'Epoch':<8}"
    for m in available:
        header += f" {m:>12}"
    print(header)
    print("-" * (8 + 13 * len(available)))

    max_epochs = max(len(results[m]["epochs"]) for m in available)
    for e in range(max_epochs):
        row = f"{e + 1:<8}"
        for m in available:
            epochs = results[m]["epochs"]
            if e < len(epochs):
                row += f" {epochs[e]['sr']:>12.3f}"
            else:
                row += f" {'N/A':>12}"
        print(row)

    # Per-task-type SR (final epoch)
    print("\n--- Per-Task-Type Success Rate (Final Epoch) ---")
    all_types = set()
    for m in available:
        last_epoch = results[m]["epochs"][-1]
        all_types.update(last_epoch["per_task_type_sr"].keys())
    all_types = sorted(all_types)

    header = f"{'Task Type':<40}"
    for m in available:
        header += f" {m:>12}"
    print(header)
    print("-" * (40 + 13 * len(available)))

    for tt in all_types:
        row = f"{tt:<40}"
        for m in available:
            last_epoch = results[m]["epochs"][-1]
            sr = last_epoch["per_task_type_sr"].get(tt, float("nan"))
            row += f" {sr:>12.3f}"
        print(row)

    # Improvement analysis
    if "memrl" in results and "task_memrl" in results:
        print("\n--- TaskMemRL vs MemRL Improvement ---")
        memrl_final = results["memrl"]["summary"]["final_sr"]
        task_final = results["task_memrl"]["summary"]["final_sr"]
        diff = task_final - memrl_final
        print(f"  Final SR: MemRL={memrl_final:.3f}, TaskMemRL={task_final:.3f}, Delta={diff:+.3f}")

        memrl_csr = results["memrl"]["summary"]["csr"]
        task_csr = results["task_memrl"]["summary"]["csr"]
        diff_csr = task_csr - memrl_csr
        print(f"  CSR:      MemRL={memrl_csr:.3f}, TaskMemRL={task_csr:.3f}, Delta={diff_csr:+.3f}")

        # Per-task-type comparison
        memrl_tt = results["memrl"]["epochs"][-1]["per_task_type_sr"]
        task_tt = results["task_memrl"]["epochs"][-1]["per_task_type_sr"]
        print("\n  Per-task-type deltas:")
        for tt in sorted(set(memrl_tt) | set(task_tt)):
            m_sr = memrl_tt.get(tt, 0.0)
            t_sr = task_tt.get(tt, 0.0)
            delta = t_sr - m_sr
            marker = " ***" if abs(delta) > 0.05 else ""
            print(f"    {tt:<40} MemRL={m_sr:.3f} TaskMemRL={t_sr:.3f} Delta={delta:+.3f}{marker}")

    # Token efficiency
    if len(available) > 1:
        print("\n--- Token Efficiency ---")
        for m in available:
            s = results[m]["summary"]
            total_tasks = sum(e["num_tasks"] for e in results[m]["epochs"])
            tokens_per_task = s["total_api_tokens"] / total_tasks if total_tasks else 0
            print(f"  {m:<15}: {tokens_per_task:,.0f} tokens/task (total: {s['total_api_tokens']:,})")


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs/alfworld"
    results = load_results(output_dir)
    print_comparison(results)
