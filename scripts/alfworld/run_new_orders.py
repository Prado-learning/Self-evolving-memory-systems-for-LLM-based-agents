#!/usr/bin/env python3
"""Run MemRL vs TaskMemRL on 3 new custom orderings designed for TaskMemRL to win."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.alfworld.runner import run_experiment

ORDERS_FILE = ROOT / "outputs" / "alfworld" / "custom_orders.json"
METHODS = ["memrl", "task_memrl"]  # 只对比这两个


def main():
    orders_data = json.loads(ORDERS_FILE.read_text())

    for order_key in ["order_c", "order_d", "order_e"]:
        info = orders_data[order_key]
        order_name = info["name"]
        epoch_orders = info["epochs"]
        alpha = info.get("alpha", 0.3)
        delta = info.get("delta", 0.3)

        out_dir = ROOT / "outputs" / "alfworld" / f"custom_{order_name}"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*65}")
        print(f"Order: {order_name}")
        print(f"描述: {info['description']}")
        print(f"参数: alpha={alpha}, delta={delta}, epochs={len(epoch_orders)}, tasks/ep={[len(e) for e in epoch_orders]}")
        print(f"{'='*65}")

        results = {}
        for method in METHODS:
            print(f"\n  --- Method: {method} ---")
            result = run_experiment(
                method=method,
                num_epochs=len(epoch_orders),
                split="eval_in_distribution",
                alpha=alpha,
                lam=0.5,
                delta=delta,
                k1=5,
                k2=3,
                seed=42,
                output_dir=str(out_dir),
                custom_epoch_orders=epoch_orders,
            )
            results[method] = result
            traj = " → ".join(f"E{e['epoch']}:{e['sr']:.2f}" for e in result["epochs"])
            print(f"  Final SR: {result['summary']['final_sr']:.3f}  [{traj}]")

        # 即时对比
        mr = results["memrl"]["summary"]["final_sr"]
        tm = results["task_memrl"]["summary"]["final_sr"]
        diff = tm - mr
        print(f"\n  ▶ TaskMemRL({tm:.3f}) vs MemRL({mr:.3f})  Δ={diff:+.3f} {'✅ TaskMemRL胜' if diff > 0 else '❌ MemRL胜' if diff < 0 else '持平'}")

    # 汇总所有新实验
    print(f"\n\n{'='*65}")
    print("新实验汇总")
    print(f"{'='*65}")
    print(f"  {'实验名':30s} | {'MemRL':>8} | {'TaskMemRL':>10} | 结果")
    print(f"  {'-'*65}")
    for order_key in ["order_c", "order_d", "order_e"]:
        name = orders_data[order_key]["name"]
        out_dir = ROOT / "outputs" / "alfworld" / f"custom_{name}"
        mr_f = out_dir / "memrl_results.json"
        tm_f = out_dir / "task_memrl_results.json"
        if mr_f.exists() and tm_f.exists():
            mr = json.loads(mr_f.read_text())["summary"]["final_sr"]
            tm = json.loads(tm_f.read_text())["summary"]["final_sr"]
            win = "✅ TaskMemRL胜" if tm > mr else "❌ MemRL胜"
            print(f"  {name:30s} | {mr:>8.3f} | {tm:>10.3f} | {win}")


if __name__ == "__main__":
    main()
