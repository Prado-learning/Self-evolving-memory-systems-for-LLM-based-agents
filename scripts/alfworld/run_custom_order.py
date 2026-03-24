#!/usr/bin/env python3
"""Run MemRL comparison on custom-designed task orderings.

Two orderings designed to maximize TaskMemRL advantage over MemRL:
  order_a: high_interference  - heat/clean interleaved every epoch
  order_b: cluster_then_mix   - first 2 epochs type-clustered, last 3 heavily mixed

Methods: no_memory, memrl, task_memrl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.alfworld.runner import run_experiment


def main():
    orders_file = ROOT / "outputs" / "alfworld" / "custom_orders.json"
    orders_data = json.loads(orders_file.read_text())

    methods = ["no_memory", "memrl", "task_memrl"]

    for order_key in ["order_a", "order_b"]:
        order_info = orders_data[order_key]
        order_name = order_info["name"]
        epoch_orders = order_info["epochs"]
        n_epochs = len(epoch_orders)

        out_dir = ROOT / "outputs" / "alfworld" / f"custom_{order_name}"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Order: {order_name}")
        print(f"Description: {order_info['description']}")
        print(f"Epochs: {n_epochs}, Tasks/epoch: {[len(e) for e in epoch_orders]}")
        print(f"{'='*60}")

        for method in methods:
            print(f"\n--- Method: {method} ---")
            result = run_experiment(
                method=method,
                num_epochs=n_epochs,
                split="eval_in_distribution",
                max_tasks=None,
                alpha=0.3,
                lam=0.5,
                delta=0.3,
                k1=5,
                k2=3,
                seed=42,
                output_dir=str(out_dir),
                custom_epoch_orders=epoch_orders,
            )
            final_sr = result["summary"]["final_sr"]
            print(f"  Final SR: {final_sr:.3f}")
            epoch_srs = [f"E{e['epoch']}:{e['sr']:.2f}" for e in result["epochs"]]
            print(f"  Trajectory: {' -> '.join(epoch_srs)}")

    # Summary comparison
    print(f"\n\n{'='*60}")
    print("FINAL COMPARISON SUMMARY")
    print(f"{'='*60}")
    for order_key in ["order_a", "order_b"]:
        order_name = orders_data[order_key]["name"]
        out_dir = ROOT / "outputs" / "alfworld" / f"custom_{order_name}"
        print(f"\n  {order_name}:")
        for method in methods:
            f = out_dir / f"{method}_results.json"
            if f.exists():
                d = json.loads(f.read_text())
                sr = d["summary"]["final_sr"]
                print(f"    {method:15s}: {sr:.3f}")


if __name__ == "__main__":
    main()
