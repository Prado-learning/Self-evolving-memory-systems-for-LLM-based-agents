#!/usr/bin/env python3
"""Run ALFWorld experiment: compare MemRL vs TaskMemRL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.alfworld.runner import run_experiment


def main():
    parser = argparse.ArgumentParser(description="Run ALFWorld MemRL experiment")
    parser.add_argument("--method", type=str, required=True,
                        choices=["no_memory", "rag", "memrl", "task_memrl"],
                        help="Retrieval method to use")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--max-tasks", type=int, default=None,
                        help="Max tasks per epoch (None = all)")
    parser.add_argument("--split", type=str, default="eval_in_distribution",
                        choices=["eval_in_distribution", "eval_out_of_distribution"])
    parser.add_argument("--alpha", type=float, default=0.3, help="Q-value learning rate")
    parser.add_argument("--lam", type=float, default=0.5, help="Similarity-Q weight balance")
    parser.add_argument("--delta", type=float, default=0.5, help="Similarity threshold")
    parser.add_argument("--k1", type=int, default=5, help="Phase-A recall size")
    parser.add_argument("--k2", type=int, default=3, help="Phase-B select size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/alfworld")
    args = parser.parse_args()

    result = run_experiment(
        method=args.method,
        num_epochs=args.epochs,
        split=args.split,
        max_tasks=args.max_tasks,
        alpha=args.alpha,
        lam=args.lam,
        delta=args.delta,
        k1=args.k1,
        k2=args.k2,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    print(f"\n=== Results for {args.method} ===")
    print(f"Final SR: {result['summary']['final_sr']:.3f}")
    print(f"CSR: {result['summary']['csr']:.3f}")
    print(f"API tokens: {result['summary']['total_api_tokens']}")
    print(f"Memory size: {result['summary']['memory_size']}")
    for e in result["epochs"]:
        print(f"  Epoch {e['epoch']}: SR={e['sr']:.3f} | {e['per_task_type_sr']}")


if __name__ == "__main__":
    main()
