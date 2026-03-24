#!/usr/bin/env python3
"""Run multi-seed ALFWorld experiment for statistical significance."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.alfworld.runner import run_experiment


def aggregate_results(all_seed_results: list[dict]) -> dict:
    """Aggregate results across seeds, computing mean and std."""
    n_seeds = len(all_seed_results)
    n_epochs = len(all_seed_results[0]["epochs"])

    # Per-epoch SR across seeds
    epoch_srs = []
    for e_idx in range(n_epochs):
        srs = [r["epochs"][e_idx]["sr"] for r in all_seed_results]
        epoch_srs.append({"mean": np.mean(srs), "std": np.std(srs), "values": srs})

    # Final SR across seeds
    final_srs = [r["summary"]["final_sr"] for r in all_seed_results]

    # Per-task-type SR from final epoch
    all_task_types = set()
    for r in all_seed_results:
        all_task_types.update(r["epochs"][-1]["per_task_type_sr"].keys())

    per_type_stats = {}
    for tt in sorted(all_task_types):
        vals = []
        for r in all_seed_results:
            vals.append(r["epochs"][-1]["per_task_type_sr"].get(tt, 0.0))
        per_type_stats[tt] = {"mean": np.mean(vals), "std": np.std(vals), "values": vals}

    return {
        "n_seeds": n_seeds,
        "final_sr": {"mean": float(np.mean(final_srs)), "std": float(np.std(final_srs)), "values": final_srs},
        "epoch_srs": [{"epoch": i + 1, "mean": float(e["mean"]), "std": float(e["std"]), "values": e["values"]} for i, e in enumerate(epoch_srs)],
        "per_task_type_sr": {k: {"mean": float(v["mean"]), "std": float(v["std"])} for k, v in per_type_stats.items()},
        "total_tokens": sum(r["summary"]["total_api_tokens"] for r in all_seed_results),
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-seed ALFWorld experiment")
    parser.add_argument("--methods", nargs="+", default=["no_memory", "rag", "memrl", "task_memrl"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-tasks", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="outputs/alfworld/multiseed")
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--lam", type=float, default=0.5)
    parser.add_argument("--delta", type=float, default=0.3)
    parser.add_argument("--k1", type=int, default=5)
    parser.add_argument("--k2", type=int, default=3)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for method in args.methods:
        print(f"\n{'='*60}")
        print(f"Method: {method}")
        print(f"{'='*60}")

        seed_results = []
        for seed in args.seeds:
            print(f"\n--- Seed {seed} ---")
            seed_out_dir = out_dir / f"seed_{seed}"
            seed_out_dir.mkdir(parents=True, exist_ok=True)

            result = run_experiment(
                method=method,
                num_epochs=args.epochs,
                max_tasks=args.max_tasks,
                alpha=args.alpha,
                lam=args.lam,
                delta=args.delta,
                k1=args.k1,
                k2=args.k2,
                seed=seed,
                output_dir=str(seed_out_dir),
            )
            seed_results.append(result)

            print(f"  Seed {seed}: Final SR = {result['summary']['final_sr']:.3f}")

        # Aggregate across seeds
        agg = aggregate_results(seed_results)
        all_results[method] = agg

        print(f"\n{method} Summary: SR = {agg['final_sr']['mean']:.3f} +/- {agg['final_sr']['std']:.3f}")

    # Save aggregated results
    agg_file = out_dir / "aggregated_results.json"
    # Convert numpy types for JSON serialization
    agg_file.write_text(json.dumps(all_results, indent=2, ensure_ascii=False, default=float))
    print(f"\nAggregated results saved to {agg_file}")

    # Print comparison table
    print(f"\n{'='*70}")
    print("Multi-Seed Comparison (mean +/- std)")
    print(f"{'='*70}")
    print(f"{'Method':<15} {'Final SR':>15} {'Epoch Trajectory':>35}")
    print("-" * 70)
    for method in args.methods:
        if method in all_results:
            r = all_results[method]
            sr = f"{r['final_sr']['mean']:.3f} +/- {r['final_sr']['std']:.3f}"
            traj = " -> ".join(f"{e['mean']:.2f}" for e in r["epoch_srs"])
            print(f"{method:<15} {sr:>15} {traj:>35}")


if __name__ == "__main__":
    main()
