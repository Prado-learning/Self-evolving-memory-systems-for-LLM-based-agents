#!/usr/bin/env python3
"""Aggregate multi-seed results and perform statistical analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy import stats


def load_seed_results(output_dir: Path, seeds: List[int], methods: List[str]) -> Dict:
    """Load individual seed results and organize by method and seed."""
    all_results = {m: [] for m in methods}

    for seed in seeds:
        seed_dir = output_dir / f"seed_{seed}"
        if not seed_dir.exists():
            print(f"Warning: {seed_dir} does not exist, skipping seed {seed}")
            continue

        for method in methods:
            result_file = seed_dir / f"{method}_results.json"
            if result_file.exists():
                with open(result_file) as f:
                    all_results[method].append(json.load(f))
            else:
                print(f"Warning: {result_file} not found")

    return all_results


def aggregate_method(method_results: List[dict], n_epochs: int) -> dict:
    """Aggregate results across seeds for a single method."""
    if not method_results:
        return {}

    # Final SR across seeds
    final_srs = [r["summary"]["final_sr"] for r in method_results]

    # Epoch-level SR
    epoch_srs = []
    for e_idx in range(n_epochs):
        srs = []
        for r in method_results:
            if e_idx < len(r["epochs"]):
                srs.append(r["epochs"][e_idx]["sr"])
            else:
                srs.append(np.nan)
        epoch_srs.append({
            "mean": float(np.nanmean(srs)),
            "std": float(np.nanstd(srs)),
            "values": srs,
        })

    # Per-task-type SR from final epoch
    all_task_types = set()
    for r in method_results:
        if r["epochs"]:
            all_task_types.update(r["epochs"][-1]["per_task_type_sr"].keys())

    per_type_stats = {}
    for tt in sorted(all_task_types):
        vals = []
        for r in method_results:
            if r["epochs"]:
                vals.append(r["epochs"][-1]["per_task_type_sr"].get(tt, 0.0))
        per_type_stats[tt] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "values": vals,
        }

    return {
        "n_seeds": len(method_results),
        "final_sr": {
            "mean": float(np.mean(final_srs)),
            "std": float(np.std(final_srs)),
            "values": final_srs,
        },
        "epoch_srs": epoch_srs,
        "per_task_type_sr": per_type_stats,
        "total_tokens": sum(r["summary"]["total_api_tokens"] for r in method_results),
    }


def perform_ttest(method_a: str, data_a: dict, method_b: str, data_b: dict) -> dict:
    """Perform paired t-test between two methods."""
    a_values = np.array(data_a["final_sr"]["values"])
    b_values = np.array(data_b["final_sr"]["values"])

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(a_values, b_values)

    # Effect size (Cohen's d)
    diff = a_values - b_values
    cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0

    return {
        "comparison": f"{method_a} vs {method_b}",
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": float(cohens_d),
        "significant_005": p_value < 0.05,
        "significant_001": p_value < 0.01,
        f"{method_a}_mean": float(np.mean(a_values)),
        f"{method_b}_mean": float(np.mean(b_values)),
        "mean_diff": float(np.mean(diff)),
    }


def generate_report(all_results: Dict, methods: List[str], seeds: List[int]) -> str:
    """Generate a text report of the analysis."""
    lines = []
    lines.append("=" * 70)
    lines.append("ALFWorld Multi-Seed Experiment Analysis Report")
    lines.append("=" * 70)
    lines.append(f"Seeds: {seeds}")
    lines.append(f"Methods: {methods}")
    lines.append("")

    # Summary table
    lines.append("-" * 70)
    lines.append("Final Success Rate Summary (mean ± std)")
    lines.append("-" * 70)
    lines.append(f"{'Method':<20} {'Final SR':>15} {'Significance':>15}")
    lines.append("-" * 70)

    for method in methods:
        if method in all_results and "final_sr" in all_results[method]:
            sr = all_results[method]["final_sr"]
            lines.append(f"{method:<20} {sr['mean']:.3f} ± {sr['std']:.3f}")
    lines.append("")

    # Epoch trajectory
    lines.append("-" * 70)
    lines.append("Success Rate Trajectory (mean across seeds)")
    lines.append("-" * 70)
    lines.append(f"{'Epoch':>8}", end="")
    for method in methods:
        if method in all_results:
            lines.append(f" {method:>15}", end="")
    lines.append("")
    lines.append("-" * 70)

    if methods and methods[0] in all_results and "epoch_srs" in all_results[methods[0]]:
        n_epochs = len(all_results[methods[0]]["epoch_srs"])
        for e_idx in range(n_epochs):
            lines.append(f"{e_idx + 1:>8}", end="")
            for method in methods:
                if method in all_results and e_idx < len(all_results[method]["epoch_srs"]):
                    sr = all_results[method]["epoch_srs"][e_idx]
                    lines.append(f" {sr['mean']:.3f} ± {sr['std']:.3f}", end="")
                else:
                    lines.append(f" {'N/A':>15}", end="")
            lines.append("")
    lines.append("")

    # Per-task-type analysis
    lines.append("-" * 70)
    lines.append("Per-Task-Type Success Rate (Final Epoch)")
    lines.append("-" * 70)

    if methods and methods[0] in all_results and "per_task_type_sr" in all_results[methods[0]]:
        task_types = sorted(all_results[methods[0]]["per_task_type_sr"].keys())
        lines.append(f"{'Task Type':<35}", end="")
        for method in methods:
            if method in all_results:
                lines.append(f" {method:>12}", end="")
        lines.append("")
        lines.append("-" * 70)

        for tt in task_types:
            short_name = tt.replace("pick_", "").replace("_then_place_in_recep", "").replace("_and_place", "+")
            lines.append(f"{short_name:<35}", end="")
            for method in methods:
                if method in all_results and tt in all_results[method]["per_task_type_sr"]:
                    sr = all_results[method]["per_task_type_sr"][tt]
                    lines.append(f" {sr['mean']:.3f}±{sr['std']:.3f}", end="")
                else:
                    lines.append(f" {'N/A':>12}", end="")
            lines.append("")
    lines.append("")

    # Statistical significance: TaskMemRL vs MemRL
    lines.append("=" * 70)
    lines.append("Statistical Significance Analysis")
    lines.append("=" * 70)

    if "memrl" in all_results and "task_memrl" in all_results:
        ttest_result = perform_ttest("task_memrl", all_results["task_memrl"],
                                     "memrl", all_results["memrl"])
        lines.append(f"\nTaskMemRL vs MemRL (Paired t-test):")
        lines.append(f"  Mean difference: {ttest_result['mean_diff']:+.4f}")
        lines.append(f"  t-statistic: {ttest_result['t_statistic']:.4f}")
        lines.append(f"  p-value: {ttest_result['p_value']:.6f}")
        lines.append(f"  Cohen's d: {ttest_result['cohens_d']:.4f}")
        lines.append(f"  Significant (p < 0.05): {ttest_result['significant_005']}")
        lines.append(f"  Significant (p < 0.01): {ttest_result['significant_001']}")

        if ttest_result['significant_005']:
            lines.append("\n  ✓ TaskMemRL is significantly better than MemRL (p < 0.05)")
        else:
            lines.append("\n  ✗ No significant difference (p >= 0.05)")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Aggregate and analyze multi-seed results")
    parser.add_argument("--output-dir", type=str, default="outputs/alfworld/multiseed_v2",
                        help="Output directory containing seed_* subdirectories")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456],
                        help="Seed values to aggregate")
    parser.add_argument("--methods", nargs="+", default=["no_memory", "rag", "memrl", "task_memrl"],
                        help="Methods to analyze")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of epochs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print(f"Loading results from {output_dir}...")
    all_results = load_seed_results(output_dir, args.seeds, args.methods)

    # Aggregate
    aggregated = {}
    for method in args.methods:
        if all_results[method]:
            aggregated[method] = aggregate_method(all_results[method], args.epochs)
            print(f"  {method}: {len(all_results[method])} seeds")

    # Save aggregated JSON
    agg_file = output_dir / "aggregated_results.json"
    # Convert numpy types
    json.dump(aggregated, open(agg_file, "w"), indent=2, default=float)
    print(f"\nAggregated results saved to {agg_file}")

    # Generate and print report
    report = generate_report(aggregated, args.methods, args.seeds)
    print("\n" + report)

    # Save report
    report_file = output_dir / "analysis_report.txt"
    Path(report_file).write_text(report)
    print(f"\nReport saved to {report_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())