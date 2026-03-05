#!/usr/bin/env python3
"""Run MemRL baseline vs task-specific comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.memrl.memrl_core import (
    MemoryBank,
    ToyMemRLEnv,
    evaluate_success_rate,
    train_baseline,
    train_task_specific,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--eval-episodes", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="outputs/memrl_comparison.json")
    args = parser.parse_args()

    env = ToyMemRLEnv()
    memory = MemoryBank(num_contexts=env.num_contexts)

    baseline_train = train_baseline(env, memory, steps=args.steps, seed=args.seed)
    improved_train = train_task_specific(env, memory, steps=args.steps, seed=args.seed)

    baseline_success = evaluate_success_rate(
        env, memory, baseline=True, train_steps=args.steps, eval_episodes=args.eval_episodes, seed=args.seed
    )
    improved_success = evaluate_success_rate(
        env, memory, baseline=False, train_steps=args.steps, eval_episodes=args.eval_episodes, seed=args.seed
    )

    result = {
        "config": {
            "steps": args.steps,
            "eval_episodes": args.eval_episodes,
            "seed": args.seed,
            "env": {
                "num_intents": env.num_intents,
                "num_task_types": env.num_task_types,
                "num_contexts": env.num_contexts,
                "num_actions": env.num_actions,
            },
        },
        "baseline": {
            "success_rate": baseline_success,
            "retrieval_accuracy": baseline_train.retrieval_accuracy,
            "avg_reward": baseline_train.avg_reward,
            "convergence_step": baseline_train.convergence_step,
        },
        "task_specific": {
            "success_rate": improved_success,
            "retrieval_accuracy": improved_train.retrieval_accuracy,
            "avg_reward": improved_train.avg_reward,
            "convergence_step": improved_train.convergence_step,
        },
        "delta": {
            "success_rate": improved_success - baseline_success,
            "avg_reward": improved_train.avg_reward - baseline_train.avg_reward,
            "convergence_step": (
                baseline_train.convergence_step - improved_train.convergence_step
                if baseline_train.convergence_step > 0 and improved_train.convergence_step > 0
                else None
            ),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("=== MemRL Comparison ===")
    print(f"Baseline success_rate      : {baseline_success:.4f}")
    print(f"Task-specific success_rate : {improved_success:.4f}")
    print(f"Delta success_rate         : {result['delta']['success_rate']:+.4f}")
    print(f"Baseline retrieval_acc     : {baseline_train.retrieval_accuracy:.4f}")
    print(f"Task-specific retrieval_acc: {improved_train.retrieval_accuracy:.4f}")
    print(f"Baseline convergence_step  : {baseline_train.convergence_step}")
    print(f"Task-specific convergence  : {improved_train.convergence_step}")
    print(f"Saved result -> {out_path}")


if __name__ == "__main__":
    main()
