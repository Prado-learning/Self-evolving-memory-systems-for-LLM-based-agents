"""Experiment runner: MemRL training loop on ALFWorld.

Supports 4 methods:
  - no_memory: pure LLM, no retrieval
  - rag: Phase A only (similarity-based retrieval)
  - memrl: Two-Phase Retrieval with global Q(intent, experience)
  - task_memrl: Two-Phase Retrieval with task-specific Q(intent, experience, task_type)
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


@dataclass
class EpisodeResult:
    game_name: str
    task_type: str
    success: bool
    steps: int
    trajectory: List[str] = field(default_factory=list)


@dataclass
class EpochResult:
    epoch: int
    method: str
    results: List[EpisodeResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.success) / len(self.results)

    @property
    def per_task_type_sr(self) -> Dict[str, float]:
        from collections import defaultdict
        counts = defaultdict(lambda: [0, 0])  # [success, total]
        for r in self.results:
            counts[r.task_type][1] += 1
            if r.success:
                counts[r.task_type][0] += 1
        return {k: v[0] / v[1] if v[1] > 0 else 0.0 for k, v in counts.items()}


def extract_goal(obs: str) -> str:
    """Extract the task goal from the initial observation."""
    lines = obs.strip().split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith("Your task is to:"):
            return line.replace("Your task is to:", "").strip()
    # Fallback: look for task description patterns
    for line in lines:
        if "put" in line.lower() or "find" in line.lower() or "clean" in line.lower():
            return line.strip()
    return lines[-1] if lines else ""


def summarize_trajectory(goal: str, trajectory: List[str], success: bool) -> str:
    """Summarize a trajectory into a reusable experience."""
    status = "SUCCESS" if success else "FAILED"
    actions = " -> ".join(trajectory[-10:])  # last 10 actions
    return f"[{status}] Goal: {goal} | Actions: {actions}"


def run_episode(
    env,
    agent,
    memory_bank,
    embedder,
    method: str,
    alpha: float = 0.3,
    lam: float = 0.5,
    delta: float = 0.5,
    k1: int = 5,
    k2: int = 3,
    max_steps: int = 50,
) -> EpisodeResult:
    """Run a single episode with the given method."""

    obs, task_type, game_name, admissible = env.reset()
    goal = extract_goal(obs)

    # Retrieve memories based on method
    memories = []
    retrieved_indices = []
    if method != "no_memory" and len(memory_bank) > 0:
        query_emb = embedder.encode(goal)

        if method == "rag":
            memories = memory_bank.retrieve_phase_a(query_emb, k1=k2, delta=delta)
        elif method == "memrl":
            memories = memory_bank.retrieve_two_phase(
                query_emb, k1=k1, k2=k2, delta=delta, lam=lam, task_type=None
            )
        elif method == "task_memrl":
            memories = memory_bank.retrieve_two_phase(
                query_emb, k1=k1, k2=k2, delta=delta, lam=lam, task_type=task_type
            )

        retrieved_indices = [m["idx"] for m in memories]

    # Interact with environment
    trajectory = []
    history = []
    success = False

    for step in range(max_steps):
        action = agent.act(goal, obs, admissible, memories=memories, history=history)
        trajectory.append(action)
        history.append(f"> {action}")

        obs, reward, done, info = env.step(action)
        history.append(f"  {obs[:100]}")

        if info["won"]:
            success = True
            break
        if done:
            break

        admissible = info.get("admissible_commands", admissible)

    # Update Q-values for retrieved memories
    reward_signal = 1.0 if success else 0.0
    for idx in retrieved_indices:
        if method == "memrl":
            memory_bank.update_q(idx, reward_signal, alpha=alpha, task_type=None)
        elif method == "task_memrl":
            memory_bank.update_q(idx, reward_signal, alpha=alpha, task_type=task_type)

    # Store new experience
    if method != "no_memory":
        experience = summarize_trajectory(goal, trajectory, success)
        goal_emb = embedder.encode(goal)
        memory_bank.add(
            intent=goal,
            experience=experience,
            embedding=goal_emb,
            task_type=task_type,
        )

    return EpisodeResult(
        game_name=game_name,
        task_type=task_type,
        success=success,
        steps=len(trajectory),
        trajectory=trajectory,
    )


def run_epoch(
    env,
    agent,
    memory_bank,
    embedder,
    method: str,
    epoch: int,
    max_tasks: Optional[int] = None,
    seed: int = 42,
    custom_order: Optional[List[int]] = None,
    **kwargs,
) -> EpochResult:
    """Run one epoch over all tasks.

    The seed controls task ordering: different seeds produce different shuffled
    orderings of the game files, so when max_tasks < total games, each seed
    samples a different subset of tasks.

    If custom_order is provided, it overrides seed-based shuffling and uses
    the explicit list of game indices for this epoch.
    """
    if custom_order is not None:
        indices = custom_order
        num_games = len(indices)
    else:
        num_games = env.num_games
        # Create a shuffled ordering based on seed + epoch
        indices = list(range(num_games))
        rng = random.Random(seed + epoch)
        rng.shuffle(indices)

        if max_tasks and max_tasks < num_games:
            indices = indices[:max_tasks]
            num_games = max_tasks

    epoch_result = EpochResult(epoch=epoch, method=method)

    for i, game_idx in enumerate(indices):
        log.info(f"  Epoch {epoch} | Task {i+1}/{num_games}")
        try:
            result = run_episode(env, agent, memory_bank, embedder, method, **kwargs)
            epoch_result.results.append(result)
            status = "OK" if result.success else "FAIL"
            log.info(
                f"    [{status}] {result.task_type} | steps={result.steps}"
            )
        except Exception as e:
            log.error(f"    ERROR: {e}")

    log.info(
        f"  Epoch {epoch} SR: {epoch_result.success_rate:.3f} | "
        f"Per-type: {epoch_result.per_task_type_sr}"
    )
    return epoch_result


def run_experiment(
    method: str,
    num_epochs: int = 5,
    split: str = "eval_in_distribution",
    max_tasks: Optional[int] = None,
    alpha: float = 0.3,
    lam: float = 0.5,
    delta: float = 0.5,
    k1: int = 5,
    k2: int = 3,
    seed: int = 42,
    output_dir: str = "outputs/alfworld",
    custom_epoch_orders: Optional[List[List[int]]] = None,
) -> Dict:
    """Run full experiment for one method."""
    from experiments.alfworld.alfworld_env import ALFWorldEnv
    from experiments.alfworld.agent import ALFWorldAgent
    from experiments.alfworld.embedding import Embedder
    from experiments.alfworld.llm_client import LLMClient
    from experiments.alfworld.memory_bank import MemoryBank

    log.info(f"=== Starting experiment: method={method}, epochs={num_epochs} ===")

    # Initialize components
    llm = LLMClient()
    embedder = Embedder()
    agent = ALFWorldAgent(llm)
    memory_bank = MemoryBank()
    env = ALFWorldEnv(split=split)

    all_epochs = []
    cumulative_solved = set()

    for epoch in range(1, num_epochs + 1):
        log.info(f"=== Epoch {epoch}/{num_epochs} ===")
        custom_order = None
        if custom_epoch_orders and (epoch - 1) < len(custom_epoch_orders):
            custom_order = custom_epoch_orders[epoch - 1]
        epoch_result = run_epoch(
            env, agent, memory_bank, embedder, method,
            epoch=epoch, max_tasks=max_tasks, seed=seed,
            custom_order=custom_order,
            alpha=alpha, lam=lam, delta=delta, k1=k1, k2=k2,
        )
        all_epochs.append(epoch_result)

        # Track cumulative success
        for r in epoch_result.results:
            if r.success:
                cumulative_solved.add(r.game_name)

    env.close()

    # Compile results
    total_tasks = sum(len(e.results) for e in all_epochs) // num_epochs if all_epochs else 0
    csr = len(cumulative_solved) / total_tasks if total_tasks > 0 else 0.0

    result = {
        "method": method,
        "config": {
            "num_epochs": num_epochs, "split": split, "max_tasks": max_tasks,
            "alpha": alpha, "lam": lam, "delta": delta, "k1": k1, "k2": k2, "seed": seed,
        },
        "summary": {
            "final_sr": all_epochs[-1].success_rate if all_epochs else 0.0,
            "csr": csr,
            "total_api_tokens": llm.total_tokens,
            "memory_size": len(memory_bank),
        },
        "epochs": [
            {
                "epoch": e.epoch,
                "sr": e.success_rate,
                "per_task_type_sr": e.per_task_type_sr,
                "num_tasks": len(e.results),
            }
            for e in all_epochs
        ],
    }

    # Save
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / f"{method}_results.json"
    out_file.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    log.info(f"Results saved to {out_file}")

    return result
