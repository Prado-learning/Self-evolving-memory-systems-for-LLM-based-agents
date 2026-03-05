"""Minimal MemRL reproduction (baseline + task-specific Q-networks)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import random


@dataclass
class Episode:
    intent: int
    task_type: int
    experience_value: float
    context_id: int


class ToyMemRLEnv:
    def __init__(self, num_intents: int = 4, num_task_types: int = 3, num_contexts: int = 8, num_actions: int = 6, noise: float = 0.12):
        self.num_intents = num_intents
        self.num_task_types = num_task_types
        self.num_contexts = num_contexts
        self.num_actions = num_actions
        self.noise = noise

    def sample(self, rng: random.Random) -> Episode:
        intent = rng.randrange(self.num_intents)
        task_type = rng.randrange(self.num_task_types)
        context_id = rng.randrange(self.num_contexts)
        experience_value = context_id + rng.uniform(-self.noise, self.noise)
        return Episode(intent=intent, task_type=task_type, experience_value=experience_value, context_id=context_id)

    def optimal_action(self, ep: Episode) -> int:
        return (ep.intent * 2 + ep.context_id + ep.task_type * 3) % self.num_actions

    def reward(self, ep: Episode, action: int) -> float:
        return 1.0 if action == self.optimal_action(ep) else -0.2


class MemoryBank:
    def __init__(self, num_contexts: int):
        self.prototypes = [float(i) for i in range(num_contexts)]

    def retrieve_slot(self, experience_value: float) -> int:
        best_idx, best_dist = 0, float("inf")
        for idx, proto in enumerate(self.prototypes):
            dist = abs(experience_value - proto)
            if dist < best_dist:
                best_idx, best_dist = idx, dist
        return best_idx


class QTable:
    def __init__(self, num_actions: int, lr: float = 0.25):
        self.num_actions = num_actions
        self.lr = lr
        self.table: Dict[Tuple[int, int], List[float]] = {}

    def _ensure(self, key: Tuple[int, int]) -> List[float]:
        if key not in self.table:
            self.table[key] = [0.0 for _ in range(self.num_actions)]
        return self.table[key]

    def act(self, key: Tuple[int, int], eps: float, rng: random.Random) -> int:
        q = self._ensure(key)
        if rng.random() < eps:
            return rng.randrange(self.num_actions)
        return max(range(self.num_actions), key=lambda a: q[a])

    def update(self, key: Tuple[int, int], action: int, reward: float) -> None:
        q = self._ensure(key)
        q[action] += self.lr * (reward - q[action])


class BaselineMemRLAgent:
    """Q(intent, memory_slot)."""

    def __init__(self, num_actions: int, lr: float = 0.25):
        self.q = QTable(num_actions=num_actions, lr=lr)

    def act(self, intent: int, memory_slot: int, eps: float, rng: random.Random) -> int:
        return self.q.act((intent, memory_slot), eps=eps, rng=rng)

    def learn(self, intent: int, memory_slot: int, action: int, reward: float) -> None:
        self.q.update((intent, memory_slot), action, reward)


class TaskSpecificMemRLAgent:
    """Q(intent, memory_slot, task_type) via per-task Q tables."""

    def __init__(self, num_actions: int, num_task_types: int, lr: float = 0.25):
        self.by_task = {t: QTable(num_actions=num_actions, lr=lr) for t in range(num_task_types)}

    def act(self, task_type: int, intent: int, memory_slot: int, eps: float, rng: random.Random) -> int:
        return self.by_task[task_type].act((intent, memory_slot), eps=eps, rng=rng)

    def learn(self, task_type: int, intent: int, memory_slot: int, action: int, reward: float) -> None:
        self.by_task[task_type].update((intent, memory_slot), action, reward)


@dataclass
class TrainResult:
    avg_reward: float
    retrieval_accuracy: float
    convergence_step: int


def _eps_at(step: int, total_steps: int, eps_start: float = 0.3, eps_end: float = 0.03) -> float:
    ratio = min(max(step / max(1, total_steps), 0.0), 1.0)
    return eps_start + (eps_end - eps_start) * ratio


def train_baseline(env: ToyMemRLEnv, memory: MemoryBank, steps: int, seed: int = 0) -> TrainResult:
    rng = random.Random(seed)
    agent = BaselineMemRLAgent(num_actions=env.num_actions)
    rewards: List[float] = []
    retrieval_hit = 0
    convergence_step = -1
    for step in range(steps):
        ep = env.sample(rng)
        slot = memory.retrieve_slot(ep.experience_value)
        retrieval_hit += int(slot == ep.context_id)
        action = agent.act(ep.intent, slot, eps=_eps_at(step, steps), rng=rng)
        r = env.reward(ep, action)
        agent.learn(ep.intent, slot, action, r)
        rewards.append(r)
        if convergence_step < 0 and step >= 100 and sum(rewards[-100:]) / 100 >= 0.58:
            convergence_step = step + 1

    return TrainResult(sum(rewards) / len(rewards), retrieval_hit / steps, convergence_step)


def train_task_specific(env: ToyMemRLEnv, memory: MemoryBank, steps: int, seed: int = 0) -> TrainResult:
    rng = random.Random(seed)
    agent = TaskSpecificMemRLAgent(num_actions=env.num_actions, num_task_types=env.num_task_types)
    rewards: List[float] = []
    retrieval_hit = 0
    convergence_step = -1
    for step in range(steps):
        ep = env.sample(rng)
        slot = memory.retrieve_slot(ep.experience_value)
        retrieval_hit += int(slot == ep.context_id)
        action = agent.act(ep.task_type, ep.intent, slot, eps=_eps_at(step, steps), rng=rng)
        r = env.reward(ep, action)
        agent.learn(ep.task_type, ep.intent, slot, action, r)
        rewards.append(r)
        if convergence_step < 0 and step >= 100 and sum(rewards[-100:]) / 100 >= 0.58:
            convergence_step = step + 1

    return TrainResult(sum(rewards) / len(rewards), retrieval_hit / steps, convergence_step)


def evaluate_success_rate(env: ToyMemRLEnv, memory: MemoryBank, baseline: bool, train_steps: int, eval_episodes: int, seed: int) -> float:
    rng = random.Random(seed)
    if baseline:
        agent = BaselineMemRLAgent(num_actions=env.num_actions)
        for step in range(train_steps):
            ep = env.sample(rng)
            slot = memory.retrieve_slot(ep.experience_value)
            action = agent.act(ep.intent, slot, eps=_eps_at(step, train_steps), rng=rng)
            agent.learn(ep.intent, slot, action, env.reward(ep, action))
        success = 0
        for _ in range(eval_episodes):
            ep = env.sample(rng)
            slot = memory.retrieve_slot(ep.experience_value)
            action = agent.act(ep.intent, slot, eps=0.0, rng=rng)
            success += int(action == env.optimal_action(ep))
        return success / eval_episodes

    agent = TaskSpecificMemRLAgent(num_actions=env.num_actions, num_task_types=env.num_task_types)
    for step in range(train_steps):
        ep = env.sample(rng)
        slot = memory.retrieve_slot(ep.experience_value)
        action = agent.act(ep.task_type, ep.intent, slot, eps=_eps_at(step, train_steps), rng=rng)
        agent.learn(ep.task_type, ep.intent, slot, action, env.reward(ep, action))
    success = 0
    for _ in range(eval_episodes):
        ep = env.sample(rng)
        slot = memory.retrieve_slot(ep.experience_value)
        action = agent.act(ep.task_type, ep.intent, slot, eps=0.0, rng=rng)
        success += int(action == env.optimal_action(ep))
    return success / eval_episodes
