"""Intent-Experience-Utility memory bank with global and task-specific Q-values.

Implements MemRL's Two-Phase Retrieval:
  Phase A: similarity-based recall (cosine > delta, top k1)
  Phase B: value-aware selection (score = (1-lambda)*sim + lambda*Q, top k2)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from experiments.alfworld.embedding import cosine_similarity


class MemoryBank:
    def __init__(self):
        self.memories: List[Dict] = []

    def add(
        self,
        intent: str,
        experience: str,
        embedding: List[float],
        task_type: str,
        q_init: float = 0.0,
    ) -> int:
        idx = len(self.memories)
        self.memories.append(
            {
                "intent": intent,
                "experience": experience,
                "embedding": embedding,
                "task_type": task_type,
                "q_global": q_init,
                "q_per_task": {},  # task_type -> q_value
            }
        )
        return idx

    def get_q(self, idx: int, task_type: Optional[str] = None) -> float:
        mem = self.memories[idx]
        if task_type is None:
            return mem["q_global"]
        return mem["q_per_task"].get(task_type, 0.0)

    def update_q(
        self, idx: int, reward: float, alpha: float = 0.3, task_type: Optional[str] = None
    ) -> None:
        mem = self.memories[idx]
        if task_type is None:
            mem["q_global"] += alpha * (reward - mem["q_global"])
        else:
            old = mem["q_per_task"].get(task_type, 0.0)
            mem["q_per_task"][task_type] = old + alpha * (reward - old)

    def retrieve_phase_a(
        self,
        query_emb: List[float],
        k1: int = 5,
        delta: float = 0.5,
    ) -> List[Dict]:
        scored = []
        for i, mem in enumerate(self.memories):
            sim = cosine_similarity(query_emb, mem["embedding"])
            if sim > delta:
                scored.append({"idx": i, "sim": sim, **mem})
        scored.sort(key=lambda x: x["sim"], reverse=True)
        return scored[:k1]

    def retrieve_two_phase(
        self,
        query_emb: List[float],
        k1: int = 5,
        k2: int = 3,
        delta: float = 0.5,
        lam: float = 0.5,
        task_type: Optional[str] = None,
    ) -> List[Dict]:
        candidates = self.retrieve_phase_a(query_emb, k1=k1, delta=delta)
        if not candidates:
            return []

        # Z-score normalize sim and Q within candidate pool
        sims = np.array([c["sim"] for c in candidates])
        if task_type is not None:
            qs = np.array([c["q_per_task"].get(task_type, 0.0) for c in candidates])
        else:
            qs = np.array([c["q_global"] for c in candidates])

        sim_norm = _zscore(sims)
        q_norm = _zscore(qs)

        for i, c in enumerate(candidates):
            c["score"] = (1 - lam) * sim_norm[i] + lam * q_norm[i]

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:k2]

    def __len__(self) -> int:
        return len(self.memories)


def _zscore(arr: np.ndarray) -> np.ndarray:
    std = arr.std()
    if std < 1e-12:
        return np.zeros_like(arr)
    return (arr - arr.mean()) / std
