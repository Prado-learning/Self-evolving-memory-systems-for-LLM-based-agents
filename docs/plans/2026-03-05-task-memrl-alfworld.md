# Task-Aware MemRL on ALFWorld Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Validate that task-specific Q-values Q(intent, experience, task_type) outperform global Q-values Q(intent, experience) on the ALFWorld benchmark with 6 distinct task types.

**Architecture:** Implement the full MemRL pipeline (Intent-Experience-Utility triplets + Two-Phase Retrieval) on ALFWorld using DeepSeek V3 as backbone LLM and local BGE embeddings. Compare 4 methods: No Memory, RAG, MemRL (global Q), TaskMemRL (task-specific Q). The only difference between MemRL and TaskMemRL is whether Q-values are maintained per-task-type or globally.

**Tech Stack:** Python 3.13, OpenAI SDK (Volcengine-compatible), transformers (local BGE embedding), ALFWorld 0.4.2, numpy, json

**Environment:**
- ALFWORLD_DATA=/home/user/alfworld_data
- .env at project root with ARK_API_KEY, ARK_MODEL, ARK_BASE_URL
- TextWorld patched for Python 3.13 compatibility

---

## Task 1: LLM Client and Embedding Module

**Files:**
- Create: `experiments/alfworld/__init__.py`
- Create: `experiments/alfworld/llm_client.py`
- Create: `experiments/alfworld/embedding.py`
- Create: `tests/alfworld/__init__.py`
- Create: `tests/alfworld/test_client.py`

## Task 2: Memory Bank with Global and Task-Specific Q-values

**Files:**
- Create: `experiments/alfworld/memory_bank.py`
- Create: `tests/alfworld/test_memory.py`

Core: each memory stores {intent, experience, embedding, task_type, q_global, q_per_task dict}. Two-Phase Retrieval with both global and task-specific Q scoring.

## Task 3: ALFWorld Environment Wrapper

**Files:**
- Create: `experiments/alfworld/alfworld_env.py`
- Create: `tests/alfworld/test_env.py`

Wraps ALFWorld config into: reset()->(obs, task_type, game_name), step(action)->(obs, reward, done, info), get_task_type(game_name)->str.

## Task 4: ALFWorld Agent with Memory

**Files:**
- Create: `experiments/alfworld/agent.py`
- Create: `tests/alfworld/test_agent.py`

ReAct-style agent: receives obs + admissible_commands + retrieved memories, calls LLM to select action.

## Task 5: Experiment Runner

**Files:**
- Create: `experiments/alfworld/runner.py`
- Create: `scripts/alfworld/run_experiment.py`

Multi-epoch loop: for each epoch, for each task -> retrieve memories, agent interacts with env, update Q-values, store new experience.

## Task 6: Run Experiment and Analyze

**Files:**
- Create: `scripts/alfworld/analyze_results.py`

## Task 7: Visualization

**Files:**
- Create: `scripts/alfworld/plot_results.py`
