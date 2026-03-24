#!/usr/bin/env python3
"""Probe ALFWorld to get game index → task_type mapping."""
import sys, random, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.alfworld.alfworld_env import ALFWorldEnv

env = ALFWorldEnv(split='eval_in_distribution')
print(f"Total games: {env.num_games}", flush=True)

type_to_indices = {}
all_games = []

for i in range(env.num_games):
    obs, task_type, game_name, _ = env.reset()
    type_to_indices.setdefault(task_type, []).append(i)
    all_games.append({'idx': i, 'task_type': task_type, 'game_name': game_name})

env.close()

print("\n=== Task type distribution ===")
for tt, idxs in sorted(type_to_indices.items()):
    print(f"  {tt}: {len(idxs)} games  indices sample: {idxs[:5]}")

# Save mapping
out = ROOT / 'outputs' / 'alfworld' / 'game_type_map.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({'type_to_indices': type_to_indices, 'all_games': all_games}, indent=2))
print(f"\nSaved to {out}")
