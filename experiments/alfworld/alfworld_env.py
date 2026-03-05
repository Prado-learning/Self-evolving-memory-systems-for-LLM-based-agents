"""ALFWorld environment wrapper with simple interface."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

ALFWORLD_DATA = os.environ.get("ALFWORLD_DATA", "/home/user/alfworld_data")

TASK_TYPE_PREFIXES = [
    "pick_and_place_simple",
    "pick_and_place_with_movable_recep",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
]


def get_task_type(game_name: str) -> str:
    for prefix in TASK_TYPE_PREFIXES:
        if game_name.startswith(prefix):
            return prefix
    return "unknown"


def make_alfworld_config(data_dir: str = ALFWORLD_DATA) -> dict:
    return {
        "general": {"training_method": "dagger"},
        "env": {
            "goal_desc_human_anns_prob": 0,
            "task_types": [1, 2, 3, 4, 5, 6, 7],
            "domain_randomization": False,
            "expert_type": "handcoded",
        },
        "dataset": {
            "data_path": os.path.join(data_dir, "json_2.1.1", "train"),
            "eval_id_data_path": os.path.join(data_dir, "json_2.1.1", "valid_seen"),
            "eval_ood_data_path": os.path.join(data_dir, "json_2.1.1", "valid_unseen"),
            "num_eval_games": -1,
        },
        "logic": {
            "domain": os.path.join(data_dir, "logic", "alfred.pddl"),
            "grammar": os.path.join(data_dir, "logic", "alfred.twl2"),
        },
        "dagger": {"training": {"max_nb_steps_per_episode": 50}},
    }


class ALFWorldEnv:
    def __init__(self, split: str = "eval_in_distribution", data_dir: str = ALFWORLD_DATA):
        os.environ["ALFWORLD_DATA"] = data_dir
        from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

        config = make_alfworld_config(data_dir)
        self._tw_env = AlfredTWEnv(config, train_eval=split)
        self._gym_env = self._tw_env.init_env(batch_size=1)
        self.num_games = len(self._tw_env.game_files)
        self._current_game_name: str = ""
        self._current_task_type: str = ""
        self._step_count: int = 0
        self._max_steps: int = 50

    @property
    def game_files(self) -> List[str]:
        return self._tw_env.game_files

    def reset(self, game_idx: Optional[int] = None) -> Tuple[str, str, str, List[str]]:
        """Reset to a new game. Returns (obs, task_type, game_name, admissible_commands)."""
        obs, info = self._gym_env.reset()
        self._step_count = 0

        # Extract game name from the file path
        game_file = info.get("extra.gamefile", [None])[0]
        if game_file:
            self._current_game_name = os.path.basename(
                os.path.dirname(os.path.dirname(game_file))
            )
        else:
            # Fallback: use the game file from tw_env
            idx = self._tw_env.random_indices[0] if hasattr(self._tw_env, "random_indices") else 0
            gf = self._tw_env.game_files[idx]
            self._current_game_name = os.path.basename(os.path.dirname(os.path.dirname(gf)))

        self._current_task_type = get_task_type(self._current_game_name)
        admissible = info.get("admissible_commands", [[]])[0]

        return obs[0], self._current_task_type, self._current_game_name, admissible

    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        """Take an action. Returns (obs, reward, done, info)."""
        self._step_count += 1
        obs, scores, dones, infos = self._gym_env.step([action])

        won = infos.get("won", [False])[0]
        done = dones[0] or self._step_count >= self._max_steps
        reward = 1.0 if won else 0.0
        admissible = infos.get("admissible_commands", [[]])[0]

        return obs[0], reward, done, {
            "won": won,
            "admissible_commands": admissible,
            "steps": self._step_count,
        }

    def close(self):
        self._gym_env.close()
