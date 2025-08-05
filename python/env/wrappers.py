"""Custom wrappers for SPDEnv to provide intrinsic and extrinsic rewards.

This module defines two example wrappers:

- `RNDReward` – implements a basic random network distillation (RND) intrinsic
  reward signal.  You will need to provide your own implementation or use an
  existing library.
- `ExtrinsicWrapper` – adds external rewards based on the game’s score
  increment or other task‑specific signals.

These wrappers should be composed around `SPDEnv` when training the agent.
"""

from __future__ import annotations
from typing import Any, Dict, Tuple

import gymnasium as gym


class RNDReward(gym.Wrapper):
    """Placeholder wrapper to add intrinsic curiosity rewards using RND."""
    def __init__(self, env: gym.Env, int_coef: float = 1.0) -> None:
        super().__init__(env)
        self.int_coef = int_coef
        # TODO: initialise your RND networks here

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        # TODO: compute intrinsic reward based on observation
        intrinsic_reward = 0.0
        reward = reward + self.int_coef * intrinsic_reward
        return obs, reward, terminated, truncated, info


class ExtrinsicWrapper(gym.Wrapper):
    """Placeholder wrapper to add external rewards based on game score."""
    def __init__(self, env: gym.Env, ext_coef: float = 1.0) -> None:
        super().__init__(env)
        self.ext_coef = ext_coef
        self._last_score: float = 0.0

    def reset(self, **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        # Reset internal score tracking
        self._last_score = info.get("score", 0.0)
        return obs, info

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Extract the current score from info and compute the difference
        current_score = info.get("score", self._last_score)
        delta = current_score - self._last_score
        self._last_score = current_score
        reward = reward + self.ext_coef * delta
        return obs, reward, terminated, truncated, info
