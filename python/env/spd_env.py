"""Gymnasium environment wrapper for Shattered Pixel Dungeon.

This module contains a minimal skeleton of the `SPDEnv` class used to
interface between the Java game (running in headless mode) and the
reinforcement learning algorithms.  You will need to implement methods to
launch the game process, send actions over a socket or pipe, and receive
observations (e.g. pixel arrays) and additional info from the game.

The environment should follow the Gymnasium API: `reset()` to start a new
episode and `step(action)` to advance the game by one action.  It should
return observations, rewards, termination flags and any extra information.

"""

from __future__ import annotations
from typing import Any, Dict, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SPDEnv(gym.Env):
    """A placeholder environment for the Shattered Pixel Dungeon agent.

    This class should manage a connection to the headless Java game.  At
    each step it should send the chosen action to the game, receive the
    rendered frame and any game state variables, assemble them into an
    observation, compute rewards and determine whether the episode has ended.
    """

    metadata = {"render_modes": []}

    def __init__(self, obs_shape: Tuple[int, int] | None = None, action_space_size: int = 8, **kwargs: Any) -> None:
        super().__init__()
        # Observation space: by default four stacked grayscale frames of size 128×128
        if obs_shape is None:
            obs_shape = (4, 128, 128)
        self.observation_space = spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
        # Action space: by default eight discrete actions (move/attack/pick/etc.)
        self.action_space = spaces.Discrete(action_space_size)
        # Internal buffers for frame stacking
        self._frames: np.ndarray | None = None
        # TODO: Initialise connection to the headless game here (e.g. via ZeroMQ or sockets)
        self._conn = None

    def reset(self, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Start a new episode.

        This method should reset the game to a fresh state, reinitialise any
        frame buffers and return the initial observation.  The dictionary in
        the second return value can include any auxiliary data.
        """
        super().reset(seed=seed)
        # TODO: start or reset the game process here
        # For now return a zero observation
        obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
        info: Dict[str, Any] = {}
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advance the environment by one action.

        Send the action to the game, receive the next frame and any other
        information, compute the reward, and determine whether the episode
        has terminated or been truncated.  The `info` dictionary can
        include raw score, health or other diagnostics.
        """
        # TODO: send the action to the Java process and receive the response
        # Placeholder: always return zero observation and zero reward
        obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
        reward: float = 0.0
        terminated: bool = False
        truncated: bool = False
        info: Dict[str, Any] = {}
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        """Clean up resources."""
        # TODO: close the connection to the game
        pass
