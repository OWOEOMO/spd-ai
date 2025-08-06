"""
Gymnasium environment wrapper for Shattered Pixel Dungeon using mouse-only control.

This module implements the SPDEnv class used to interface between the Java game (running in ai-mode)
and reinforcement learning algorithms. It spawns the game as a subprocess, sends mouse clicks based
on discrete actions, captures frames from the screen using mss, and returns stacked grayscale
observations.

Before using this environment, make sure to start the Shattered Pixel Dungeon jar with the --ai-mode
flag and the FPS limits disabled. Provide the jar path and screen capture region when constructing
the environment.
"""

from __future__ import annotations
from typing import Any, Dict, Tuple, List, Optional

import subprocess
import time
import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# External dependencies for screen capture and image processing.
try:
    import mss  # type: ignore
    import cv2  # type: ignore
except ImportError:
    mss = None
    cv2 = None


class SPDEnv(gym.Env):
    """Environment for controlling Shattered Pixel Dungeon via mouse clicks."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        jar_path: str,
        capture_region: Tuple[int, int, int, int],
        grid_size: Tuple[int, int] = (18, 32),
        frame_skip: int = 4,
        frame_stack: int = 4,
        grayscale_size: Tuple[int, int] = (84, 84),
    ) -> None:
        """
        Initialize the environment.

        Args:
            jar_path: Path to the compiled desktop jar (e.g. desktop-3.2.0.jar).
            capture_region: (left, top, width, height) tuple describing the region of the screen
                to capture for observations. You can find this by manually positioning the game
                window and measuring its location.
            grid_size: Number of tile rows and columns; actions will map to clicking the centre
                of these tiles.
            frame_skip: Number of game frames to skip between actions.
            frame_stack: Number of recent grayscale frames to stack as the observation.
            grayscale_size: Size to downsample each frame to (height, width).
        """
        super().__init__()
        self.jar_path = jar_path
        self.capture_region = capture_region
        self.grid_size = grid_size
        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        self.grayscale_size = grayscale_size

        # Discrete actions correspond to clicking on a tile in the grid.
        self.action_space = spaces.Discrete(grid_size[0] * grid_size[1])
        # Observation is a stack of grayscale frames.
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(frame_stack, grayscale_size[0], grayscale_size[1]), dtype=np.uint8
        )

        # Frame buffer for stacking
        self._frames: List[np.ndarray] = []

        # Process handle
        self.proc: Optional[subprocess.Popen] = None

        # Screen capture context; ensure dependencies are present
        if mss is None or cv2 is None:
            raise ImportError("Both 'mss' and 'opencv-python' must be installed to use SPDEnv.")
        self.sct = mss.mss()

    def _start_proc(self) -> None:
        """Start the game subprocess in ai-mode."""
        # Terminate existing process if running
        if self.proc is not None:
            self.proc.terminate()
            self.proc.wait()
            self.proc = None
        cmd = ["java", "-jar", self.jar_path, "--ai-mode"]
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # Allow time for the window to open
        time.sleep(3.0)
        # Drain any initial output
        if self.proc.stdout:
            try:
                while True:
                    line = self.proc.stdout.readline()
                    if not line:
                        break
            except Exception:
                pass

    def _send_click(self, row: int, col: int) -> None:
        """Send a click at the specified tile row and column via JSON over stdin."""
        left, top, width, height = self.capture_region
        tile_w = width / self.grid_size[1]
        tile_h = height / self.grid_size[0]
        x = int(left + (col + 0.5) * tile_w)
        y = int(top + (row + 0.5) * tile_h)
        cmd_dict = {"x": x, "y": y, "button": 0}
        if self.proc and self.proc.stdin:
            self.proc.stdin.write(json.dumps(cmd_dict) + "\n")
            self.proc.stdin.flush()

    def _send_reset_command(self) -> None:
        """Send a reset command to the Java process."""
        if self.proc and self.proc.stdin:
            self.proc.stdin.write(json.dumps({"cmd": "reset"}) + "\n")
            self.proc.stdin.flush()

    def _ensure_proc(self) -> None:
        """Start the subprocess if it's not already running."""
        if self.proc is None or self.proc.poll() is not None:
            self._start_proc()

    def _capture_frame(self) -> np.ndarray:
        """Capture a grayscale frame from the specified screen region and downsample it."""
        left, top, width, height = self.capture_region
        screenshot = self.sct.grab({"left": left, "top": top, "width": width, "height": height})
        img = np.asarray(screenshot)[:, :, :3]  # Drop alpha channel
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, self.grayscale_size, interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self._ensure_proc()
        self._send_reset_command()
        time.sleep(2.0)  # 等待遊戲跳到第一層
        self._frames = []
        first_frame = self._capture_frame()
        for _ in range(self.frame_stack):
            self._frames.append(first_frame)
        return np.stack(self._frames, axis=0), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Send a mouse click action, wait, capture frames, and return observation and reward."""
        # Compute tile coordinates from action index
        row = action // self.grid_size[1]
        col = action % self.grid_size[1]
        self._send_click(row, col)
        # Skip a few frames to allow the game to update
        for _ in range(self.frame_skip):
            time.sleep(0.05)
            frame = self._capture_frame()
            self._frames.append(frame)
            if len(self._frames) > self.frame_stack:
                self._frames.pop(0)
        obs = np.stack(self._frames, axis=0)
        # TODO: Implement proper reward and termination detection by parsing game output or using OCR.
        reward = 0.0
        terminated = False
        truncated = False
        info: Dict[str, Any] = {}
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        """Terminate the subprocess and release resources."""
        if self.proc:
            self.proc.terminate()
            self.proc.wait()
            self.proc = None
        self.sct.close()
