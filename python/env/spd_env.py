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
from pathlib import Path
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

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        jar_path: str,
        capture_region: Tuple[int, int, int, int],
        grid_size: Tuple[int, int] = (32, 64),
        frame_skip: int = 4,
        frame_stack: int = 4,
        grayscale_size: Tuple[int, int] = (84, 84),
        highlight_duration: int = 5,          # ← 新增：邊框維持的影格數
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
        self.highlight_duration = highlight_duration
        self._highlight_counter: int = 0      # 還要畫幾張影格
        self._last_click: Optional[Tuple[int, int]] = None  # 最近一次點擊的 (row, col)
        self.prev_lvl  = 1
        self.prev_exp  = 0
        self.prev_gold = 0
        self.prev_depth = 1

        # Discrete actions correspond to clicking on a tile in the grid.
        self.action_space = spaces.Discrete(grid_size[0] * grid_size[1])
        # Observation is a stack of grayscale frames.
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(grayscale_size[0], grayscale_size[1], frame_stack), dtype=np.uint8
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
        """Start the SPD desktop jar in --ai-mode and keep stdin alive."""
        if self.proc is not None:
            self.proc.terminate()
            self.proc.wait()
        ROOT = Path(__file__).resolve().parents[2]   # spd-ai 根
        jar_path = ROOT / "game" / "shattered-pixel-dungeon" / "desktop" / "build" / "libs" / "desktop-3.2.0.jar"
        
        cmd = [
            "java",
            "--add-opens", "java.base/java.lang=ALL-UNNAMED",  # 👈 一定要帶
            "-jar", str(jar_path),
            "--ai-mode"
        ]
        
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,          # 建議暫存起來以便偵錯
            stderr=subprocess.STDOUT,
            text=True,                       # 讓 .stdin / .stdout 都收發字串
            bufsize=1,                       # line-buffered
            #creationflags=subprocess.CREATE_NO_WINDOW  # ← 無視窗但保留 stdio
        )

        # 等視窗穩定＋檢查是否已經 crash
        time.sleep(2.0)
        if self.proc.poll() is not None:     # 子行程已經退出
            output = self.proc.stdout.read() if self.proc.stdout else ""
            raise RuntimeError(
                f"SPD jar terminated early with exit code {self.proc.returncode}\n{output}"
            )

    def _send_click(self, row:int, col:int):
        l, t, w, h = self.capture_region
        tx, ty = w / self.grid_size[1], h / self.grid_size[0]
        self._send({"x": int((col+0.5)*tx), "y": int((row+0.5)*ty), "button": 0})

    def _ensure_proc(self) -> None:
        """Start the subprocess if it's not already running."""
        if self.proc is None or self.proc.poll() is not None:
            self._start_proc()

    def _capture_frame(self) -> np.ndarray:
        """Capture a grayscale frame and optionally draw the highlighted cell."""
        left, top, width, height = self.capture_region
        screenshot = self.sct.grab({"left": left, "top": top, "width": width, "height": height})
        img = np.asarray(screenshot)[:, :, :3]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, self.grayscale_size, interpolation=cv2.INTER_AREA)

        # 如果需要，把點擊格子畫出邊框
        if self._highlight_counter > 0 and self._last_click is not None:
            r, c = self._last_click
            tile_h = self.grayscale_size[0] / self.grid_size[0]
            tile_w = self.grayscale_size[1] / self.grid_size[1]
            # 取整以免邊框模糊
            y1 = int(r * tile_h)
            y2 = int((r + 1) * tile_h) - 1
            x1 = int(c * tile_w)
            x2 = int((c + 1) * tile_w) - 1
            # 使用白色單像素邊框
            cv2.rectangle(resized, (x1, y1), (x2, y2), color=255, thickness=1)
            self._highlight_counter -= 1

        return resized

    def _send(self, obj: dict):
        if self.proc and self.proc.stdin:
            self.proc.stdin.write(json.dumps(obj) + "\n")
            self.proc.stdin.flush()

    def _query_state(self, timeout=1.0) -> dict:
        if not (self.proc and self.proc.stdin and self.proc.stdout):
            return {}

        self._send({"cmd": "get_state"})

        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                break                      # 子行程掛了？
            if line.startswith("##STATE##"):
                try:
                    return json.loads(line[len("##STATE##"):])
                except json.JSONDecodeError as e:
                    print("⚠️  JSON parse error:", e, line.strip())
                    return {}
            # 其餘行視為雜訊，忽略
        return {}

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self._ensure_proc()
        self._send({"cmd": "reset"})
        time.sleep(2.0)  # 等待遊戲跳到第一層
        self._frames = []
        self.prev_lvl  = 1
        self.prev_exp  = 0
        self.prev_gold = 0
        self.prev_depth = 1
        first_frame = first = self._grab_gray()
        for _ in range(self.frame_stack):
            self._frames.append(first_frame)
        return self._obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Send a mouse click action, wait, capture frames, and return observation and reward."""
        # Compute tile coordinates from action index
        r, c = divmod(action, self.grid_size[1])
        self._send_click(r, c)
        # print(f"[Click]: ({row}, {col})")
        # Skip a few frames to allow the game to update
        for _ in range(self.frame_skip):
            time.sleep(0.05)
            self._frames.append(self._grab_gray())
            if len(self._frames) > self.frame_stack:
                self._frames.pop(0)

        # TODO: Implement proper reward and termination detection by parsing game output or using OCR.
        state = self._query_state()
        if not state:
            return self._obs(), 0.0, False, False, {}
        
        # experience gain
        xp_now  = state["lvl"]*100 + state["exp"]
        xp_prev = self.prev_lvl*100 + self.prev_exp
        reward  = xp_now - xp_prev

        # gold & depth bonus
        reward += 0.1*(state["gold"] - self.prev_gold)
        if state["depth"] > self.prev_depth:
            reward += 50

        # death penalty
        alive = state.get("alive", True)
        if not alive:
            reward -= 100

        terminated = not alive
        truncated = False

        # update baselines
        self.prev_lvl   = state["lvl"]
        self.prev_exp   = state["exp"]
        self.prev_gold  = state["gold"]
        self.prev_depth = state["depth"]

        info: Dict[str, Any] = {}

        return self._obs(), reward, terminated, truncated, info

    def render(self, mode: str = "human") -> None:
        if mode != "human":
            raise NotImplementedError
        if not self._frames:
            return
        frame = self._frames[-1]        # 取最近一張 (含白框)
        bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # 放大顯示，例如放大 4 倍
        scale = 4
        enlarged = cv2.resize(
            bgr,
            (bgr.shape[1] * scale, bgr.shape[0] * scale),
            interpolation=cv2.INTER_NEAREST  # 保留像素感
        )

        cv2.imshow("SPDEnv", enlarged)
        cv2.waitKey(1)

    def close(self) -> None:
        """Terminate the subprocess and release resources."""
        if self.proc:
            self.proc.terminate()
            self.proc.wait()
            self.proc = None
        self.sct.close()
        cv2.destroyAllWindows()         # ← 確保視窗關閉

    def _grab_gray(self):
        l, t, w, h = self.capture_region
        img = np.asarray(self.sct.grab({"left": l, "top": t, "width": w, "height": h}))[:,:,:3]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, self.grayscale_size, interpolation=cv2.INTER_AREA)

    def _obs(self):
        # 堆疊在最後一維 → (H, W, C)
        return np.stack(self._frames, axis=-1).astype(np.uint8)