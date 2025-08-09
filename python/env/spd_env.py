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
        frame_skip: int = 1,
        frame_stack: int = 4,
        tick: float = 0.004,
        grayscale_size: Tuple[int, int] = (84, 84),
        highlight_duration: int = 5,          # ← 新增：邊框維持的影格數
        forbidden_rects_norm=None, 
        remap_forbidden=True,
        extra_java_args: Optional[List[str]] = None,
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
        self.tick = tick
        self.grayscale_size = grayscale_size
        self.highlight_duration = highlight_duration
        self._highlight_counter: int = 0      # 還要畫幾張影格
        self._last_click: Optional[Tuple[int, int]] = None  # 最近一次點擊的 (row, col)
        self.prev_lvl  = 1
        self.prev_exp  = 0
        self.prev_gold = 0
        self.prev_depth = 1
        self.prev_explored = 0
        self.prev_alive = True
        
        self.forbidden_rects_norm = forbidden_rects_norm or []  # [(x1,y1,x2,y2)] in [0,1]
        self.remap_forbidden = remap_forbidden
        self._forbidden_actions = set()
        self._nearest_safe = {}   # a -> safe_a
        self._build_deadzone()    # ← 初始化後計算一次
        self.extra_java_args = extra_java_args or []
        
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
        
    def _build_deadzone(self):
        """把右上角禁點區轉成 action 索引集合，並建立最近安全格對應。"""
        H, W = self.grid_size  # row, col
        self._forbidden_actions.clear()
        centers = []  # 每個 action 的像素中心（視窗座標，不含 capture_region 偏移）

        # 每格中心在「視窗座標」的 x,y（我們傳給 Java 的就是視窗座標）
        for r in range(H):
            for c in range(W):
                # 以 0~1 正規化座標來判斷是否落在禁區
                xn = (c + 0.5) / W
                yn = (r + 0.5) / H
                action = r * W + c
                for (x1, y1, x2, y2) in self.forbidden_rects_norm:
                    if x1 <= xn <= x2 and y1 <= yn <= y2:
                        self._forbidden_actions.add(action)
                        break
                centers.append((c, r))  # grid 空間座標，後面算最近安全格用

        # 建立最近安全格映射
        safe_actions = [a for a in range(H*W) if a not in self._forbidden_actions]
        for a in range(H*W):
            if a in self._forbidden_actions:
                # 找最近的安全格（用格子距離）
                ac, ar = centers[a]
                best = min(safe_actions, key=lambda b: (centers[b][0]-ac)**2 + (centers[b][1]-ar)**2)
                self._nearest_safe[a] = best
            
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
        cmd.extend(self.extra_java_args)
        
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
        # 若這格是禁點，就依設定改送最近安全格 or 直接忽略
        a = row * self.grid_size[1] + col
        if a in self._forbidden_actions:
            if self.remap_forbidden:
                a = self._nearest_safe[a]
                row, col = divmod(a, self.grid_size[1])
            else:
                # 直接不送點擊
                return
        # 把 row/col 轉成「視窗座標」像素點（不加 capture_region 偏移）
        _, _, w, h = self.capture_region
        tx, ty = w / self.grid_size[1], h / self.grid_size[0]
        x = int((col + 0.5) * tx)
        y = int((row + 0.5) * ty)
        self._send({"x": x, "y": y, "button": 0})

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
            
        if self.forbidden_rects_norm:
            Hn, Wn = self.grayscale_size  # 84x84 之類
            for (x1,y1,x2,y2) in self.forbidden_rects_norm:
                x1p, y1p = int(x1 * Wn), int(y1 * Hn)
                x2p, y2p = int(x2 * Wn), int(y2 * Hn)
                cv2.rectangle(resized, (x1p, y1p), (x2p, y2p), color=200, thickness=1)
                
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
        self.prev_explored = 0
        self.prev_alive = True

        first_frame = first = self._grab_gray()
        for _ in range(self.frame_stack):
            self._frames.append(first_frame)
        return self._obs(), {}

    def step(self, action: int):
        r, c = divmod(action, self.grid_size[1])
        self._send_click(r, c)

        for _ in range(self.frame_skip):
            time.sleep(self.tick)
            self._frames.append(self._grab_gray())
            if len(self._frames) > self.frame_stack:
                self._frames.pop(0)

        state = self._query_state()
        if not state:
            return self._obs(), 0.0, False, False, {}
        
        alive = state.get("alive", True)
        death_this_step = (self.prev_alive and not alive)  # 這一回合剛死
        
        # ---- 拆解外部獎勵 ----
        xp_now  = state["lvl"]*100 + state["exp"]
        xp_prev = self.prev_lvl*100 + self.prev_exp
        xp_delta = xp_now - xp_prev

        gold_delta = state["gold"] - self.prev_gold
        depth_up = state["depth"] > self.prev_depth

        explored = state.get("explored", -1)
        if explored >= 0 and state["depth"] != self.prev_depth:
            # 跨樓層重新基準，避免上一層殘值造成突刺
            self.prev_explored = 0

        # 只在「還活著而且不是剛死的這一步」才計入探索獎勵
        explored_delta_raw = max(0, explored - self.prev_explored) if (explored >= 0) else 0
        if (not alive) or death_this_step:
            explored_delta = 0
        else:
            # 加步進上限，避免一次爆量
            explored_delta = explored_delta_raw #min(explored_delta_raw, 8)   # 每步最多 8 格，可自行調整

        # 死亡懲罰（只在死亡那一回合給，之後就 done 了）
        death_pen = 100 if death_this_step else 0

        # 這裡是「外部」獎勵（intrinsic 另算）
        ext_reward = (
            xp_delta
            + 0.1 * gold_delta
            + (50 if depth_up else 0)
            + 0.3 * explored_delta
            - death_pen
        )

        info = {
            "ext_reward": float(ext_reward),
            "ext_breakdown": {
                "xp":        int(xp_delta),
                "gold":      int(gold_delta),
                "explored":  int(explored_delta),      # 注意：這裡是「clamp 後」的值
                "depth_up":  bool(depth_up),
                "dead":      bool(death_this_step),
            }
        }
        
        terminated = not alive
        truncated = False
        
        # 更新 baseline
        self.prev_lvl   = state["lvl"]
        self.prev_exp   = state["exp"]
        self.prev_gold  = state["gold"]
        self.prev_depth = state["depth"]
        self.prev_alive = alive
        if explored >= 0:
            self.prev_explored = explored


        return self._obs(), float(ext_reward), terminated, truncated, info


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