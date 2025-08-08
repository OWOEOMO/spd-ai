"""
Basic smoke-test for SPDEnv.

Run with:
    python -m pytest -q python/test/smoke.py
or simply
    python -m test.smoke
"""

from pathlib import Path
import os
import numpy as np
import pytest

from env.spd_env import SPDEnv

# ──★ 0. 可配置參數 ──────────────────────────────
JAR_PATH = Path("E:\GithubClone\spd-ai\game\shattered-pixel-dungeon\desktop\build\libs\desktop-3.2.0.jar").resolve()
CAPTURE_REGION = (1558, 422, 720, 1220)      # 依你的螢幕位置調整
N_STEPS = 100
RENDER_EVERY = 5
PRINT_REWARD_EVERY = 10                      # 每幾步印一次 reward（可調整）

# ──★ 1. helper：逐回合執行，必要時自動 reset ─────
def run_episode(env, n_steps=N_STEPS):
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray), "reset() should return an ndarray"
    assert obs.shape == env.observation_space.shape

    total_reward = 0.0

    for step in range(n_steps):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)

        total_reward += reward
        if step % PRINT_REWARD_EVERY == 0:
            print(f"[{step:03}] reward = {reward:6.2f} | total = {total_reward:7.2f}")

        if step % RENDER_EVERY == 0:
            env.render()

        if done or truncated:
            print(f"🚫 Episode ended (done={done}, truncated={truncated})")
            print(f"🔁 Resetting. Episode total reward: {total_reward:.2f}\n")
            obs, info = env.reset()
            total_reward = 0.0


# ──★ 2. PyTest 版本 ─────────────────────────────
@pytest.mark.smoke
def test_spd_env_smoke():
    env = SPDEnv(jar_path=str(JAR_PATH), capture_region=CAPTURE_REGION)
    try:
        run_episode(env)
    finally:
        env.close()


# ──★ 3. 手動執行（python -m test.smoke）──────────
if __name__ == "__main__":
    env = SPDEnv(jar_path=str(JAR_PATH), capture_region=CAPTURE_REGION)
    try:
        run_episode(env)
    finally:
        env.close()
    print("✅  Smoke test finished without error")
