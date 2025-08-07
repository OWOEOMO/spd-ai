# train/train_rnd.py
import torch as th
import torch.nn as nn
import numpy as np
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper

from env.spd_env import SPDEnv

# ── 0. 路徑設定 ──────────────────────────────────────────
JAR_PATH       = str(Path("../game/shattered-pixel-dungeon/desktop/build/libs/desktop-3.2.0.jar").resolve())
CAPTURE_REGION = (1558, 422, 720, 1220)
TOTAL_STEPS    = 1_000_000

# ── 1. 極簡 RND wrapper ─────────────────────────────────
class _RNDTarget(nn.Module):
    def __init__(self, obs_shape, out_dim=512):
        super().__init__()
        c, h, w = obs_shape
        self.net = nn.Sequential(
            nn.Conv2d(c, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32,64,4,2),   nn.ReLU(),
            nn.Conv2d(64,64,3,1),   nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * ((h-20)//8) * ((w-20)//8), out_dim)
        )
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x): return self.net(x/255.)

class _RNDPredictor(nn.Module):
    def __init__(self, obs_shape, out_dim=512):
        super().__init__()
        c, h, w = obs_shape
        self.net = nn.Sequential(
            nn.Conv2d(c, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32,64,4,2),   nn.ReLU(),
            nn.Conv2d(64,64,3,1),   nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * ((h-20)//8) * ((w-20)//8), 512), nn.ReLU(),
            nn.Linear(512, out_dim)
        )

    def forward(self, x): return self.net(x/255.)

class RNDReward(VecEnvWrapper):
    """給定 obs，計算 (target-predictor)^2 當作 intrinsic reward。"""
    def __init__(self, venv, int_coef=1.0, lr=1e-4):
        super().__init__(venv)
        obs_shape = self.observation_space.shape   # (C, H, W)
        self.int_coef = int_coef

        self.target    = _RNDTarget(obs_shape).to("cuda" if th.cuda.is_available() else "cpu")
        self.predictor = _RNDPredictor(obs_shape).to("cuda" if th.cuda.is_available() else "cpu")
        self.opt       = th.optim.Adam(self.predictor.parameters(), lr=lr)
        self.mse       = nn.MSELoss(reduction="none")

    def step_wait(self):
        obs, ext_rew, done, info = self.venv.step_wait()

        # --- RND intrinsic ---
        with th.no_grad():
            tgt_feat = self.target(th.from_numpy(obs).float())
        pred_feat = self.predictor(th.from_numpy(obs).float())
        int_rew   = (self.mse(pred_feat, tgt_feat).mean(dim=1).cpu().numpy())

        # 更新 predictor
        loss = int_rew.mean()
        self.opt.zero_grad(); loss.backward(); self.opt.step()

        rew = ext_rew + self.int_coef * int_rew
        return obs, rew, done, info

# ── 2. VecEnv 工廠 ─────────────────────────────────────
def make_env(int_coef=1.0):
    def _init():
        env = SPDEnv(jar_path=JAR_PATH, capture_region=CAPTURE_REGION)
        env = VecTransposeImage(env)                # (H,W,C) → (C,H,W)
        env = VecFrameStack(env, n_stack=4)         # time-stack
        env = RNDReward(env, int_coef=int_coef)     # intrinsic
        return env
    return _init

# ── 3. 兩階段訓練 ───────────────────────────────────────
if __name__ == "__main__":
    # ① 只用 intrinsic reward 探索
    vec_env = DummyVecEnv([make_env(int_coef=1.0)])
    model = PPO("CnnPolicy", vec_env,
                learning_rate=2.5e-4, n_steps=2048, batch_size=64,
                n_epochs=4, gamma=0.99, verbose=1)
    model.learn(TOTAL_STEPS)
    model.save("spd_ppo_stage1")

    # ② intrinsic 0.5 + extrinsic 1.0（外部獎用 env 內建）
    vec_env2 = DummyVecEnv([make_env(int_coef=0.5)])
    model.set_env(vec_env2)
    model.learn(500_000)
    model.save("spd_ppo_stage2")
