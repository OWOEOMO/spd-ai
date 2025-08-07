"""
簡易 wrappers：
* RNDReward: 內部隨機網路探索 (very light)  
* ExtrinsicWrapper: 將 env 原有 reward 乘上係數
"""

from __future__ import annotations
import numpy as np
import gymnasium as gym
from sb3_contrib.common.reward_norm import RewardNormalizer  # 內建 wrapper
from sb3_contrib.common.vec_env import VecNormalize         # ← 只是示例


# ---- RND --------------------------------------------------------------------
class _SimpleRNDModel:
    """極簡版 RND：線性層 + ReLU，不做訓練，只算 L2 損失。"""
    def __init__(self, obs_shape):
        H, W, C = obs_shape
        feat_dim = 128
        rng = np.random.default_rng(123)
        self.W = rng.standard_normal((H*W*C, feat_dim)).astype(np.float32)

    def __call__(self, obs: np.ndarray):
        flat = obs.reshape(obs.shape[0], -1) / 255.0
        return np.maximum(flat @ self.W, 0.0)

class RNDReward(gym.Wrapper):
    def __init__(self, env, int_coef: float = 1.0):
        super().__init__(env)
        # 用 sb3-contrib 內建 RND 探索器
        from sb3_contrib.common.exploration.rnd import RNDModel
        self.rnd = RNDModel(env.observation_space, env.action_space)
        self.int_coef = int_coef

    def step(self, action):
        obs, ext_r, done, trunc, info = self.env.step(action)
        int_r = self.rnd.reward(obs)
        return obs, ext_r + self.int_coef * int_r, done, trunc, info
    
# ---- Extrinsic --------------------------------------------------------------
class ExtrinsicWrapper(gym.Wrapper):
    def __init__(self, env, ext_coef=1.0):
        super().__init__(env); self.ext_coef = float(ext_coef)

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        return obs, self.ext_coef * r, term, trunc, info
