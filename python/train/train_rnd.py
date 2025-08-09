# train/train_rnd.py
import time  # ★
import torch as th
import torch.nn as nn
import numpy as np
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper
from stable_baselines3.common.callbacks import BaseCallback  # ★
from stable_baselines3.common.monitor import Monitor        # ★
from stable_baselines3.common.vec_env import SubprocVecEnv

from env.spd_env import SPDEnv
from colorama import init as colorama_init, Fore, Style
colorama_init(autoreset=True)

# ── 0. 路徑/裝置設定 ─────────────────────────────────────
JAR_PATH       = str(Path("../game/shattered-pixel-dungeon/desktop/build/libs/desktop-3.2.0.jar").resolve())
CAPTURE_REGION = (1558, 422, 720, 1220)
TOTAL_STEPS    = 1_000_000

DEVICE = "cuda" if th.cuda.is_available() else "cpu"  # ★
th.backends.cudnn.benchmark = True                    # ★ 固定尺寸影像時提升卷積效能
if hasattr(th, "set_float32_matmul_precision"):       # ★ Ampere+：開 TF32
    th.set_float32_matmul_precision("high")

# ── 1. 極簡 RND wrapper ─────────────────────────────────
class _RNDTarget(nn.Module):
    def __init__(self, obs_shape, out_dim=512):
        super().__init__()
        c, h, w = obs_shape        # (C,84,84)
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, 8, 4), nn.ReLU(),   # 84→20
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),  # 20→9
            nn.Conv2d(64, 64, 3, 1), nn.ReLU()   # 9 →7
        )
        with th.no_grad():
            n_flat = self.cnn(th.zeros(1, c, h, w)).view(1, -1).shape[1]  # 3136
        self.fc = nn.Linear(n_flat, out_dim)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        x = x / 255.0
        x = self.cnn(x)
        return self.fc(x.flatten(1))

class _RNDPredictor(nn.Module):
    def __init__(self, obs_shape, out_dim=512):
        super().__init__()
        c, h, w = obs_shape
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
        )
        with th.no_grad():
            n_flat = self.cnn(th.zeros(1, c, h, w)).view(1, -1).shape[1]
        self.head = nn.Sequential(
            nn.Linear(n_flat, 512), nn.ReLU(),
            nn.Linear(512, out_dim)
        )

    def forward(self, x):
        x = x / 255.0
        x = self.cnn(x)
        return self.head(x.flatten(1))

from colorama import init as colorama_init, Fore, Style
colorama_init(autoreset=True)

class RNDReward(VecEnvWrapper):
    def __init__(self, venv, int_coef=1.0, lr=1e-4,
                 print_each_reward: bool = True,
                 min_abs_total: float = 1e-3,      # 總獎勵小於此值就不印
                 min_abs_int: float = 5e-3):       # 內在獎勵小於此值就不印
        super().__init__(venv)
        obs_shape = self.observation_space.shape
        self.int_coef = int_coef

        self.target    = _RNDTarget(obs_shape).to(DEVICE)
        self.predictor = _RNDPredictor(obs_shape).to(DEVICE)
        self.opt       = th.optim.Adam(self.predictor.parameters(), lr=lr)
        self.mse       = nn.MSELoss(reduction="none")

        self.last_loss = 0.0
        self.print_each_reward = print_each_reward
        self.min_abs_total = min_abs_total
        self.min_abs_int   = min_abs_int
        self._t = 0  # global step counter

    def _cnum(self, v: float, digits=3):
        if v > 0:   return f"{Fore.GREEN}+{v:.{digits}f}{Style.RESET_ALL}"
        if v < 0:   return f"{Fore.RED}{v:.{digits}f}{Style.RESET_ALL}"
        return f"{v:.{digits}f}"

    def _cint(self, v: int):
        if v > 0:   return f"{Fore.GREEN}+{v}{Style.RESET_ALL}"
        if v < 0:   return f"{Fore.RED}{v}{Style.RESET_ALL}"
        return f"{v}"

    def step_wait(self):
        # 這裡回來的是「多環境」批次
        obs, ext_rew, done, infos = self.venv.step_wait()

        # === RND ===
        obs_t  = th.from_numpy(obs).to(DEVICE).float()
        with th.no_grad():
            tgt_feat = self.target(obs_t)
        pred_feat = self.predictor(obs_t)

        mse_each = self.mse(pred_feat, tgt_feat).mean(dim=1)  # (B,)
        loss     = mse_each.mean()
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        self.last_loss = float(loss.detach().cpu().item())

        int_rew = mse_each.detach().cpu().numpy()
        total   = ext_rew + self.int_coef * int_rew

        # 把拆解塞回 infos
        for i in range(len(infos)):
            infos[i]["int_reward"]   = float(int_rew[i])
            infos[i]["ext_reward"]   = float(ext_rew[i])
            infos[i]["total_reward"] = float(total[i])

        # === 只在有變化時印 ===
        if self.print_each_reward:
            for i in range(len(infos)):
                br = infos[i].get("ext_breakdown", {})  # 由 SPDEnv 塞進來
                # 有外在原因的變化？
                changed_parts = []
                if br:
                    if br.get("xp", 0) != 0:        changed_parts.append(f"xp={self._cint(br['xp'])}")
                    if br.get("gold", 0) != 0:      changed_parts.append(f"gold={self._cint(br['gold'])}")
                    if br.get("explored", 0) != 0:  changed_parts.append(f"explored={self._cint(br['explored'])}")
                    if br.get("depth_up", False):    changed_parts.append(f"{Fore.CYAN}depth+1{Style.RESET_ALL}")
                    if br.get("dead", False):        changed_parts.append(f"{Fore.RED}dead{Style.RESET_ALL}")

                # 決定要不要印：有任何外在變化 or 內在變化大 or 總獎勵大
                should_print = (
                    bool(changed_parts) or
                    abs(int_rew[i]) >= self.min_abs_int or
                    abs(total[i])   >= self.min_abs_total
                )
                if not should_print:
                    continue

                total_s = self._cnum(float(total[i]))
                ext_s   = self._cnum(float(ext_rew[i]))
                int_s   = f"{Fore.MAGENTA}{int_rew[i]:+.3f}{Style.RESET_ALL}"
                reason  = (" | " + " ".join(changed_parts)) if changed_parts else ""
                print(f"[R] t={self._t+i} total={total_s} ext={ext_s} int={int_s}{reason}")

        self._t += len(infos)
        return obs, total, done, infos

    def reset(self, **kwargs):
        return self.venv.reset(**kwargs)


# ── 1.5 即時顯示 Callback ──────────────────────────────
class RewardPrinterCallback(BaseCallback):
    def __init__(self, log_every=200, verbose=1):
        super().__init__(verbose)
        self.log_every = log_every
        self.buf_total, self.buf_ext, self.buf_int = [], [], []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if not info: continue
            self.buf_total.append(info.get("total_reward", 0.0))
            self.buf_ext.append(info.get("ext_reward", 0.0))
            self.buf_int.append(info.get("int_reward", 0.0))

        if self.num_timesteps % self.log_every == 0 and self.buf_total:
            mt = float(np.mean(self.buf_total))
            me = float(np.mean(self.buf_ext))
            mi = float(np.mean(self.buf_int))
            print(f"[{self.num_timesteps}] mean total={mt:.3f} | ext={me:.3f} | int={mi:.3f}")
            # TensorBoard
            self.logger.record("reward/mean_total", mt)
            self.logger.record("reward/mean_ext", me)
            self.logger.record("reward/mean_int", mi)
            self.buf_total.clear(); self.buf_ext.clear(); self.buf_int.clear()
        return True

# ---- VecEnv 工廠（加 Monitor 方便 episode 統計） ----
def make_env(i):
    def _init():
        x = 20 + i*500
        y = 40
        extra_args = [f"--pos={x},{y}"]  # 透過 SPDEnv 傳給 jar
        env = SPDEnv(
            jar_path=JAR_PATH,
            capture_region=(x, y, 480, 270),  # 對應小窗
            grid_size=(24, 42),
            extra_java_args=extra_args,       # 你在 SPDEnv._start_proc() 加這參數透傳
            forbidden_rects_norm=[(0.85,0.0,1.0,0.12)],
        )
        return Monitor(env)
    return _init

if __name__ == "__main__":
    N = 4  # 視你的 CPU/GPU 而定，2~4 起步
    vec_env = SubprocVecEnv([make_env(i) for i in range(N)])
    vec_env = VecTransposeImage(vec_env)
    vec_env = VecFrameStack(vec_env, n_stack=4)
    vec_env = RNDReward(vec_env, int_coef=1.0)

    # 也可加大 policy 的 MLP 頭，增加算量/VRAM
    policy_kwargs=dict(
        net_arch=[dict(pi=[1024,1024], vf=[1024,1024])],  # 比預設大很多
        activation_fn=th.nn.ReLU,
        ortho_init=False
    )
    
    # 建議修改：model = PPO(...)
    model = PPO(
        "CnnPolicy", vec_env,
        device=DEVICE,                 # ✅ 讓 policy 用上 GPU
        learning_rate=2.5e-4,
        n_steps=4096,                  # ↑收集多一點再更新
        batch_size=4096,               # ↑整批喂 GPU, 提高利用率
        n_epochs=10,                   # ↑更新輪數
        gamma=0.99, verbose=1,
        policy_kwargs=policy_kwargs,
        # tensorboard_log="runs/spd"
    )

    cb = RewardPrinterCallback(log_every=1000)
    model.learn(TOTAL_STEPS, callback=cb)
    model.save("spd_ppo_stage1")

    # 第二階段（可選）
    vec_env2 = DummyVecEnv([make_env()])
    vec_env2 = VecTransposeImage(vec_env2)
    vec_env2 = VecFrameStack(vec_env2, n_stack=4)
    vec_env2 = RNDReward(vec_env2, int_coef=0.5)
    model.set_env(vec_env2)
    model.learn(500_000, callback=cb)
    model.save("spd_ppo_stage2")