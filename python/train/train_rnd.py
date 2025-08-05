"""
Training script for Phase 1: intrinsic reward (RND) for Shattered Pixel Dungeon.
This script creates multiple parallel environments wrapped with RNDReward and trains a PPO agent.
"""

import argparse
from sb3_contrib.ppo_recurrent import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv

# Import your environment and wrappers
from python.env.spd_env import SPDEnv
from python.env.wrappers import RNDReward


def main():
    parser = argparse.ArgumentParser(description="Train an agent with intrinsic RND reward.")
    parser.add_argument("--num-env", type=int, default=8, help="Number of parallel environments.")
    parser.add_argument("--total-steps", type=int, default=1_500_000, help="Total timesteps to train.")
    args = parser.parse_args()

    # Create a list of factory functions for SubprocVecEnv
    def make_env(_):
        def _thunk():
            env = SPDEnv()
            env = RNDReward(env, int_coef=1.0)
            return env
        return _thunk

    env_fns = [make_env(i) for i in range(args.num_env)]
    vec_env = SubprocVecEnv(env_fns)

    # Instantiate recurrent PPO agent
    model = RecurrentPPO(
        "CnnLstmPolicy",
        vec_env,
        n_steps=2048,
        batch_size=512,
        n_epochs=4,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        verbose=1,
    )

    # Train the model
    model.learn(total_timesteps=args.total_steps)

    # Save the trained model
    model.save("rnd_phase1")


if __name__ == "__main__":
    main()
