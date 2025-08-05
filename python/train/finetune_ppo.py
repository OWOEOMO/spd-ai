"""
Fine-tuning script for Phase 2: extrinsic reward for Shattered Pixel Dungeon.
This script loads a pre-trained intrinsic model, wraps the environment with extrinsic rewards,
and continues training with a smaller intrinsic reward coefficient.
"""

import argparse
from sb3_contrib.ppo_recurrent import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv

# Import your environment and wrappers
from python.env.spd_env import SPDEnv
from python.env.wrappers import RNDReward, ExtrinsicWrapper


def main():
    parser = argparse.ArgumentParser(description="Fine-tune an agent with extrinsic reward.")
    parser.add_argument("--load", type=str, default="rnd_phase1.zip", help="Path to the pre-trained model file.")
    parser.add_argument("--num-env", type=int, default=8, help="Number of parallel environments.")
    parser.add_argument("--total-steps", type=int, default=500_000, help="Total timesteps to fine-tune.")
    parser.add_argument("--beta", type=float, default=0.2, help="Intrinsic reward coefficient during fine-tuning.")
    args = parser.parse_args()

    # Factory to create a single environment instance
    def make_env(_):
        def _thunk():
            env = SPDEnv()
            # lower intrinsic coefficient to keep exploring while optimizing extrinsic reward
            env = RNDReward(env, int_coef=args.beta)
            env = ExtrinsicWrapper(env)
            return env
        return _thunk

    env_fns = [make_env(i) for i in range(args.num_env)]
    vec_env = SubprocVecEnv(env_fns)

    # Load the pre-trained model
    model = RecurrentPPO.load(args.load, env=vec_env)

    # Fine-tune the model
    model.learn(total_timesteps=args.total_steps)

    # Save the fine-tuned model
    model.save("spd_final")


if __name__ == "__main__":
    main()
