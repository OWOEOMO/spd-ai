#!/usr/bin/env python3
"""
Evaluate a trained Shattered Pixel Dungeon agent.

This script loads a trained policy and runs it in the SPD environment for a specified
number of episodes. Optionally it can render frames or save rollouts to disk.

Usage:
    python -m python.eval.rollout --model checkpoint.zip --episodes 5 --render
"""

import argparse

from stable_baselines3 import PPO
frompython.env.spd_env import SPDEnv


def parse_args():
    """Parse command-line arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a trained SPD agent.")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model checkpoint (.zip).")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run.")
    parser.add_argument("--render", action="store_true", help="Render the environment to screen.")
    return parser.parse_args()


def run_episode(model, env, render=False):
    """Run a single episode using the provided model and environment."""
    obs, info = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        if render:
            # Rendering is handled by SPDEnv; pass to let env.render() if implemented
            pass
    return total_reward


def main():
    args = parse_args()
    env = SPDEnv()
    model = PPO.load(args.model, env=env)
    rewards = []
    for _ in range(args.episodes):
        rewards.append(run_episode(model, env, render=args.render))
    env.close()
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    print(f"Average reward over {args.episodes} episodes: {avg_reward}")


if __name__ == "__main__":
    main()
