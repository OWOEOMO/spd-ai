#!/usr/bin/env python3
"""
Utility script to plot score curves for Shattered Pixel Dungeon agent evaluation.

This script reads evaluation results (e.g., CSV with episode and reward columns)
and produces a simple matplotlib chart showing reward over episodes or steps.
"""

import argparse
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Plot reward curve from evaluation results.")
    parser.add_argument("csv_path", help="Path to CSV file with evaluation results containing episode and reward columns.")
    parser.add_argument("--episode-col", default="episode", help="Column name for episode index.")
    parser.add_argument("--reward-col", default="reward", help="Column name for reward value.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    x = df[args.episode_col] if args.episode_col in df.columns else range(len(df))
    y = df[args.reward_col]

    plt.figure()
    plt.plot(x, y)
    plt.xlabel(args.episode_col)
    plt.ylabel(args.reward_col)
    plt.title("Evaluation Reward Curve")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
