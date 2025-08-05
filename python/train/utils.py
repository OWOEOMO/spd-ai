"""
Utility functions for training scripts.

This module includes functions to parse command-line arguments,
schedule coefficients (e.g., intrinsic reward weight), configure logging,
and create callbacks for training.

Functions:
    parse_common_args: parse common command-line arguments for training scripts.
    linear_schedule: create a linear learning rate or coefficient schedule.
    beta_schedule: linearly anneal the intrinsic reward coefficient.
"""

from typing import Callable
import argparse


def parse_common_args(description: str = None) -> argparse.Namespace:
    """
    Parse common command-line arguments for training scripts.

    Args:
        description: Optional description for the ArgumentParser.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--total-steps", type=int, default=1_000_000,
                        help="Total number of timesteps to train.")
    parser.add_argument("--env-ports", type=str, default="",
                        help="Comma-separated list of environment ports for multi-process training.")
    parser.add_argument("--log-dir", type=str, default="./runs",
                        help="Directory to store tensorboard logs.")
    parser.add_argument("--model-path", type=str, default="model.zip",
                        help="Path to save or load the model.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed.")
    return parser.parse_args()


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Create a linear scheduler for a parameter.

    Returns a function that takes progress (from 1.0 to 0.0) and returns
    a value linearly scaled from initial_value to zero.

    Args:
        initial_value: Initial value at the start of training (progress=1.0).

    Returns:
        Callable[[float], float]: A function mapping progress to value.
    """
    def scheduler(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return scheduler


def beta_schedule(initial_beta: float, final_beta: float, duration: float) -> Callable[[float], float]:
    """
    Create a schedule for the intrinsic reward weight (beta) that anneals from initial_beta to final_beta.

    Args:
        initial_beta: Starting beta coefficient.
        final_beta: Ending beta coefficient.
        duration: Fraction of training timesteps over which to anneal (0 < duration <= 1).

    Returns:
        Callable[[float], float]: A function mapping progress_remaining to beta.
    """
    assert 0 <= final_beta <= initial_beta, "final_beta must be <= initial_beta"
    assert 0 < duration <= 1.0, "duration must be within (0,1]"
    def scheduler(progress_remaining: float) -> float:
        # progress_remaining is 1 at start and 0 at end; invert for elapsed fraction
        elapsed = 1.0 - progress_remaining
        if elapsed <= duration:
            return initial_beta - (initial_beta - final_beta) * (elapsed / duration)
        else:
            return final_beta
    return scheduler
