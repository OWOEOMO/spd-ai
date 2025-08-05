# Design Overview

This document outlines the design for training an AI agent to play **Shattered Pixel Dungeon**.

## Two‑phase training

We train the agent in two phases:

### Phase 1: Unsupervised exploration

Use an intrinsic reward signal (for example Random Network Distillation) to encourage the agent to explore and survive.  The environment uses only intrinsic reward plus small survival penalties for taking damage or dying.

### Phase 2: Fine‑tuning with external rewards

After the agent has learned to navigate and survive, we enable external rewards based on the game’s official scoring system.  We compute an incremental reward by taking the difference of the score between consecutive steps or by breaking down the scoring formula into events (exploration, bosses, quests, treasure, etc.).  The intrinsic reward coefficient is reduced in this stage so that the agent still explores but focuses on progressing.

## Environment

The Java game is run in headless mode and communicates with the Python environment via sockets or pipes.  Observations are stacks of preprocessed frames (e.g. 128×128 grayscale images); actions are discrete and correspond to moving, attacking, picking up items and other game interactions.  Rewards are computed by wrappers around the base environment.

## Wrappers

Wrappers modify the behaviour of the base environment:

- **`RNDReward`** – computes an intrinsic curiosity bonus based on how novel the current observation is.
- **`ExtrinsicWrapper`** – computes an external reward by taking the difference of the game’s score between steps.

## Directory structure

See the `README.md` for an overview of the repository layout and the purpose of each top‑level directory.
