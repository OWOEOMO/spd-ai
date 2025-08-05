# Shattered Pixel Dungeon AI Project

This repository contains an AI research project aimed at training an agent to play **Shattered Pixel Dungeon**.  The project uses reinforcement learning in two phases:

1. **Unsupervised exploration** – the agent learns to survive and explore using purely intrinsic rewards (for example curiosity‑driven bonuses).
2. **Fine‑tuning** – after the agent has learned to roam the dungeon, we add external rewards based on the game’s scoring system to encourage progressing deeper and winning runs.

The repository is organised into a few main areas:

- **game/** – your copy of the Shattered Pixel Dungeon source code along with small modifications for headless training and high‑FPS rendering.  See `game/README_game.md` for more information.
- **python/** – Gymnasium environments, wrappers, training scripts and utilities for reinforcement learning.
- **data/** – raw recordings, extracted frames and any data used for supervised or self‑supervised pre‑training.
- **checkpoints/** – model weights and TensorBoard logs.
- **docs/** – design notes, reward specifications and troubleshooting guides.

This repository is only a skeleton to get you started; you will need to populate it with the actual game source code, write your environment logic, and develop your training routines.
