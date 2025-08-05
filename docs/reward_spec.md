# Reward Specification

Shattered Pixel Dungeon awards a final score at the end of each run based on multiple factors.  To use this score for reinforcement learning we need to convert it into an incremental reward signal.

## Official scoring

The official score is composed of:

- **Progression** – proportional to the product of the deepest floor reached and the hero's level (capped).
- **Treasure** – the amount of gold and the value of items carried (capped).
- **Exploration** – bonus for exploring new floors and fully clearing floors.
- **Bosses** – defeating bosses yields fixed bonuses and sometimes penalties for taking hits.
- **Quests** – completing side quests grants fixed bonuses.
- **Endgame multipliers** – multiplying factors based on how the game ends (death, ascension, etc.).
- **Challenge modifiers** – each active challenge adds an additional multiplier.

See the game source for the exact formulas.

## Incremental reward

To generate a dense learning signal we compute the difference in score between timesteps (`score_t − score_{t−1}`) and add it to the reward.  Alternatively, the scoring formula can be broken down into events and each event can trigger an immediate reward:

- +x for discovering a new tile (exploration).
- +y for clearing a floor.
- +z for killing a boss.
- +w for completing a quest.
- etc.

The choice of coefficients is up to the experimenter; they should reflect the relative importance of each activity.
