# Common tasks for the Shattered Pixel Dungeon AI project.
# These commands are examples; adjust them to your environment.

.PHONY: headless train_rnd finetune

# Build a headless version of the game using Gradle.  Requires the game source in ./game.
headless:
	cd game && ./gradlew :desktop:headlessDist

# Train the agent with intrinsic rewards only.
train_rnd:
	python3 python/train/train_rnd.py

# Fine‑tune the agent using external rewards.
finetune:
	python3 python/train/finetune_ppo.py
