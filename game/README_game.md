# Game Source Code and Modifications

This directory is intended to hold a copy of the **Shattered Pixel Dungeon** source code that you will train against.  The original game is available from the developer’s GitHub repository (see <https://github.com/00-Evan/shattered-pixel-dungeon>).

To prepare the game for headless training you should:

1. **Clone the source** into this `game/` directory.  The typical layout has a `core/` module with the game logic and a `desktop/` module with an LWJGL launch class.
2. **Disable vertical sync and FPS limits** in the desktop launcher.  For example, in `DesktopLauncher.java` set `vSyncEnabled = false;` and `foregroundFPS = 0;` to allow the game to run as fast as possible during training.
3. **Add a headless mode** that renders into a framebuffer instead of an on‑screen window and sends raw pixel buffers over a socket or pipe to the Python environment.  You can implement a separate launcher (e.g. `HeadlessLauncher`) that accepts command line arguments such as `--ai-mode` and `--port`.
4. **Expose useful internal variables**, such as current score or health, if you plan to use them as external rewards.  These values can be included in the data sent to the Python side.

This repository does not include the game source by default for licensing reasons.  Please clone the upstream game and apply your own modifications as described above.
