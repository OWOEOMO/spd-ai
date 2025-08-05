# Troubleshooting

This document collects some common issues and potential solutions encountered when working on this project.

## Game does not run headless

- Ensure you compiled the `:desktop` module with your headless launcher.
- Verify that `vSyncEnabled = false` and `foregroundFPS = 0` in your desktop launcher configuration so that the game does not try to sync with the monitor.
- Make sure you are passing the `--ai-mode` flag (or equivalent) to switch off the graphical user interface.
- Check that the JVM has permissions to open the display or use a framebuffer on your system.

## No data coming from the game

- Verify that the socket or pipe addresses match in both the Java and Python processes.
- Make sure that the game actually sends pixel buffers; as a sanity check use a small client to read a few bytes and confirm that the stream is not empty.
- Check for exceptions on the Java side that might be causing it to exit prematurely.

## Training is unstable or diverging

- Normalise observations (e.g. convert pixel values to floats in the range [0, 1]) before feeding them to the network.
- Adjust the intrinsic reward coefficient `beta` and experiment with different decay schedules so that the agent does not get stuck exploring trivial novelty.
- Use a recurrent policy (e.g. `CnnLstmPolicy`) to help the agent maintain memory in a partially observable environment.
- If the agent collapses to a single behaviour, consider increasing exploration (entropy bonus) or using other intrinsic motivation methods.
