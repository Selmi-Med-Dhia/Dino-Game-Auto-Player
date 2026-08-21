# Dino Auto-Player

A small Windows Python app for Chrome's offline Dinosaur game. You choose one detector point on the screen. The app calibrates the local background, watches a circle plus a short look-ahead strip, estimates obstacle speed, and presses **Space** when an obstacle is about to reach the selected point.

## Quick start

1. Install **Python 3.11+** from python.org. On Windows, make sure the `py` launcher is installed.
2. Open Chrome's Dinosaur game (`chrome://dino`) and leave it visible.
3. Double-click **`run.bat`**.
4. When the screen dims, click an **empty point a little in front of the dinosaur, around the middle height of a cactus**.
5. A cyan circle marks the point. Click **Start autoplay**.
6. The app briefly focuses the game, calibrates the clean background, then starts/restarts the game with Space.
7. Press **F8** at any time to stop.

## Choosing the point

Good placement matters more than any setting. Put the circle:

- on empty background;
- horizontally in front of the dinosaur;
- high enough that the ground line is not inside the circle;
- low enough that normal cacti cross it;
- typically around 70-130 px in front of the dinosaur, depending on browser zoom.

The predictor scans about 150 px to the **right** of this point and estimates right-to-left obstacle motion. Faster obstacles therefore trigger slightly earlier.

## Tuning

- **Predictive lead** (default 60 ms): increase if jumps become late as the game speeds up; decrease if jumps are too early.
- **Circle radius** (default 14 px): larger is easier to hit but more likely to see unrelated pixels.
- **Min contrast** (default 26): increase if the app false-triggers on visual noise; decrease if a low-contrast theme is missed.

## Why it handles light/dark modes

It does not hard-code “black cactus.” During calibration it stores the local background image, then detects pixels whose absolute brightness difference from that background exceeds a threshold. A white obstacle on dark background and a dark obstacle on light background are both detected.

## Multiple cacti

A jump disarms the detector until the circle has genuinely cleared for a short interval. That prevents one wide cactus from causing repeated Space presses, while a later separate cactus can trigger a new jump after the gap.

## Notes

- Keep the game visible and unobstructed while it runs.
- The cyan detector marker is excluded from screen capture on supported Windows builds. If Windows does not support that API, the marker automatically hides while autoplay is active so it cannot detect itself.
- The app uses only screen capture and synthetic keyboard/mouse input; it does not modify Chrome or the game.
