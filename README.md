# Dino Auto-Player

A Windows Python auto-player for Chrome's offline Dinosaur game. It captures a thin strip of the screen, learns the local background, tracks obstacle speed, groups nearby cacti, and presses **Space** at a time chosen to make the Dino land just after the complete hazard clears.

## Quick start

1. Install Python 3.11+.
2. Open `chrome://dino` in Chrome and leave the game visible.
3. Double-click `run.bat`.
4. Click an empty point roughly **100 px in front of the Dino**, around the middle of a cactus body.
5. Click **Start autoplay**.
6. Press **F8** to stop.

## How jump timing works

The bot no longer times a jump only from the cactus's leading edge.

For every detected hazard it estimates:

- the **entry / leading edge**;
- the **exit / trailing edge**;
- the right-to-left screen speed;
- the width of a close cactus cluster.

Close cacti are merged into one hazard when their time gap is too short for a clean land-and-rejump. The planner then uses the **trailing edge of the last cactus in that cluster** and targets touchdown about 25 ms after it clears the Dino.

A leading-edge safety deadline is still kept as a fallback for unusually wide hazards or late detections.

## Why the jump model is about 440 ms

The default air-time model is based on Chromium's normal T-Rex jump configuration: 60 FPS jump animation, gravity `0.6`, minimum/maximum jump height `30`, initial jump velocity around `-10`, and drop velocity `-5`. A normal tap is roughly 0.43–0.47 seconds airborne across the game's normal speed range, so the planner uses 0.44 seconds.

## Back-to-back cacti

The app does **not** re-arm merely because the detector circle becomes clear. That can fail when the next cactus reaches the detector before the Dino has landed.

Instead, after a predictive jump it waits until the previous jump is expected to be essentially complete. A close following cactus is cleared in the same jump; a genuinely separate cactus can trigger a new Space immediately after touchdown.

## Detection

- Uses adaptive background subtraction, so both light and dark game themes work.
- Scans a horizontal area rather than relying on one exact pixel.
- Uses a roughly **600 px look-ahead band** so fast obstacles are seen early enough to position landing, not just avoid a collision at the last moment.
- Uses physical Windows screen coordinates and enables DPI awareness to avoid 125%/150% display-scaling coordinate mismatches.
- Hides the cyan marker during autoplay so the overlay cannot contaminate screen capture.

## Tuning

- **Marker ahead of Dino (px)**: set this close to the actual horizontal distance between the Dino and your selected cyan point. Default: `100`.
- **Leading-edge safety (ms)**: emergency latest-safe takeoff guard. Default: `70`.
- **Circle radius**: detector-circle size. Default: `14`.
- **Min contrast**: increase if visual noise causes detections; decrease if the cactus is not being seen. Default: `22`.

## Tests

The replay tests use cactus-like shapes, animated ground noise, light/dark themes, 120 Hz sampling, screen speeds from 300 to 1500 px/s, close cactus clusters, separate following cacti, and a discrete model of Chromium's current normal jump physics.

The app uses only screen capture plus synthetic mouse/keyboard input; it does not modify Chrome or inject code into the game.
