# Dino Auto-Player — Servo Simulation

A Windows Python auto-player for Chrome's offline Dinosaur game, designed as a **software simulation of a future Arduino + optical sensor + servo** build.

The important change is that obstacle detection happens **far ahead of the Dino**. The app treats the selected point like a physical LDR/phototransistor mounted on the monitor, measures the obstacle envelope there, then schedules a future servo command so the delayed physical Space contact makes the Dino land just after the obstacle's trailing edge.

## Quick start

1. Install Python 3.11+.
2. Open `chrome://dino` and leave it visible.
3. Double-click `run.bat`.
4. Selection step 1: click the **front x-position of the Dino**.
5. Selection step 2: click an **empty point far ahead** at cactus-body height. If your screen allows it, start around **700–1100 physical pixels ahead**.
6. Set the simulated servo delay and Space-contact duration.
7. Click **Start simulation**. Press **F8** to stop.

The two clicks let the app measure the virtual sensor distance automatically; you no longer need to guess a `Marker ahead` number.

## Hardware-style timing model

Detection and action are deliberately separate events:

1. A cactus or close cactus group crosses the far virtual sensor.
2. Short clear gaps are merged into one **obstacle envelope**. This prevents the fork shape of a cactus from becoming several fake obstacles.
3. The trailing-edge crossing time is projected from the sensor to the Dino using measured game speed.
4. The desired physical key-contact time is chosen so the Dino lands just after that trailing edge.
5. The **servo/contact delay** is subtracted from that contact time, producing the earlier servo-command time.
6. The PC does not press Space immediately. It actually waits the configured servo delay, then holds Space for the configured contact duration. That makes the simulation behave like a mechanical actuator instead of merely compensating for one mathematically.

If the sensor is too close for the current speed + actuator delay, the UI reports **how many milliseconds late** it is and approximately **how many extra pixels farther ahead** the sensor should be moved.

## Why this design is more transferable to Arduino

A number of simple Dino bots just inspect a near-Dino pixel/rectangle and press Space immediately. That can work well on a PC but does not transfer to a servo because mechanical actuation has latency.

This implementation borrows the more hardware-friendly ideas used by open-source optical-sensor Dino projects:

- treat the sensor output as an **envelope**, not a single pixel;
- merge short gaps inside fork-shaped cacti;
- keep recent envelope durations as a speed-related signal;
- adapt timing as the game accelerates;
- keep detection running independently of action timing.

The PC version additionally uses motion between frames to estimate screen speed directly. On Arduino, the same timing can later be approximated from envelope history, or measured more accurately with two horizontally separated optical sensors.

## Tuning that transfers to hardware

- **Virtual sensor distance** — set by the two clicks. Later, mount the physical optical sensor at approximately the same screen distance ahead of the Dino.
- **Servo/contact delay (ms)** — command sent to servo → key physically depressed. Default `160 ms`.
- **Space contact/hold (ms)** — how long the servo physically holds Space. Default `80 ms`.
- **Land after exit margin (ms)** — desired touchdown after the final cactus trailing edge. Default `30 ms`.
- **Sensor radius** and **Min contrast** are only PC-vision settings; they do not need to transfer to Arduino.

A larger servo delay needs a farther sensor. At high game speed a sensor can become physically too close even if detection itself is perfect; the app reports this condition instead of hiding it.

## Reliability details

- Uses Windows physical pixels and DPI awareness so Tk, mouse input, and `mss` capture agree at 125%/150% scaling.
- Captures only a thin sensor-height strip for high polling rate.
- Uses adaptive background subtraction and automatically rebases on a whole-scene day/night inversion instead of interpreting it as a giant cactus.
- Uses a rolling median of edge-motion speed samples to reject quantization spikes.
- Keeps scheduled commands after the obstacle has left the sensor area.
- Prevents a second physical Space contact from occurring while the previous jump is still expected to be airborne.
- Uses a leading-edge safety limit for unusually wide cactus groups.

## Tests

The synthetic replay suite covers:

- light and dark themes;
- day/night background rebasing;
- 300–1500 px/s obstacle motion;
- delayed servo contact;
- correct command-time shift as servo latency changes;
- future commands that fire after the cactus is no longer visible at the sensor;
- fork/close-cactus envelope merging;
- separate following cacti;
- sensor-too-close diagnostics;
- wide-cluster leading-edge safety;
- Chromium-like normal jump airtime.

Run tests with:

```bash
python -m pytest -q
```

The app uses screen capture and synthetic input only. It does not modify or inject code into Chrome.
