"""Replay-style detector tests with Dino-like moving scenes.

These are deliberately more realistic than the small unit fixtures: they include
an animated ground line, cactus arms, both light/dark themes, 120 Hz sampling,
and several game speeds.
"""

import numpy as np

from detector import AutoJumpDetector, DetectorConfig

WIDTH = 320
HEIGHT = 52
SENSOR_X = 32
SENSOR_Y = 30


def scene(light: bool, phase: int = 0) -> np.ndarray:
    bg = 247 if light else 22
    fg = 83 if light else 210
    frame = np.full((HEIGHT, WIDTH), bg, dtype=np.uint8)

    # A thin moving ground pattern below the detector circle. This should not
    # be mistaken for an approaching obstacle.
    ground_y = 48
    frame[ground_y, :] = fg
    for x in range((phase % 17) - 17, WIDTH, 34):
        x0 = max(0, x)
        x1 = min(WIDTH, x + 3)
        if x1 > x0:
            frame[ground_y - 2 : ground_y, x0:x1] = fg
    return frame


def cactus(frame: np.ndarray, x: float, light: bool) -> None:
    fg = 83 if light else 210
    x = int(round(x))
    bottom = 47

    def rect(x0: int, x1: int, y0: int, y1: int) -> None:
        x0 = max(0, x0)
        x1 = min(WIDTH, x1)
        y0 = max(0, y0)
        y1 = min(HEIGHT, y1)
        if x1 > x0 and y1 > y0:
            frame[y0:y1, x0:x1] = fg

    # Trunk plus two arms, roughly matching the geometry that matters to the
    # Chrome Dino collision detector.
    rect(x, x + 9, bottom - 31, bottom)
    rect(x - 5, x + 3, bottom - 23, bottom - 18)
    rect(x - 5, x - 1, bottom - 27, bottom - 18)
    rect(x + 7, x + 14, bottom - 18, bottom - 14)
    rect(x + 11, x + 14, bottom - 23, bottom - 14)


def make_detector(light: bool) -> AutoJumpDetector:
    detector = AutoJumpDetector(
        WIDTH,
        HEIGHT,
        SENSOR_X,
        SENSOR_Y,
        DetectorConfig(radius=14),
    )
    detector.calibrate([scene(light) for _ in range(18)])
    return detector


def replay_one_cactus(speed_px_s: float, light: bool) -> list:
    detector = make_detector(light)
    dt = 1.0 / 120.0
    timestamp = 1.0
    x = 285.0
    jumps = []

    while x > -35:
        frame = scene(light, int(timestamp * speed_px_s))
        cactus(frame, x, light)
        telemetry = detector.process(frame, timestamp)
        if telemetry.should_jump:
            jumps.append(telemetry)
        x -= speed_px_s * dt
        timestamp += dt

    return jumps


def test_replay_detects_one_cactus_at_multiple_game_speeds():
    for speed in (300.0, 400.0, 650.0, 900.0, 1200.0):
        jumps = replay_one_cactus(speed, light=True)
        assert len(jumps) == 1, f"speed={speed}: got {len(jumps)} jumps"
        # Speed should be learned before the jump at realistic frame rates.
        assert jumps[0].speed_px_s >= 0.75 * speed


def test_replay_works_in_dark_theme():
    for speed in (400.0, 900.0):
        jumps = replay_one_cactus(speed, light=False)
        assert len(jumps) == 1


def test_replay_rearms_for_separate_cacti_but_not_tight_cluster():
    def run(gap: float) -> int:
        detector = make_detector(True)
        speed = 650.0
        dt = 1.0 / 120.0
        timestamp = 1.0
        x = 285.0
        count = 0
        while x + gap > -40:
            frame = scene(True, int(timestamp * speed))
            cactus(frame, x, True)
            cactus(frame, x + gap, True)
            if detector.process(frame, timestamp).should_jump:
                count += 1
            x -= speed * dt
            timestamp += dt
        return count

    assert run(70.0) == 1   # one jump clears a tight cactus cluster
    assert run(160.0) == 2  # a genuine gap rearms for the next cactus
