"""Replay-style detector tests with Dino-like moving scenes."""

import numpy as np

from detector import AutoJumpDetector, DetectorConfig

WIDTH = 760
HEIGHT = 52
SENSOR_X = 42
SENSOR_Y = 30


def scene(light: bool, phase: int = 0) -> np.ndarray:
    bg = 247 if light else 22
    fg = 83 if light else 210
    frame = np.full((HEIGHT, WIDTH), bg, dtype=np.uint8)
    ground_y = 48
    frame[ground_y, :] = fg
    for x in range((phase % 17) - 17, WIDTH, 34):
        x0 = max(0, x)
        x1 = min(WIDTH, x + 3)
        if x1 > x0:
            frame[ground_y - 2:ground_y, x0:x1] = fg
    return frame


def cactus(frame: np.ndarray, x: float, light: bool) -> None:
    fg = 83 if light else 210
    x = int(round(x))
    bottom = 47

    def rect(x0, x1, y0, y1):
        x0, x1 = max(0, x0), min(WIDTH, x1)
        y0, y1 = max(0, y0), min(HEIGHT, y1)
        if x1 > x0 and y1 > y0:
            frame[y0:y1, x0:x1] = fg

    rect(x, x + 9, bottom - 31, bottom)
    rect(x - 5, x + 3, bottom - 23, bottom - 18)
    rect(x - 5, x - 1, bottom - 27, bottom - 18)
    rect(x + 7, x + 14, bottom - 18, bottom - 14)
    rect(x + 11, x + 14, bottom - 23, bottom - 14)


def make_detector(light: bool) -> AutoJumpDetector:
    d = AutoJumpDetector(
        WIDTH,
        HEIGHT,
        SENSOR_X,
        SENSOR_Y,
        DetectorConfig(radius=14, lookahead_px=600, sensor_ahead_px=100.0),
    )
    d.calibrate([scene(light) for _ in range(18)])
    return d


def replay(obstacles, speed_px_s: float, light=True):
    d = make_detector(light)
    dt = 1 / 120
    t = 1.0
    lead_x = 680.0
    jumps = []
    while lead_x + max(obstacles) > -80:
        f = scene(light, int(t * speed_px_s))
        for offset in obstacles:
            cactus(f, lead_x + offset, light)
        tel = d.process(f, t)
        if tel.should_jump:
            jumps.append((t, lead_x, tel))
        lead_x -= speed_px_s * dt
        t += dt
    return jumps


def test_replay_detects_one_cactus_and_lands_after_exit_at_multiple_speeds():
    for speed in (300.0, 400.0, 650.0, 900.0, 1200.0, 1500.0):
        jumps = replay([0.0], speed, True)
        assert len(jumps) == 1, (speed, len(jumps))
        tel = jumps[0][2]
        assert tel.speed_px_s >= 0.72 * speed
        assert tel.predicted_exit_tti_dino_s is not None
        assert tel.landing_error_s is not None
        assert 0.015 <= tel.landing_error_s <= 0.070, (speed, tel.landing_error_s)


def test_replay_works_in_dark_theme():
    for speed in (400.0, 900.0, 1400.0):
        assert len(replay([0.0], speed, False)) == 1


def test_tight_following_cactus_is_merged_and_landing_targets_last_exit():
    speed = 650.0
    jumps = replay([0.0, 70.0], speed, True)
    assert len(jumps) == 1
    tel = jumps[0][2]
    assert tel.cluster_width_px is not None and tel.cluster_width_px > 70
    assert tel.landing_error_s is not None
    assert 0.015 <= tel.landing_error_s <= 0.075


def test_separate_cacti_get_two_jumps_not_an_ignored_midair_second_space():
    speed = 650.0
    jumps = replay([0.0, 180.0], speed, True)
    assert len(jumps) == 2
    t0, t1 = jumps[0][0], jumps[1][0]
    assert t1 - t0 >= 0.425


def chromium_normal_jump_airtime(internal_speed: float) -> float:
    """Discrete 60 Hz model of Chromium's normal Trex jump configuration."""
    gravity = 0.6
    y = 0.0
    velocity = -10.0 - internal_speed / 10.0
    frames = 0
    while frames < 120:
        frames += 1
        y += round(velocity)
        velocity += gravity
        if y < -30.0 and velocity < -5.0:
            velocity = -5.0
        if y > 0.0:
            return frames / 60.0
    raise AssertionError("jump never landed")


def test_default_airtime_matches_current_chromium_normal_jump_physics():
    configured = DetectorConfig().jump_air_time_s
    for internal_speed in (6.0, 9.0, 13.0):
        actual = chromium_normal_jump_airtime(internal_speed)
        assert abs(configured - actual) <= 0.035, (internal_speed, configured, actual)
