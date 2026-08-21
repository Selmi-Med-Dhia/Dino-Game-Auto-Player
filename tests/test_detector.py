import numpy as np

from detector import AutoJumpDetector, DetectorConfig


def make_detector(bg=255, **overrides):
    values = dict(
        radius=8,
        min_pixel_delta=20,
        column_occupancy=0.20,
        circle_occupancy=0.05,
        clear_occupancy=0.02,
        lead_time_s=0.060,
        cooldown_s=0.100,
        clear_hold_s=0.020,
    )
    values.update(overrides)
    cfg = DetectorConfig(**values)
    d = AutoJumpDetector(width=120, height=24, sensor_x=20, sensor_y=12, config=cfg)
    frames = [np.full((24, 120), bg, dtype=np.uint8) for _ in range(8)]
    d.calibrate(frames)
    return d


def paint_obstacle(frame, x0, x1, value, y0=5, y1=21):
    frame[y0:y1, x0:x1] = value
    return frame


def test_black_obstacle_on_white_background_triggers_in_circle():
    d = make_detector(bg=255)
    frame = np.full((24, 120), 255, np.uint8)
    paint_obstacle(frame, 18, 25, 0)
    t = d.process(frame, 1.0)
    assert t.should_jump
    assert t.occupancy > 0.05


def test_white_obstacle_on_dark_background_triggers():
    d = make_detector(bg=0)
    frame = np.zeros((24, 120), np.uint8)
    paint_obstacle(frame, 18, 25, 255)
    t = d.process(frame, 1.0)
    assert t.should_jump


def test_same_cactus_does_not_repeat_jump():
    d = make_detector(bg=255)
    frame = np.full((24, 120), 255, np.uint8)
    paint_obstacle(frame, 18, 30, 0)
    assert d.process(frame, 1.0).should_jump
    assert not d.process(frame, 1.15).should_jump
    assert not d.process(frame, 1.30).should_jump


def test_separate_cacti_can_trigger_twice_after_clear_gap():
    d = make_detector(bg=255)
    obstacle = np.full((24, 120), 255, np.uint8)
    paint_obstacle(obstacle, 18, 25, 0)
    clear = np.full((24, 120), 255, np.uint8)

    assert d.process(obstacle, 1.00).should_jump
    d.process(clear, 1.02)
    d.process(clear, 1.05)  # clear long enough to rearm
    assert d.process(obstacle, 1.20).should_jump


def test_speed_tracking_predicts_before_circle():
    d = make_detector(bg=255, min_speed_px_s=100, max_speed_px_s=2000)

    # Obstacle moves left 8 px every 20 ms = 400 px/s.
    positions = [70, 62, 54, 46]
    times = [1.00, 1.02, 1.04, 1.06]
    telemetry = None
    for x, t in zip(positions, times):
        frame = np.full((24, 120), 255, np.uint8)
        paint_obstacle(frame, x, x + 6, 0)
        telemetry = d.process(frame, t)

    assert telemetry is not None
    assert 300 <= telemetry.speed_px_s <= 500
    # At x=46, distance to sensor is 26 px; tti about 65ms, close to lead.
    # Move one more step to 42 -> 22px / 400 = 55ms, should predict a jump.
    frame = np.full((24, 120), 255, np.uint8)
    paint_obstacle(frame, 42, 48, 0)
    telemetry = d.process(frame, 1.07)
    assert telemetry.predicted_tti_s is not None
    assert telemetry.predicted_tti_s <= 0.060
    assert telemetry.should_jump


def test_minor_noise_does_not_trigger():
    d = make_detector(bg=255)
    frame = np.full((24, 120), 255, np.uint8)
    frame[10:12, 20:22] = 245
    t = d.process(frame, 1.0)
    assert not t.should_jump

def test_predictive_jump_does_not_rearm_before_same_obstacle_passes():
    d = make_detector(bg=255, min_speed_px_s=100, max_speed_px_s=2000, lead_time_s=0.08)

    # Teach about 400 px/s and get a predictive jump while still right of circle.
    for x, t in [(65, 1.00), (57, 1.02), (49, 1.04)]:
        frame = np.full((24, 120), 255, np.uint8)
        paint_obstacle(frame, x, x + 8, 0)
        result = d.process(frame, t)
    assert result.should_jump

    # Same cactus keeps moving into and across the circle. It must not trigger again.
    for x, t in [(41, 1.06), (33, 1.08), (25, 1.10), (18, 1.12), (12, 1.14)]:
        frame = np.full((24, 120), 255, np.uint8)
        paint_obstacle(frame, x, x + 8, 0)
        assert not d.process(frame, t).should_jump

def test_close_cactus_cluster_is_treated_as_one_hazard():
    d = make_detector(bg=255, min_speed_px_s=100, max_speed_px_s=2000, lead_time_s=0.08)

    # Two blocks with only a tiny gap move together like a cactus cluster.
    jumped = 0
    t = 1.0
    for lead_x in range(72, 4, -4):
        frame = np.full((24, 120), 255, np.uint8)
        paint_obstacle(frame, lead_x, lead_x + 6, 0)
        paint_obstacle(frame, lead_x + 9, lead_x + 15, 0)
        if d.process(frame, t).should_jump:
            jumped += 1
        t += 0.01
    assert jumped == 1


def test_calibration_raises_threshold_above_small_background_flicker():
    cfg = DetectorConfig(radius=8, min_pixel_delta=5, noise_multiplier=6)
    d = AutoJumpDetector(width=60, height=20, sensor_x=15, sensor_y=10, config=cfg)
    rng = np.random.default_rng(123)
    frames = [np.clip(128 + rng.integers(-4, 5, size=(20, 60)), 0, 255).astype(np.uint8) for _ in range(20)]
    d.calibrate(frames)
    assert d.threshold >= 5

    # A frame with only the same scale of flicker should not look like a cactus.
    frame = np.clip(128 + rng.integers(-4, 5, size=(20, 60)), 0, 255).astype(np.uint8)
    assert not d.process(frame, 1.0).should_jump
