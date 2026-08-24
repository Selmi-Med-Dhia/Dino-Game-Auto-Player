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
        sensor_ahead_px=0,
        jump_air_time_s=0.12,
        landing_margin_s=0.02,
        cluster_gap_time_s=0.08,
        cluster_gap_fallback_px=20,
        min_entry_clearance_s=0.05,
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
    d = make_detector(bg=255, jump_air_time_s=0.10)
    obstacle = np.full((24, 120), 255, np.uint8)
    paint_obstacle(obstacle, 18, 25, 0)
    clear = np.full((24, 120), 255, np.uint8)

    assert d.process(obstacle, 1.00).should_jump
    d.process(clear, 1.12)
    d.process(clear, 1.15)
    assert d.process(obstacle, 1.30).should_jump


def test_speed_tracking_uses_trailing_edge_landing_plan():
    d = make_detector(
        bg=255,
        min_speed_px_s=100,
        max_speed_px_s=2000,
        jump_air_time_s=0.20,
        landing_margin_s=0.02,
    )

    jumped = None
    for x, t in [(90, 1.00), (82, 1.02), (74, 1.04), (66, 1.06), (58, 1.08)]:
        frame = np.full((24, 120), 255, np.uint8)
        paint_obstacle(frame, x, x + 12, 0)
        telemetry = d.process(frame, t)
        if telemetry.should_jump:
            jumped = telemetry
            break

    assert jumped is not None
    assert 300 <= jumped.speed_px_s <= 500
    assert jumped.obstacle_exit_distance_px is not None
    assert jumped.obstacle_exit_distance_px > jumped.obstacle_distance_px
    assert jumped.predicted_exit_tti_dino_s is not None
    assert jumped.landing_error_s is not None
    assert jumped.landing_error_s >= 0.02
    assert jumped.landing_error_s < 0.06


def test_minor_noise_does_not_trigger():
    d = make_detector(bg=255)
    frame = np.full((24, 120), 255, np.uint8)
    frame[10:12, 20:22] = 245
    t = d.process(frame, 1.0)
    assert not t.should_jump


def test_predictive_jump_does_not_rearm_before_same_obstacle_passes_or_landing():
    d = make_detector(
        bg=255,
        min_speed_px_s=100,
        max_speed_px_s=2000,
        jump_air_time_s=0.20,
        landing_margin_s=0.02,
    )

    jumped_at = None
    for x, t in [(90, 1.00), (82, 1.02), (74, 1.04), (66, 1.06), (58, 1.08), (50, 1.10)]:
        frame = np.full((24, 120), 255, np.uint8)
        paint_obstacle(frame, x, x + 10, 0)
        result = d.process(frame, t)
        if result.should_jump:
            jumped_at = t
            break
    assert jumped_at is not None

    t = jumped_at + 0.02
    for x in (42, 34, 26, 18, 10):
        frame = np.full((24, 120), 255, np.uint8)
        paint_obstacle(frame, x, x + 10, 0)
        assert not d.process(frame, t).should_jump
        t += 0.02


def test_close_cactus_cluster_is_treated_as_one_hazard():
    d = make_detector(
        bg=255,
        min_speed_px_s=100,
        max_speed_px_s=2000,
        jump_air_time_s=0.22,
        cluster_gap_time_s=0.10,
        cluster_gap_fallback_px=35,
    )

    jumped = 0
    t = 1.0
    for lead_x in range(100, -10, -4):
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

    frame = np.clip(128 + rng.integers(-4, 5, size=(20, 60)), 0, 255).astype(np.uint8)
    assert not d.process(frame, 1.0).should_jump
