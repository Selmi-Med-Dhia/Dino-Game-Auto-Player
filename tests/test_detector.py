import numpy as np

from detector import AutoJumpDetector, DetectorConfig


def make_detector(**overrides):
    values = dict(
        radius=8,
        min_pixel_delta=20,
        column_occupancy=0.12,
        circle_occupancy=0.04,
        sensor_ahead_px=700,
        actuator_delay_s=0.16,
        jump_air_time_s=0.45,
        landing_margin_s=0.03,
        envelope_finalize_gap_s=0.08,
        bootstrap_speed_px_s=400,
    )
    values.update(overrides)
    cfg = DetectorConfig(**values)
    d = AutoJumpDetector(width=180, height=32, sensor_x=24, sensor_y=15, config=cfg)
    bg = np.full((32, 180), 250, dtype=np.uint8)
    d.calibrate([bg.copy() for _ in range(12)])
    return d


def frame_with_block(x0=None, x1=None, value=40):
    frame = np.full((32, 180), 250, dtype=np.uint8)
    if x0 is not None and x1 is not None:
        frame[5:27, max(0, x0):min(180, x1)] = value
    return frame


def test_speed_estimator_handles_repeated_integer_edge_positions():
    d = make_detector(min_speed_px_s=80, max_speed_px_s=2000)
    # ~400 px/s, with repeated integer positions as happens at high capture rate.
    samples = [(80, 1.000), (80, 1.004), (78, 1.005), (76, 1.010), (74, 1.015)]
    for x, t in samples:
        d.process(frame_with_block(x, x + 8), t)
    assert 300 <= d.speed_px_s <= 500


def test_day_night_inversion_rebases_instead_of_triggering():
    d = make_detector()
    dark = np.full((32, 180), 15, dtype=np.uint8)
    tel = d.process(dark, 1.0)
    assert tel.background_rebased
    assert not tel.should_jump


def test_far_sensor_finalizes_envelope_then_schedules_future_command():
    d = make_detector(sensor_ahead_px=800, envelope_finalize_gap_s=0.05)
    t = 1.0
    dt = 0.01

    # Teach ~400 px/s while obstacle approaches from the right.
    x = 90.0
    while x > 10:
        tel = d.process(frame_with_block(int(x), int(x) + 10), t)
        x -= 400 * dt
        t += dt

    # Continue clear frames so envelope finalizes. It should not necessarily
    # command immediately: the cactus is still far from the Dino.
    finalized = False
    scheduled = None
    for _ in range(40):
        tel = d.process(frame_with_block(), t)
        if tel.envelope_duration_s is not None:
            finalized = True
        if tel.scheduled_command_in_s is not None:
            scheduled = tel.scheduled_command_in_s
            break
        t += dt

    assert finalized
    assert scheduled is not None
    assert scheduled > 0.5


def test_too_close_sensor_reports_required_extra_distance():
    d = make_detector(
        sensor_ahead_px=180,
        actuator_delay_s=0.20,
        jump_air_time_s=0.45,
        envelope_finalize_gap_s=0.10,
        min_speed_px_s=80,
        max_speed_px_s=2000,
    )
    t = 1.0
    dt = 0.01
    x = 100.0
    while x > 5:
        d.process(frame_with_block(int(x), int(x) + 10), t)
        x -= 800 * dt
        t += dt

    result = None
    for _ in range(30):
        result = d.process(frame_with_block(), t)
        if result.should_jump:
            break
        t += dt

    assert result is not None and result.should_jump
    assert result.trigger_reason == "sensor-too-close"
    assert result.late_by_s > 0
    assert result.required_extra_sensor_px > 0

def test_background_rebase_preserves_already_scheduled_command():
    d = make_detector(sensor_ahead_px=800, envelope_finalize_gap_s=0.05)
    t = 1.0
    dt = 0.01
    x = 90.0
    while x > 10:
        d.process(frame_with_block(int(x), int(x) + 10), t)
        x -= 400 * dt
        t += dt

    # Finalize and obtain a future command.
    while True:
        tel = d.process(frame_with_block(), t)
        t += dt
        if tel.scheduled_command_in_s is not None:
            break

    # Full scene inversion should rebase vision, not discard pending action.
    dark = np.full((32, 180), 10, dtype=np.uint8)
    tel = d.process(dark, t)
    assert tel.background_rebased

    fired = False
    for _ in range(300):
        t += dt
        tel = d.process(dark, t)
        if tel.should_jump:
            fired = True
            break
    assert fired
