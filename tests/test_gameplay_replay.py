"""Replay tests for far-ahead detection and simulated servo latency."""

import numpy as np

from detector import AutoJumpDetector, DetectorConfig

WIDTH = 560
HEIGHT = 54
SENSOR_X = 42
SENSOR_Y = 28


def scene(light=True, phase=0):
    bg = 247 if light else 20
    fg = 82 if light else 214
    frame = np.full((HEIGHT, WIDTH), bg, dtype=np.uint8)
    # Ground is deliberately below the optical sensor band.
    frame[50, :] = fg
    for x in range((phase % 19) - 19, WIDTH, 38):
        x0 = max(0, x)
        x1 = min(WIDTH, x + 3)
        if x1 > x0:
            frame[48:50, x0:x1] = fg
    return frame


def cactus(frame, x, light=True, width_scale=1.0):
    fg = 82 if light else 214
    x = int(round(x))
    trunk = max(7, int(round(9 * width_scale)))

    def rect(x0, x1, y0, y1):
        x0, x1 = max(0, x0), min(WIDTH, x1)
        y0, y1 = max(0, y0), min(HEIGHT, y1)
        if x1 > x0 and y1 > y0:
            frame[y0:y1, x0:x1] = fg

    rect(x, x + trunk, 17, 48)
    rect(x - 5, x + 3, 24, 30)
    rect(x - 5, x - 1, 20, 30)
    rect(x + trunk - 2, x + trunk + 6, 29, 35)
    rect(x + trunk + 3, x + trunk + 6, 24, 35)


def make_detector(light, sensor_ahead_px=1100.0, actuator_delay_s=0.160):
    cfg = DetectorConfig(
        radius=14,
        lookahead_px=420,
        sensor_ahead_px=sensor_ahead_px,
        actuator_delay_s=actuator_delay_s,
        jump_air_time_s=0.450,
        landing_margin_s=0.030,
        envelope_finalize_gap_s=0.120,
    )
    d = AutoJumpDetector(WIDTH, HEIGHT, SENSOR_X, SENSOR_Y, cfg)
    d.calibrate([scene(light) for _ in range(20)])
    return d


def replay(offsets, speed, light=True, sensor_ahead_px=1100.0, actuator_delay_s=0.160):
    d = make_detector(light, sensor_ahead_px, actuator_delay_s)
    dt = 1 / 120
    t = 1.0
    lead_x = 500.0
    commands = []
    last_obstacle_x = lead_x + max(offsets)

    # Continue long after obstacles leave the virtual sensor: future servo
    # commands are intentionally scheduled from the far-sensor event.
    total = (last_obstacle_x + sensor_ahead_px + 500) / speed + 2.0
    end_t = t + total

    while t < end_t:
        frame = scene(light, int(t * speed))
        for offset in offsets:
            cactus(frame, lead_x + offset, light)
        tel = d.process(frame, t)
        if tel.should_jump:
            commands.append((t, tel))
        lead_x -= speed * dt
        t += dt
    return commands


def test_far_sensor_servo_sim_lands_after_exit_across_game_speeds():
    # 1100 px is intentionally far enough for a 160 ms servo at these replay
    # speeds. This is the key behavior the Arduino version will rely on.
    for speed in (300.0, 450.0, 650.0, 900.0, 1200.0, 1500.0):
        commands = replay([0.0], speed, True)
        assert len(commands) == 1, (speed, len(commands))
        _t, tel = commands[0]
        assert tel.trigger_reason in {"scheduled-exit", "post-landing-rejump"}
        assert tel.landing_error_s is not None
        assert 0.015 <= tel.landing_error_s <= 0.055, (speed, tel.landing_error_s)
        assert tel.required_extra_sensor_px < 1.0


def test_dark_theme_still_works():
    for speed in (450.0, 900.0, 1300.0):
        assert len(replay([0.0], speed, False)) == 1


def test_servo_latency_moves_command_earlier_but_contact_time_stays_aligned():
    speed = 650.0
    no_delay = replay([0.0], speed, True, actuator_delay_s=0.0)[0]
    servo = replay([0.0], speed, True, actuator_delay_s=0.20)[0]

    command_shift = no_delay[0] - servo[0]
    # Command should be about 200 ms earlier when the servo itself needs 200 ms.
    assert 0.17 <= command_shift <= 0.23, command_shift

    # The physical contact times should remain nearly equal.
    contact_no_delay = no_delay[0]
    contact_servo = servo[0] + 0.20
    assert abs(contact_no_delay - contact_servo) <= 0.025


def test_tight_cactus_pair_is_one_envelope_and_one_command():
    speed = 650.0
    # 55 px ~= 85 ms at this speed: below the 120 ms envelope clear threshold.
    commands = replay([0.0, 55.0], speed, True)
    assert len(commands) == 1
    assert commands[0][1].envelope_duration_s is not None
    assert commands[0][1].envelope_duration_s > 0.08


def test_separate_following_cactus_gets_second_contact_after_first_landing():
    speed = 650.0
    commands = replay([0.0, 260.0], speed, True)
    assert len(commands) == 2
    (t0, tel0), (t1, tel1) = commands
    delay = tel0.actuator_delay_s
    first_landing = t0 + delay + 0.450
    second_contact = t1 + delay
    assert second_contact >= first_landing - 0.015


def test_close_sensor_is_flagged_instead_of_silently_pretending_timing_is_good():
    commands = replay([0.0], 1200.0, True, sensor_ahead_px=500.0, actuator_delay_s=0.20)
    assert len(commands) == 1
    tel = commands[0][1]
    assert tel.trigger_reason == "sensor-too-close"
    assert tel.late_by_s > 0
    assert tel.required_extra_sensor_px > 0


def chromium_airtime(internal_speed: float) -> float:
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


def test_airtime_constant_matches_chromium_normal_jump_order_of_magnitude():
    configured = DetectorConfig().jump_air_time_s
    for speed in (6.0, 9.0, 13.0):
        assert abs(configured - chromium_airtime(speed)) <= 0.045

def test_very_wide_cluster_uses_leading_edge_safety_if_exit_alignment_would_be_too_late():
    # Four tightly packed cacti make an unusually long envelope. The planner
    # must prioritize not colliding with the leading edge over perfect landing.
    commands = replay([0.0, 45.0, 90.0, 135.0, 180.0, 225.0], 650.0, True)
    assert len(commands) == 1
    assert commands[0][1].trigger_reason in {"entry-safety", "sensor-too-close"}
