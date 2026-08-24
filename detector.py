from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import heapq
from typing import Optional

import numpy as np


@dataclass
class DetectorConfig:
    """Configuration for a hardware-faithful far-sensor Dino planner."""

    radius: int = 14
    lookahead_px: int = 420
    behind_px: int = 36

    # Vision: the selected point emulates a future LDR/phototransistor spot.
    min_pixel_delta: float = 20.0
    noise_multiplier: float = 6.0
    column_occupancy: float = 0.055
    circle_occupancy: float = 0.018
    min_component_width: int = 2
    vision_band_half_height: int = 17
    gate_half_width_px: int = 4
    max_global_change_fraction: float = 0.55
    background_learn_alpha: float = 0.015

    # Motion estimator. Screen vision gives a direct speed estimate; the
    # envelope telemetry below is also kept because that is easy to reproduce
    # later on Arduino with a light sensor.
    min_speed_px_s: float = 90.0
    max_speed_px_s: float = 3000.0
    bootstrap_speed_px_s: float = 420.0
    speed_history: int = 7
    speed_sample_max_dt_s: float = 0.180

    # Physical layout/timing.
    sensor_ahead_px: float = 650.0
    actuator_delay_s: float = 0.160  # command -> physical key contact
    jump_air_time_s: float = 0.450   # contact -> landing for short Space hold
    landing_margin_s: float = 0.030  # desired landing after trailing edge

    # Keep a short clear period before declaring the obstacle envelope over.
    # This merges fork gaps and tight cactus groups in the same spirit as
    # proven LDR-based hardware bots.
    envelope_finalize_gap_s: float = 0.120

    # The actual key contact still needs a little room before the leading edge.
    min_entry_clearance_s: float = 0.090

    cooldown_s: float = 0.070
    rearm_before_landing_s: float = 0.012


@dataclass
class DetectionTelemetry:
    should_jump: bool
    occupancy: float
    threshold: float
    speed_px_s: float
    planning_speed_px_s: float
    armed: bool

    obstacle_distance_px: Optional[float] = None
    obstacle_exit_distance_px: Optional[float] = None
    cluster_width_px: Optional[float] = None

    predicted_entry_tti_sensor_s: Optional[float] = None
    predicted_exit_tti_sensor_s: Optional[float] = None
    predicted_entry_tti_dino_s: Optional[float] = None
    predicted_exit_tti_dino_s: Optional[float] = None

    actuator_delay_s: float = 0.0
    landing_error_s: Optional[float] = None
    trigger_reason: str = "none"
    background_rebased: bool = False

    envelope_active: bool = False
    envelope_duration_s: Optional[float] = None
    rolling_min_envelope_s: Optional[float] = None
    scheduled_command_in_s: Optional[float] = None
    late_by_s: float = 0.0
    required_extra_sensor_px: float = 0.0


@dataclass
class _PendingCommand:
    command_t: float
    expected_contact_t: float
    expected_landing_t: float
    exit_at_dino_t: float
    planning_speed: float
    envelope_duration_s: float
    late_by_s: float
    reason: str


class AutoJumpDetector:
    """Far-ahead virtual optical sensor with delayed-actuator scheduling.

    The important difference from a near-Dino pixel bot is that detection and
    action are separate events:

    1. A cactus/group crosses the far virtual sensor.
    2. Its envelope is finalized after a short clear gap.
    3. The trailing-edge crossing time is projected to the Dino using speed.
    4. A *future servo command* is scheduled so physical key contact + jump
       airtime ends just after that trailing edge reaches the Dino.

    This is intentionally close to what an Arduino + optical sensor + servo can
    implement later. The PC version uses screen vision only to make speed
    estimation and testing more reliable.
    """

    def __init__(
        self,
        width: int,
        height: int,
        sensor_x: int,
        sensor_y: int,
        config: DetectorConfig | None = None,
    ):
        self.width = int(width)
        self.height = int(height)
        self.sensor_x = int(sensor_x)
        self.sensor_y = int(sensor_y)
        self.config = config or DetectorConfig()

        yy, xx = np.ogrid[: self.height, : self.width]
        self.circle_mask = (
            (xx - self.sensor_x) ** 2 + (yy - self.sensor_y) ** 2
            <= self.config.radius ** 2
        )

        half = max(5, int(self.config.vision_band_half_height))
        self.band_top = max(0, self.sensor_y - half)
        self.band_bottom = min(self.height, self.sensor_y + half + 1)
        if self.band_bottom - self.band_top < 4:
            self.band_top = 0
            self.band_bottom = self.height

        self.background: Optional[np.ndarray] = None
        self.noise_level = 0.0
        self.threshold = float(self.config.min_pixel_delta)

        self.prev_edge_x: Optional[float] = None
        self.prev_edge_t: Optional[float] = None
        self.speed_samples: deque[float] = deque(maxlen=max(3, int(self.config.speed_history)))
        self.speed_px_s = 0.0

        # Hardware-like obstacle envelope state at the virtual sensor gate.
        self.envelope_active = False
        self.envelope_start_t: Optional[float] = None
        self.envelope_last_seen_t: Optional[float] = None
        self.envelope_speed_samples: list[float] = []
        self.envelope_durations: deque[float] = deque(maxlen=5)
        self.last_finalized_envelope_s: Optional[float] = None

        # Future servo commands, ordered by command time.
        self.pending: list[tuple[float, int, _PendingCommand]] = []
        self.pending_seq = 0

        self.armed = True
        self.last_command_t = -1e9
        self.expected_contact_t = -1e9
        self.expected_landing_t = -1e9
        self.last_planned_landing_t = -1e9

        self.last_late_by_s = 0.0
        self.last_required_extra_sensor_px = 0.0

    def calibrate(self, frames: list[np.ndarray]) -> None:
        if not frames:
            raise ValueError("Calibration needs at least one frame")
        stack = np.stack([self._validate_frame(frame) for frame in frames]).astype(np.float32)
        self.background = np.median(stack, axis=0)
        deviations = np.abs(stack - self.background)
        band_dev = deviations[:, self.band_top : self.band_bottom, :]
        self.noise_level = float(np.median(band_dev))
        self.threshold = max(
            float(self.config.min_pixel_delta),
            self.noise_level * float(self.config.noise_multiplier) + 4.0,
        )
        self.reset_tracking()

    def reset_tracking(self) -> None:
        self.prev_edge_x = None
        self.prev_edge_t = None
        self.speed_samples.clear()
        self.speed_px_s = 0.0

        self.envelope_active = False
        self.envelope_start_t = None
        self.envelope_last_seen_t = None
        self.envelope_speed_samples.clear()
        self.envelope_durations.clear()
        self.last_finalized_envelope_s = None

        self.pending.clear()
        self.pending_seq = 0

        self.armed = True
        self.last_command_t = -1e9
        self.expected_contact_t = -1e9
        self.expected_landing_t = -1e9
        self.last_planned_landing_t = -1e9
        self.last_late_by_s = 0.0
        self.last_required_extra_sensor_px = 0.0

    def process(self, frame: np.ndarray, timestamp: float) -> DetectionTelemetry:
        arr = self._validate_frame(frame)
        if self.background is None:
            raise RuntimeError("Detector must be calibrated first")

        frame_f = arr.astype(np.float32)
        diff = np.abs(frame_f - self.background)
        changed = diff >= self.threshold
        band_changed = changed[self.band_top : self.band_bottom, :]

        # Detect day/night inversion or another full-band scene change. Do not
        # destroy already scheduled servo commands when rebasing the background.
        global_change = float(band_changed.mean()) if band_changed.size else 0.0
        if global_change >= self.config.max_global_change_fraction:
            self.background = frame_f.copy()
            self.prev_edge_x = None
            self.prev_edge_t = None
            return self._finish_frame(
                timestamp,
                occupancy=0.0,
                planning_speed=self._planning_speed(),
                trigger_reason="background-rebase",
                background_rebased=True,
            )

        circle_pixels = changed[self.circle_mask]
        occupancy = float(circle_pixels.mean()) if circle_pixels.size else 0.0

        column_fraction = band_changed.mean(axis=0)
        active_cols = column_fraction >= self.config.column_occupancy
        components = self._components(active_cols)

        ahead = [
            (start, end)
            for start, end in components
            if end >= self.sensor_x - self.config.radius
        ]

        cluster_start = None
        cluster_end = None
        entry_distance = None
        exit_distance = None
        planning_speed = self._planning_speed()

        if ahead:
            # Nearest component is used for motion. All components that are close
            # enough to the sensor gate naturally become one time envelope below.
            first_start, first_end = ahead[0]
            cluster_start = float(first_start)
            cluster_end = float(first_end)
            self._update_speed(cluster_start, timestamp)
            planning_speed = self._planning_speed()

            # For live telemetry only, merge visible nearby components using the
            # same time-gap idea used by the physical envelope state.
            visible_gap_px = planning_speed * self.config.envelope_finalize_gap_s
            for start, end in ahead[1:]:
                if float(start) - cluster_end - 1.0 > visible_gap_px:
                    break
                cluster_end = float(end)

            entry_distance = cluster_start - self.sensor_x
            exit_distance = cluster_end - self.sensor_x
        else:
            self.prev_edge_x = None
            self.prev_edge_t = None
            alpha = float(self.config.background_learn_alpha)
            if alpha > 0 and not self.envelope_active:
                self.background = (1.0 - alpha) * self.background + alpha * frame_f

        # Exact virtual sensor gate. A hardware LDR has a finite spot, so use a
        # few columns around the selected x instead of one brittle pixel.
        gate_left = self.sensor_x - int(self.config.gate_half_width_px)
        gate_right = self.sensor_x + int(self.config.gate_half_width_px)
        gate_active = any(end >= gate_left and start <= gate_right for start, end in components)
        gate_active = gate_active or occupancy >= self.config.circle_occupancy

        if gate_active:
            if not self.envelope_active:
                self.envelope_active = True
                self.envelope_start_t = timestamp
                self.envelope_speed_samples = []
            self.envelope_last_seen_t = timestamp
            if self.speed_px_s >= self.config.min_speed_px_s:
                self.envelope_speed_samples.append(self.speed_px_s)
        elif self.envelope_active and self.envelope_last_seen_t is not None:
            if timestamp - self.envelope_last_seen_t >= self.config.envelope_finalize_gap_s:
                self._finalize_envelope(timestamp)

        entry_tti_sensor = None
        exit_tti_sensor = None
        entry_tti_dino = None
        exit_tti_dino = None
        if entry_distance is not None and exit_distance is not None and planning_speed > 0:
            entry_tti_sensor = entry_distance / planning_speed
            exit_tti_sensor = exit_distance / planning_speed
            travel = self.config.sensor_ahead_px / planning_speed
            entry_tti_dino = entry_tti_sensor + travel
            exit_tti_dino = exit_tti_sensor + travel

        cluster_width = None
        if cluster_start is not None and cluster_end is not None:
            cluster_width = cluster_end - cluster_start + 1.0

        telemetry = self._finish_frame(
            timestamp,
            occupancy=occupancy,
            planning_speed=planning_speed,
            trigger_reason="none",
        )
        telemetry.obstacle_distance_px = entry_distance
        telemetry.obstacle_exit_distance_px = exit_distance
        telemetry.cluster_width_px = cluster_width
        telemetry.predicted_entry_tti_sensor_s = entry_tti_sensor
        telemetry.predicted_exit_tti_sensor_s = exit_tti_sensor
        telemetry.predicted_entry_tti_dino_s = entry_tti_dino
        telemetry.predicted_exit_tti_dino_s = exit_tti_dino
        return telemetry

    def _finalize_envelope(self, now: float) -> None:
        assert self.envelope_start_t is not None
        assert self.envelope_last_seen_t is not None

        # Use the actual last-active instant as the trailing-edge crossing time;
        # the clear gap is only debounce and must not shift obstacle geometry.
        exit_sensor_t = self.envelope_last_seen_t
        entry_sensor_t = self.envelope_start_t
        duration = max(0.001, exit_sensor_t - entry_sensor_t)
        self.envelope_durations.append(duration)
        self.last_finalized_envelope_s = duration

        if self.envelope_speed_samples:
            speed = float(np.median(np.asarray(self.envelope_speed_samples)))
        else:
            speed = self._planning_speed()
        speed = float(np.clip(speed, self.config.min_speed_px_s, self.config.max_speed_px_s))

        sensor_travel_s = self.config.sensor_ahead_px / speed
        entry_at_dino_t = entry_sensor_t + sensor_travel_s
        exit_at_dino_t = exit_sensor_t + sensor_travel_s

        # Primary target: physical key contact early enough that touchdown is
        # just after the trailing edge. A very wide group can make that target
        # unsafe for the leading edge, so cap it at the last safe contact time.
        exit_aligned_contact_t = (
            exit_at_dino_t
            + self.config.landing_margin_s
            - self.config.jump_air_time_s
        )
        latest_safe_contact_t = entry_at_dino_t - self.config.min_entry_clearance_s
        desired_contact_t = min(exit_aligned_contact_t, latest_safe_contact_t)
        used_entry_safety = latest_safe_contact_t < exit_aligned_contact_t
        ideal_command_t = desired_contact_t - self.config.actuator_delay_s

        # A second servo command may be issued while the first jump is airborne
        # provided its *physical contact* occurs after the previous landing.
        earliest_command_for_rejump = (
            self.last_planned_landing_t
            + self.config.rearm_before_landing_s
            - self.config.actuator_delay_s
        )
        command_t = max(ideal_command_t, earliest_command_for_rejump)
        expected_contact_t = command_t + self.config.actuator_delay_s
        expected_landing_t = expected_contact_t + self.config.jump_air_time_s

        # If we only learned the full cluster after its ideal servo-command time,
        # the virtual sensor is physically too close for this latency/speed. We
        # still command ASAP, but report exactly how much farther ahead it needs
        # to be. This is the tuning metric that transfers to the Arduino build.
        late_by = max(0.0, now - command_t)
        required_extra_px = late_by * speed
        if late_by > 0:
            command_t = now
            expected_contact_t = command_t + self.config.actuator_delay_s
            expected_landing_t = expected_contact_t + self.config.jump_air_time_s

        reason = "entry-safety" if used_entry_safety else "scheduled-exit"
        if late_by > 0:
            reason = "sensor-too-close"
        elif command_t > ideal_command_t + 1e-6:
            reason = "post-landing-rejump"

        cmd = _PendingCommand(
            command_t=command_t,
            expected_contact_t=expected_contact_t,
            expected_landing_t=expected_landing_t,
            exit_at_dino_t=exit_at_dino_t,
            planning_speed=speed,
            envelope_duration_s=duration,
            late_by_s=late_by,
            reason=reason,
        )
        self.pending_seq += 1
        heapq.heappush(self.pending, (command_t, self.pending_seq, cmd))
        self.last_planned_landing_t = expected_landing_t
        self.last_late_by_s = late_by
        self.last_required_extra_sensor_px = required_extra_px

        self.envelope_active = False
        self.envelope_start_t = None
        self.envelope_last_seen_t = None
        self.envelope_speed_samples = []

    def _finish_frame(
        self,
        timestamp: float,
        *,
        occupancy: float,
        planning_speed: float,
        trigger_reason: str,
        background_rebased: bool = False,
    ) -> DetectionTelemetry:
        # Re-arm based on expected physical landing, not command time.
        if not self.armed and timestamp >= self.expected_landing_t - self.config.rearm_before_landing_s:
            self.armed = True

        should_jump = False
        reason = trigger_reason
        landing_error = None
        command_exit_tti = None

        if self.pending:
            _due_t, _seq, cmd = self.pending[0]
            if self.armed and timestamp >= cmd.command_t and timestamp - self.last_command_t >= self.config.cooldown_s:
                heapq.heappop(self.pending)
                should_jump = True
                reason = cmd.reason
                self.armed = False
                self.last_command_t = timestamp
                self.expected_contact_t = timestamp + self.config.actuator_delay_s
                self.expected_landing_t = self.expected_contact_t + self.config.jump_air_time_s
                command_exit_tti = cmd.exit_at_dino_t - timestamp
                # Actual predicted landing relative to the projected cluster exit.
                landing_error = self.expected_landing_t - cmd.exit_at_dino_t

        rolling_min = min(self.envelope_durations) if self.envelope_durations else None
        scheduled_in = None
        if self.pending:
            scheduled_in = max(0.0, self.pending[0][0] - timestamp)

        return DetectionTelemetry(
            should_jump=should_jump,
            occupancy=occupancy,
            threshold=self.threshold,
            speed_px_s=self.speed_px_s,
            planning_speed_px_s=planning_speed,
            armed=self.armed,
            predicted_exit_tti_dino_s=command_exit_tti,
            actuator_delay_s=self.config.actuator_delay_s,
            landing_error_s=landing_error,
            trigger_reason=reason,
            background_rebased=background_rebased,
            envelope_active=self.envelope_active,
            envelope_duration_s=self.last_finalized_envelope_s,
            rolling_min_envelope_s=rolling_min,
            scheduled_command_in_s=scheduled_in,
            late_by_s=self.last_late_by_s,
            required_extra_sensor_px=self.last_required_extra_sensor_px,
        )

    def _planning_speed(self) -> float:
        if self.speed_px_s >= self.config.min_speed_px_s:
            return self.speed_px_s
        return float(self.config.bootstrap_speed_px_s)

    def _update_speed(self, edge_x: float, timestamp: float) -> None:
        if self.prev_edge_x is None or self.prev_edge_t is None:
            self.prev_edge_x = edge_x
            self.prev_edge_t = timestamp
            return

        dt = timestamp - self.prev_edge_t
        dx = self.prev_edge_x - edge_x
        if dx == 0 and dt <= self.config.speed_sample_max_dt_s:
            return

        if 0.001 <= dt <= self.config.speed_sample_max_dt_s and dx > 0:
            instant = dx / dt
            if self.config.min_speed_px_s <= instant <= self.config.max_speed_px_s:
                self.speed_samples.append(float(instant))
                self.speed_px_s = float(np.median(np.asarray(self.speed_samples)))

        self.prev_edge_x = edge_x
        self.prev_edge_t = timestamp

    def _components(self, active: np.ndarray) -> list[tuple[int, int]]:
        components: list[tuple[int, int]] = []
        start: Optional[int] = None
        for index, value in enumerate(active):
            if value and start is None:
                start = index
            elif not value and start is not None:
                end = index - 1
                if end - start + 1 >= self.config.min_component_width:
                    components.append((start, end))
                start = None
        if start is not None:
            end = len(active) - 1
            if end - start + 1 >= self.config.min_component_width:
                components.append((start, end))
        return components

    def _validate_frame(self, frame: np.ndarray) -> np.ndarray:
        arr = np.asarray(frame)
        if arr.shape != (self.height, self.width):
            raise ValueError(f"Expected frame {(self.height, self.width)}, got {arr.shape}")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr
