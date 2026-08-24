from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class DetectorConfig:
    radius: int = 14
    lookahead_px: int = 600
    behind_px: int = 32
    min_pixel_delta: float = 22.0
    noise_multiplier: float = 6.0
    column_occupancy: float = 0.06
    circle_occupancy: float = 0.020
    clear_occupancy: float = 0.010

    # Leading-edge safety fallback. The normal planner below is driven by the
    # trailing edge so the landing occurs just after the whole cactus cluster.
    lead_time_s: float = 0.070

    min_speed_px_s: float = 120.0
    max_speed_px_s: float = 2600.0
    speed_alpha: float = 0.24
    min_component_width: int = 2
    cooldown_s: float = 0.115
    clear_hold_s: float = 0.030

    # Chrome's current normal jump physics are approximately 0.43-0.47 s
    # airborne for a tap (60 FPS, gravity .6, initial velocity around -10).
    # 0.44 s is deliberately near the middle of that range.
    jump_air_time_s: float = 0.440

    # The UI asks the user to put the detector roughly 70-130 px in front of
    # the dinosaur. This converts sensor-edge timing to the time the same edge
    # reaches the dinosaur. Small placement errors only shift timing by a few
    # tens of milliseconds and are covered by the margins below.
    sensor_ahead_px: float = 100.0

    # Aim to touch down shortly after the trailing edge clears the dinosaur.
    landing_margin_s: float = 0.025

    # If the time gap between two components is smaller than this, one jump is
    # safer than landing and trying to jump again. This also joins the pieces
    # of multi-cactus groups into one hazard.
    cluster_gap_time_s: float = 0.160
    cluster_gap_fallback_px: float = 90.0

    # Never wait for perfect landing alignment if the leading edge is this
    # close to the dinosaur. This is the last-safe-takeoff guard.
    min_entry_clearance_s: float = 0.115

    # Do not send another Space while Chrome is expected to still be airborne;
    # a key press during a jump is ignored and can leave the bot "stuck" for a
    # following cactus.
    rearm_before_landing_s: float = 0.015


@dataclass
class DetectionTelemetry:
    should_jump: bool
    occupancy: float
    threshold: float
    speed_px_s: float
    obstacle_distance_px: Optional[float]
    predicted_tti_s: Optional[float]
    armed: bool
    obstacle_exit_distance_px: Optional[float] = None
    predicted_exit_tti_s: Optional[float] = None
    predicted_entry_tti_dino_s: Optional[float] = None
    predicted_exit_tti_dino_s: Optional[float] = None
    cluster_width_px: Optional[float] = None
    landing_error_s: Optional[float] = None


class AutoJumpDetector:
    """Frame-by-frame detector for horizontally approaching Dino obstacles.

    The selected point is represented by ``sensor_x`` inside the supplied
    frame. Frames are grayscale uint8 arrays. The detector calibrates to a
    clean background, finds foreground columns, estimates right-to-left speed,
    groups close cacti, and schedules takeoff from the *trailing* edge so the
    dinosaur lands just after the complete hazard has passed.
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

        self.background: Optional[np.ndarray] = None
        self.noise_level: float = 0.0
        self.threshold: float = self.config.min_pixel_delta

        self.prev_edge_x: Optional[float] = None
        self.prev_edge_t: Optional[float] = None
        self.speed_px_s: float = 0.0

        self.armed = True
        self.last_jump_t = -1e9
        self.airborne_until_t = -1e9
        self.rearm_ready_t = -1e9
        self.last_jump_was_predictive = False
        self.clear_since: Optional[float] = None

    def calibrate(self, frames: list[np.ndarray]) -> None:
        if not frames:
            raise ValueError("Calibration needs at least one frame")
        stack = np.stack([self._validate_frame(f) for f in frames]).astype(np.float32)
        self.background = np.median(stack, axis=0)
        deviations = np.abs(stack - self.background)
        self.noise_level = float(np.median(deviations))
        self.threshold = max(
            self.config.min_pixel_delta,
            self.noise_level * self.config.noise_multiplier + 4.0,
        )
        self.reset_tracking()

    def reset_tracking(self) -> None:
        self.prev_edge_x = None
        self.prev_edge_t = None
        self.speed_px_s = 0.0
        self.armed = True
        self.last_jump_t = -1e9
        self.airborne_until_t = -1e9
        self.rearm_ready_t = -1e9
        self.last_jump_was_predictive = False
        self.clear_since = None

    def process(self, frame: np.ndarray, timestamp: float) -> DetectionTelemetry:
        frame_f = self._validate_frame(frame).astype(np.float32)
        if self.background is None:
            raise RuntimeError("Detector must be calibrated first")

        diff = np.abs(frame_f - self.background)
        changed = diff >= self.threshold

        circle_pixels = changed[self.circle_mask]
        occupancy = float(circle_pixels.mean()) if circle_pixels.size else 0.0

        # Collapse the scan band to foreground occupancy by x column. This is
        # the same robust area/interval idea used by successful open-source Dino
        # bots, but with adaptive background subtraction instead of one color.
        column_fraction = changed.mean(axis=0)
        active_cols = column_fraction >= self.config.column_occupancy
        components = self._components(active_cols)

        cluster_start: Optional[float] = None
        cluster_end: Optional[float] = None
        entry_distance: Optional[float] = None
        exit_distance: Optional[float] = None

        # Find the nearest visible hazard that has not fully passed the sensor.
        ahead = [
            (start, end)
            for start, end in components
            if end >= self.sensor_x - self.config.radius
        ]

        if ahead:
            first_start, first_end = ahead[0]
            cluster_start = float(first_start)
            cluster_end = float(first_end)

            # Use a time-based merge threshold once speed is known. A following
            # cactus too close for a clean land-and-rejump becomes part of the
            # same cluster, so the trailing edge is the exit of the *last* one.
            if self.speed_px_s >= self.config.min_speed_px_s:
                merge_gap = self.speed_px_s * self.config.cluster_gap_time_s
            else:
                merge_gap = self.config.cluster_gap_fallback_px
            merge_gap = float(np.clip(merge_gap, 45.0, 220.0))

            for start, end in ahead[1:]:
                gap = float(start) - cluster_end - 1.0
                if gap > merge_gap:
                    break
                cluster_end = float(end)

            entry_distance = cluster_start - self.sensor_x
            exit_distance = cluster_end - self.sensor_x
            self._update_speed(cluster_start, timestamp)
        else:
            self.prev_edge_x = None
            self.prev_edge_t = None

        entry_tti_sensor: Optional[float] = None
        exit_tti_sensor: Optional[float] = None
        entry_tti_dino: Optional[float] = None
        exit_tti_dino: Optional[float] = None
        landing_error: Optional[float] = None
        landing_due = False
        entry_urgent = False

        speed = self.speed_px_s
        if (
            entry_distance is not None
            and exit_distance is not None
            and speed >= self.config.min_speed_px_s
        ):
            entry_tti_sensor = entry_distance / speed
            exit_tti_sensor = exit_distance / speed

            # The marker is ahead of the dinosaur. Add that travel time so the
            # planner reasons about when each edge reaches the dinosaur itself.
            sensor_to_dino = self.config.sensor_ahead_px / speed
            entry_tti_dino = entry_tti_sensor + sensor_to_dino
            exit_tti_dino = exit_tti_sensor + sensor_to_dino

            # If we jump now, this is positive when landing would occur after
            # the cluster exit. Trigger near landing_margin_s so touch-down is
            # just beyond the trailing edge, leaving maximum room for the next
            # obstacle.
            landing_error = self.config.jump_air_time_s - exit_tti_dino
            landing_due = landing_error >= self.config.landing_margin_s

            # Wide clusters can make trailing-edge alignment too late for the
            # leading edge. This guard is the latest safe takeoff deadline.
            entry_urgent = entry_tti_dino <= self.config.min_entry_clearance_s

        # Fallback for the very first obstacle frames before speed has been
        # learned, or if a low-contrast shape reaches the circle unexpectedly.
        direct_hit = (
            occupancy >= self.config.circle_occupancy
            and exit_distance is not None
            and exit_distance >= 0
        )
        near_sensor_fallback = (
            entry_distance is not None
            and exit_distance is not None
            and exit_distance >= 0
            and entry_distance <= self.config.radius + 3
        )
        obstacle_present = landing_due or entry_urgent or direct_hit or near_sensor_fallback

        # Chrome ignores Space while the T-Rex is already jumping. The previous
        # clear-zone-only rearm could deadlock on back-to-back cacti because the
        # next cactus enters the sensor before touchdown. For a speed-tracked
        # jump we instead rearm from the predicted landing/clear time. If speed
        # was not known (direct emergency jump), keep the conservative clear
        # hold so a static/ambiguous blob cannot trigger repeatedly.
        if not self.armed and timestamp >= self.rearm_ready_t:
            if self.last_jump_was_predictive:
                self.armed = True
                self.clear_since = None
            else:
                clear_now = occupancy <= self.config.clear_occupancy
                if clear_now:
                    if self.clear_since is None:
                        self.clear_since = timestamp
                    elif timestamp - self.clear_since >= self.config.clear_hold_s:
                        self.armed = True
                else:
                    self.clear_since = None

        should_jump = False
        if (
            obstacle_present
            and self.armed
            and timestamp >= self.airborne_until_t - self.config.rearm_before_landing_s
            and (timestamp - self.last_jump_t) >= self.config.cooldown_s
        ):
            should_jump = True
            self.armed = False
            self.last_jump_t = timestamp
            self.airborne_until_t = timestamp + self.config.jump_air_time_s
            self.last_jump_was_predictive = speed >= self.config.min_speed_px_s
            rearm_at = self.airborne_until_t - self.config.rearm_before_landing_s
            if exit_tti_dino is not None:
                rearm_at = max(
                    rearm_at,
                    timestamp + max(0.0, exit_tti_dino) + self.config.landing_margin_s,
                )
            self.rearm_ready_t = rearm_at
            self.clear_since = None

        cluster_width = None
        if cluster_start is not None and cluster_end is not None:
            cluster_width = cluster_end - cluster_start + 1.0

        return DetectionTelemetry(
            should_jump=should_jump,
            occupancy=occupancy,
            threshold=self.threshold,
            speed_px_s=self.speed_px_s,
            obstacle_distance_px=entry_distance,
            predicted_tti_s=entry_tti_sensor,
            armed=self.armed,
            obstacle_exit_distance_px=exit_distance,
            predicted_exit_tti_s=exit_tti_sensor,
            predicted_entry_tti_dino_s=entry_tti_dino,
            predicted_exit_tti_dino_s=exit_tti_dino,
            cluster_width_px=cluster_width,
            landing_error_s=landing_error,
        )

    def _update_speed(self, edge_x: float, timestamp: float) -> None:
        if self.prev_edge_x is None or self.prev_edge_t is None:
            self.prev_edge_x = edge_x
            self.prev_edge_t = timestamp
            return

        dt = timestamp - self.prev_edge_t
        dx = self.prev_edge_x - edge_x  # positive for right-to-left motion

        # At high capture rates an edge often remains on the same integer pixel
        # for several frames. Keep the old timestamp until it actually moves;
        # otherwise a 1-pixel move divided by one tiny frame dt looks like an
        # impossible speed spike.
        if dx == 0 and dt <= 0.100:
            return

        if 0.001 <= dt <= 0.150 and dx > 0:
            instant = dx / dt
            if self.config.min_speed_px_s <= instant <= self.config.max_speed_px_s:
                if self.speed_px_s <= 0:
                    self.speed_px_s = instant
                else:
                    a = self.config.speed_alpha
                    self.speed_px_s = (1.0 - a) * self.speed_px_s + a * instant

        self.prev_edge_x = edge_x
        self.prev_edge_t = timestamp

    def _components(self, active: np.ndarray) -> list[tuple[int, int]]:
        components: list[tuple[int, int]] = []
        start: Optional[int] = None
        for i, value in enumerate(active):
            if value and start is None:
                start = i
            elif not value and start is not None:
                end = i - 1
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
