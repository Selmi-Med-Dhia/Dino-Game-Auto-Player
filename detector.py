from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class DetectorConfig:
    radius: int = 14
    lookahead_px: int = 150
    behind_px: int = 24
    min_pixel_delta: float = 26.0
    noise_multiplier: float = 6.0
    column_occupancy: float = 0.14
    circle_occupancy: float = 0.055
    clear_occupancy: float = 0.025
    lead_time_s: float = 0.060
    min_speed_px_s: float = 120.0
    max_speed_px_s: float = 2200.0
    speed_alpha: float = 0.24
    min_component_width: int = 2
    cooldown_s: float = 0.115
    clear_hold_s: float = 0.030


@dataclass
class DetectionTelemetry:
    should_jump: bool
    occupancy: float
    threshold: float
    speed_px_s: float
    obstacle_distance_px: Optional[float]
    predicted_tti_s: Optional[float]
    armed: bool


class AutoJumpDetector:
    """Frame-by-frame detector for a horizontally approaching Dino obstacle.

    The selected point is represented by ``sensor_x`` inside the supplied frame.
    Frames are grayscale uint8 arrays. The detector calibrates to a clean
    background, detects either light-on-dark or dark-on-light contrast, tracks
    the nearest obstacle edge, estimates its horizontal speed and predicts
    when it reaches the selected sensor circle.
    """

    def __init__(self, width: int, height: int, sensor_x: int, sensor_y: int, config: DetectorConfig | None = None):
        self.width = int(width)
        self.height = int(height)
        self.sensor_x = int(sensor_x)
        self.sensor_y = int(sensor_y)
        self.config = config or DetectorConfig()

        yy, xx = np.ogrid[: self.height, : self.width]
        self.circle_mask = (xx - self.sensor_x) ** 2 + (yy - self.sensor_y) ** 2 <= self.config.radius ** 2

        self.background: Optional[np.ndarray] = None
        self.noise_level: float = 0.0
        self.threshold: float = self.config.min_pixel_delta

        self.prev_edge_x: Optional[float] = None
        self.prev_edge_t: Optional[float] = None
        self.speed_px_s: float = 0.0

        self.armed = True
        self.last_jump_t = -1e9
        self.clear_since: Optional[float] = None

    def calibrate(self, frames: list[np.ndarray]) -> None:
        if not frames:
            raise ValueError("Calibration needs at least one frame")
        stack = np.stack([self._validate_frame(f) for f in frames]).astype(np.float32)
        self.background = np.median(stack, axis=0)
        deviations = np.abs(stack - self.background)
        self.noise_level = float(np.median(deviations))
        self.threshold = max(self.config.min_pixel_delta, self.noise_level * self.config.noise_multiplier + 4.0)
        self.reset_tracking()

    def reset_tracking(self) -> None:
        self.prev_edge_x = None
        self.prev_edge_t = None
        self.speed_px_s = 0.0
        self.armed = True
        self.last_jump_t = -1e9
        self.clear_since = None

    def process(self, frame: np.ndarray, timestamp: float) -> DetectionTelemetry:
        frame_f = self._validate_frame(frame).astype(np.float32)
        if self.background is None:
            raise RuntimeError("Detector must be calibrated first")

        diff = np.abs(frame_f - self.background)
        changed = diff >= self.threshold

        circle_pixels = changed[self.circle_mask]
        occupancy = float(circle_pixels.mean()) if circle_pixels.size else 0.0

        # A 1-D view of contrast across the scan strip. This is more robust to
        # cactus shape than tracking a single pixel.
        column_fraction = changed.mean(axis=0)
        active_cols = column_fraction >= self.config.column_occupancy
        components = self._components(active_cols)

        obstacle_edge: Optional[float] = None
        obstacle_distance: Optional[float] = None

        # Pick the nearest component that is at, or approaching from, the right
        # side of the selected sensor. If it already overlaps the sensor, its
        # left edge can be slightly left of sensor_x.
        candidates = []
        for start, end in components:
            if end < self.sensor_x - self.config.radius:
                continue
            distance = float(start - self.sensor_x)
            candidates.append((abs(max(distance, 0.0)), start, end))
        if candidates:
            _, start, _ = min(candidates, key=lambda item: item[0])
            obstacle_edge = float(start)
            obstacle_distance = obstacle_edge - self.sensor_x
            self._update_speed(obstacle_edge, timestamp)
        else:
            self.prev_edge_x = None
            self.prev_edge_t = None

        predicted_tti: Optional[float] = None
        predictive_hit = False
        if obstacle_distance is not None:
            if obstacle_distance <= self.config.radius:
                predictive_hit = True
                predicted_tti = 0.0
            elif self.speed_px_s >= self.config.min_speed_px_s:
                predicted_tti = obstacle_distance / self.speed_px_s
                predictive_hit = 0.0 <= predicted_tti <= self.config.lead_time_s

        direct_hit = occupancy >= self.config.circle_occupancy
        obstacle_present = direct_hit or predictive_hit

        # Rearm only after the obstacle that caused the jump has moved through
        # a guard zone around the sensor. This matters for predictive jumps:
        # the circle itself may still be clear for a few frames *before* the
        # cactus arrives, so circle-only rearming could double-trigger.
        guard_ahead = max(
            self.config.radius * 2.0,
            min(90.0, self.speed_px_s * self.config.lead_time_s + self.config.radius + 8.0),
        )
        guard_left = self.sensor_x - self.config.radius
        guard_right = self.sensor_x + guard_ahead
        guard_active = any(end >= guard_left and start <= guard_right for start, end in components)

        clear_now = occupancy <= self.config.clear_occupancy and not guard_active
        if clear_now:
            if self.clear_since is None:
                self.clear_since = timestamp
            elif (timestamp - self.clear_since) >= self.config.clear_hold_s:
                self.armed = True
        else:
            self.clear_since = None

        should_jump = False
        if obstacle_present and self.armed and (timestamp - self.last_jump_t) >= self.config.cooldown_s:
            should_jump = True
            self.armed = False
            self.last_jump_t = timestamp
            self.clear_since = None

        return DetectionTelemetry(
            should_jump=should_jump,
            occupancy=occupancy,
            threshold=self.threshold,
            speed_px_s=self.speed_px_s,
            obstacle_distance_px=obstacle_distance,
            predicted_tti_s=predicted_tti,
            armed=self.armed,
        )

    def _update_speed(self, edge_x: float, timestamp: float) -> None:
        if self.prev_edge_x is not None and self.prev_edge_t is not None:
            dt = timestamp - self.prev_edge_t
            dx = self.prev_edge_x - edge_x  # positive for right-to-left motion
            if 0.001 <= dt <= 0.100 and dx > 0:
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
