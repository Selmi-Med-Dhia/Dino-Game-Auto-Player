from __future__ import annotations

import ctypes
import heapq
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
from mss import MSS
from pynput import keyboard
from pynput.keyboard import Controller as KeyboardController
from pynput.mouse import Button, Controller as MouseController

from detector import AutoJumpDetector, DetectorConfig

APP_TITLE = "Dino Auto-Player — Servo Simulation"
STOP_KEY = keyboard.Key.f8


def enable_dpi_awareness() -> str:
    """Keep Tk/pynput coordinates aligned with MSS physical pixels on Windows."""
    if sys.platform != "win32":
        return "not-windows"
    try:
        if ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            return "per-monitor-v2"
    except (AttributeError, OSError):
        pass
    try:
        result = ctypes.windll.shcore.SetProcessDpiAwareness(2)
        if result in (0, -2147024891):
            return "per-monitor"
    except (AttributeError, OSError):
        pass
    try:
        if ctypes.windll.user32.SetProcessDPIAware():
            return "system"
    except (AttributeError, OSError):
        pass
    return "unknown"


DPI_MODE = enable_dpi_awareness()  # Must happen before creating Tk windows.


class WinPoint(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


def physical_cursor_pos(fallback_x: int, fallback_y: int) -> tuple[int, int]:
    if sys.platform == "win32":
        try:
            point = WinPoint()
            if ctypes.windll.user32.GetPhysicalCursorPos(ctypes.byref(point)):
                return int(point.x), int(point.y)
        except (AttributeError, OSError):
            pass
    return int(fallback_x), int(fallback_y)


class Marker:
    def __init__(self, root: tk.Tk, x: int, y: int, radius: int, color: str):
        size = radius * 2 + 18
        self.window = tk.Toplevel(root)
        self.window.overrideredirect(True)
        self.window.attributes("-topmost", True)
        self.window.configure(bg="#ff00ff")
        try:
            self.window.wm_attributes("-transparentcolor", "#ff00ff")
        except tk.TclError:
            pass
        self.window.geometry(f"{size}x{size}{x - size // 2:+d}{y - size // 2:+d}")
        canvas = tk.Canvas(
            self.window,
            width=size,
            height=size,
            bg="#ff00ff",
            highlightthickness=0,
        )
        canvas.pack()
        pad = 7
        canvas.create_oval(pad, pad, size - pad, size - pad, outline=color, width=3)
        canvas.create_line(size // 2 - 6, size // 2, size // 2 + 6, size // 2, fill=color, width=2)
        canvas.create_line(size // 2, size // 2 - 6, size // 2, size // 2 + 6, fill=color, width=2)
        self.window.update_idletasks()
        self._make_click_through()

    def _make_click_through(self) -> None:
        if sys.platform != "win32":
            return
        try:
            hwnd = self.window.winfo_id()
            user32 = ctypes.windll.user32
            style = user32.GetWindowLongW(hwnd, -20)
            user32.SetWindowLongW(hwnd, -20, style | 0x20 | 0x80000 | 0x80 | 0x08000000)
        except Exception:
            pass

    def hide(self) -> None:
        self.window.withdraw()

    def show(self) -> None:
        self.window.deiconify()

    def destroy(self) -> None:
        try:
            self.window.destroy()
        except tk.TclError:
            pass


class TwoPointSelector:
    """Select Dino front first, then a far-ahead virtual optical sensor."""

    def __init__(self, root: tk.Tk, on_complete, on_cancel):
        self.root = root
        self.on_complete = on_complete
        self.on_cancel = on_cancel
        self.dino_point: tuple[int, int] | None = None

        self.overlay = tk.Toplevel(root)
        self.overlay.overrideredirect(True)
        self.overlay.attributes("-topmost", True)
        self.overlay.attributes("-alpha", 0.24)
        self.overlay.configure(bg="black", cursor="crosshair")

        if sys.platform == "win32":
            user32 = ctypes.windll.user32
            vx, vy = user32.GetSystemMetrics(76), user32.GetSystemMetrics(77)
            vw, vh = user32.GetSystemMetrics(78), user32.GetSystemMetrics(79)
        else:
            vx = vy = 0
            vw, vh = root.winfo_screenwidth(), root.winfo_screenheight()

        self.overlay.geometry(f"{vw}x{vh}{vx:+d}{vy:+d}")
        self.overlay.bind("<Button-1>", self._clicked)
        self.overlay.bind("<Escape>", lambda _event: self.cancel())
        self.label = tk.Label(
            self.overlay,
            text="1/2 — Click the FRONT of the Dino (nose/chest x-position)\nThis is only a distance reference. Esc cancels.",
            fg="white",
            bg="black",
            font=("Segoe UI", 18, "bold"),
            padx=20,
            pady=12,
        )
        self.label.place(relx=0.5, rely=0.08, anchor="n")
        self.overlay.focus_force()
        self.overlay.grab_set()

    def _clicked(self, event: tk.Event) -> None:
        logical_x = self.overlay.winfo_rootx() + int(event.x)
        logical_y = self.overlay.winfo_rooty() + int(event.y)
        point = physical_cursor_pos(logical_x, logical_y)

        if self.dino_point is None:
            self.dino_point = point
            self.label.config(
                text=(
                    "2/2 — Click an EMPTY point FAR AHEAD of the Dino at cactus-body height\n"
                    "Aim for roughly 700–1100 px ahead if your screen allows it. This is the virtual future optical sensor."
                )
            )
            return

        sensor_point = point
        self.overlay.grab_release()
        self.overlay.destroy()
        self.on_complete(self.dino_point, sensor_point)

    def cancel(self) -> None:
        try:
            self.overlay.grab_release()
        except tk.TclError:
            pass
        self.overlay.destroy()
        self.on_cancel()


class DelayedKeyActuator:
    """Simulate a servo: command now, physical key contact after a delay."""

    def __init__(self, controller: KeyboardController, stop_event: threading.Event):
        self.controller = controller
        self.stop_event = stop_event
        self.cv = threading.Condition()
        self.jobs: list[tuple[float, int, float]] = []
        self.counter = 0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def schedule(self, contact_time: float, hold_s: float) -> None:
        with self.cv:
            self.counter += 1
            heapq.heappush(self.jobs, (float(contact_time), self.counter, float(hold_s)))
            self.cv.notify_all()

    def _run(self) -> None:
        while not self.stop_event.is_set():
            with self.cv:
                if not self.jobs:
                    self.cv.wait(timeout=0.05)
                    continue
                due, _seq, hold_s = self.jobs[0]
                wait = due - time.perf_counter()
                if wait > 0:
                    self.cv.wait(timeout=min(wait, 0.05))
                    continue
                heapq.heappop(self.jobs)

            if self.stop_event.is_set():
                return
            self.controller.press(keyboard.Key.space)
            if self.stop_event.wait(max(0.005, hold_s)):
                self.controller.release(keyboard.Key.space)
                return
            self.controller.release(keyboard.Key.space)

    def stop(self) -> None:
        with self.cv:
            self.jobs.clear()
            self.cv.notify_all()
        self.thread.join(timeout=0.5)


class DinoAutoPlayerApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title(APP_TITLE)
        self.root.geometry("560x700")
        self.root.minsize(530, 690)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.dino_point: tuple[int, int] | None = None
        self.sensor_point: tuple[int, int] | None = None
        self.dino_marker: Marker | None = None
        self.sensor_marker: Marker | None = None

        self.stop_event = threading.Event()
        self.running = False
        self.keyboard = KeyboardController()
        self.mouse = MouseController()
        self.listener = keyboard.Listener(on_press=self._global_key)
        self.listener.start()
        self.messages: queue.Queue[tuple[str, object]] = queue.Queue()
        self.jump_count = 0

        self.status_var = tk.StringVar(value="Select the Dino and far sensor positions.")
        self.distance_var = tk.StringVar(value="Not selected")
        self.speed_var = tk.StringVar(value="—")
        self.exit_var = tk.StringVar(value="—")
        self.jump_var = tk.StringVar(value="0")
        self.reason_var = tk.StringVar(value="—")
        self.envelope_var = tk.StringVar(value="—")
        self.timing_var = tk.StringVar(value="—")

        self.actuator_delay_ms = tk.DoubleVar(value=160.0)
        self.key_hold_ms = tk.DoubleVar(value=80.0)
        self.landing_margin_ms = tk.DoubleVar(value=30.0)
        self.radius = tk.IntVar(value=14)
        self.sensitivity = tk.DoubleVar(value=20.0)

        self._build_ui()
        self.root.after(50, self._poll_messages)
        self.root.after(250, self.select_points)

    def _build_ui(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("vista")
        except tk.TclError:
            pass

        outer = ttk.Frame(self.root, padding=18)
        outer.pack(fill="both", expand=True)

        ttk.Label(
            outer,
            text="Dino Auto-Player — Servo Simulation",
            font=("Segoe UI", 18, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text=(
                "This version is tuned like the future Arduino build: place a virtual optical sensor far ahead of the Dino, then model the servo's command-to-key-contact delay. "
                "The planner still targets the EXIT of the complete cactus cluster."
            ),
            wraplength=520,
        ).pack(anchor="w", pady=(6, 14))

        info = ttk.LabelFrame(outer, text="Live telemetry", padding=12)
        info.pack(fill="x")
        rows = (
            ("Virtual sensor distance", self.distance_var),
            ("Measured / planning speed", self.speed_var),
            ("Hazard exit at Dino", self.exit_var),
            ("Sensor envelope", self.envelope_var),
            ("Sensor timing margin", self.timing_var),
            ("Last trigger", self.reason_var),
            ("Servo commands", self.jump_var),
        )
        for row, (label, variable) in enumerate(rows):
            ttk.Label(info, text=label).grid(row=row, column=0, sticky="w", pady=2)
            ttk.Label(info, textvariable=variable, font=("Segoe UI", 10, "bold")).grid(row=row, column=1, sticky="e", pady=2)
        info.columnconfigure(1, weight=1)

        settings = ttk.LabelFrame(outer, text="Hardware-transferable tuning", padding=12)
        settings.pack(fill="x", pady=12)
        setting_rows = (
            ("Servo/contact delay (ms)", self.actuator_delay_ms, 0, 400),
            ("Space contact/hold (ms)", self.key_hold_ms, 20, 180),
            ("Land after exit margin (ms)", self.landing_margin_ms, 0, 90),
            ("Sensor radius", self.radius, 8, 24),
            ("Min contrast", self.sensitivity, 10, 55),
        )
        for row, (label, variable, low, high) in enumerate(setting_rows):
            ttk.Label(settings, text=label).grid(row=row, column=0, sticky="w", pady=3)
            ttk.Scale(
                settings,
                from_=low,
                to=high,
                variable=variable,
                orient="horizontal",
            ).grid(row=row, column=1, sticky="ew", padx=10, pady=3)
            ttk.Label(settings, textvariable=variable, width=7).grid(row=row, column=2, sticky="e")
        settings.columnconfigure(1, weight=1)

        buttons = ttk.Frame(outer)
        buttons.pack(fill="x")
        self.select_btn = ttk.Button(buttons, text="Select Dino + sensor", command=self.select_points)
        self.select_btn.pack(side="left")
        self.start_btn = ttk.Button(buttons, text="Start simulation", command=self.start)
        self.start_btn.pack(side="left", padx=8)
        self.stop_btn = ttk.Button(buttons, text="Stop (F8)", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left")

        ttk.Label(outer, textvariable=self.status_var, wraplength=520).pack(anchor="w", pady=(12, 0))
        ttk.Label(
            outer,
            text=(
                "For Arduino transfer: keep the same physical screen distance between the Dino and the optical sensor, then set Servo/contact delay to the measured command→key-contact time of your servo linkage. "
                "The PC intentionally delays the actual Space press by that amount."
            ),
            wraplength=520,
        ).pack(anchor="w", pady=(8, 0))

    def select_points(self) -> None:
        if self.running:
            self.stop()
            return
        self.root.withdraw()
        TwoPointSelector(self.root, self._points_selected, self._selection_cancelled)

    def _selection_cancelled(self) -> None:
        self.root.deiconify()
        self.root.lift()
        self.status_var.set("Point selection cancelled.")

    def _points_selected(self, dino_point: tuple[int, int], sensor_point: tuple[int, int]) -> None:
        distance = sensor_point[0] - dino_point[0]
        if distance <= 80:
            self.root.deiconify()
            self.root.lift()
            messagebox.showwarning(
                APP_TITLE,
                "The virtual sensor should be well to the RIGHT of the Dino. Select it at least ~250 px ahead; 700–1100 px is a better servo-simulation range when the screen allows it.",
            )
            self.root.after(50, self.select_points)
            return

        self.dino_point = dino_point
        self.sensor_point = sensor_point
        self.distance_var.set(f"{distance} physical px ahead")

        if self.dino_marker:
            self.dino_marker.destroy()
        if self.sensor_marker:
            self.sensor_marker.destroy()
        self.dino_marker = Marker(self.root, dino_point[0], dino_point[1], 10, "#61d095")
        self.sensor_marker = Marker(self.root, sensor_point[0], sensor_point[1], int(self.radius.get()), "#00d8ff")

        self.root.deiconify()
        self.root.lift()
        self.status_var.set(
            f"Virtual sensor is {distance}px ahead. Leave Chrome Dino visible and click Start simulation."
        )

    def start(self) -> None:
        if self.running:
            return
        if self.dino_point is None or self.sensor_point is None:
            self.select_points()
            return

        self.stop_event.clear()
        self.running = True
        self.jump_count = 0
        self.jump_var.set("0")
        self.start_btn.config(state="disabled")
        self.select_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_var.set("Running servo simulation… press F8 to stop.")

        self.root.withdraw()
        if self.dino_marker:
            self.dino_marker.hide()
        if self.sensor_marker:
            self.sensor_marker.hide()

        settings = {
            "radius": int(self.radius.get()),
            "sensitivity": float(self.sensitivity.get()),
            "actuator_delay_s": float(self.actuator_delay_ms.get()) / 1000.0,
            "key_hold_s": float(self.key_hold_ms.get()) / 1000.0,
            "landing_margin_s": float(self.landing_margin_ms.get()) / 1000.0,
            "sensor_ahead_px": float(self.sensor_point[0] - self.dino_point[0]),
        }
        threading.Thread(target=self._run_detector, args=(settings,), daemon=True).start()

    def stop(self) -> None:
        if self.running:
            self.stop_event.set()

    def _restore(self) -> None:
        self.root.deiconify()
        self.root.lift()
        if self.dino_marker:
            self.dino_marker.show()
        if self.sensor_marker:
            self.sensor_marker.show()
        self.start_btn.config(state="normal")
        self.select_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_var.set("Stopped. Tune the simulated hardware latency or choose a new far sensor position.")

    def _run_detector(self, settings: dict[str, float]) -> None:
        assert self.sensor_point is not None
        sensor_screen_x, sensor_screen_y = self.sensor_point
        radius = int(settings["radius"])
        behind = max(36, radius + 14)
        lookahead = 420
        capture_h = max(54, radius * 2 + 22)
        actuator: DelayedKeyActuator | None = None

        try:
            time.sleep(0.18)
            self.mouse.position = (sensor_screen_x, sensor_screen_y)
            self.mouse.click(Button.left, 1)
            time.sleep(0.18)

            with MSS() as sct:
                virtual = sct.monitors[0]
                vleft, vtop = int(virtual["left"]), int(virtual["top"])
                vright = vleft + int(virtual["width"])
                vbottom = vtop + int(virtual["height"])

                left = max(vleft, sensor_screen_x - behind)
                right = min(vright, sensor_screen_x + lookahead)
                top = max(vtop, sensor_screen_y - capture_h // 2)
                bottom = min(vbottom, sensor_screen_y + capture_h // 2)
                width, height = right - left, bottom - top
                if width < 40 or height < 20:
                    raise RuntimeError("The virtual sensor is too close to a desktop edge. Move it farther inside the screen.")

                sensor_x = sensor_screen_x - left
                sensor_y = sensor_screen_y - top
                region = {"left": left, "top": top, "width": width, "height": height}

                config = DetectorConfig(
                    radius=min(radius, max(4, min(width, height) // 2 - 1)),
                    lookahead_px=right - sensor_screen_x,
                    behind_px=sensor_screen_x - left,
                    min_pixel_delta=float(settings["sensitivity"]),
                    sensor_ahead_px=float(settings["sensor_ahead_px"]),
                    actuator_delay_s=float(settings["actuator_delay_s"]),
                    landing_margin_s=float(settings["landing_margin_s"]),
                )
                detector = AutoJumpDetector(width, height, sensor_x, sensor_y, config)
                actuator = DelayedKeyActuator(self.keyboard, self.stop_event)

                print(
                    f"[Dino] DPI={DPI_MODE}; virtual-sensor={config.sensor_ahead_px:.0f}px ahead; "
                    f"servo-delay={config.actuator_delay_s * 1000:.0f}ms; hold={settings['key_hold_s'] * 1000:.0f}ms; "
                    f"capture=({left},{top},{width}x{height})",
                    flush=True,
                )

                calibration: list[np.ndarray] = []
                deadline = time.perf_counter() + 0.45
                while time.perf_counter() < deadline and not self.stop_event.is_set():
                    calibration.append(self._grab_gray(sct, region))
                    time.sleep(0.012)
                if self.stop_event.is_set():
                    return
                detector.calibrate(calibration)

                print(
                    f"[Dino] calibration: {len(calibration)} frames, threshold={detector.threshold:.1f}, noise={detector.noise_level:.2f}",
                    flush=True,
                )

                # Start/restart the game immediately; obstacle actions below use
                # the simulated servo delay.
                self.keyboard.press(keyboard.Key.space)
                time.sleep(0.050)
                self.keyboard.release(keyboard.Key.space)
                time.sleep(0.12)

                last_ui = 0.0
                while not self.stop_event.is_set():
                    now = time.perf_counter()
                    telemetry = detector.process(self._grab_gray(sct, region), now)

                    if telemetry.background_rebased:
                        print("[Dino] scene/background inversion detected; background rebased", flush=True)

                    if telemetry.should_jump:
                        contact_time = now + config.actuator_delay_s
                        actuator.schedule(contact_time, float(settings["key_hold_s"]))
                        self.jump_count += 1
                        exit_tti = telemetry.predicted_exit_tti_dino_s
                        exit_ms = exit_tti * 1000.0 if exit_tti is not None else float("nan")
                        land_ms = telemetry.landing_error_s * 1000.0 if telemetry.landing_error_s is not None else float("nan")
                        print(
                            f"[Dino] SERVO COMMAND #{self.jump_count}: reason={telemetry.trigger_reason}; "
                            f"contact-in={config.actuator_delay_s * 1000:.0f}ms; exit-at-dino={exit_ms:.0f}ms; "
                            f"land-after-exit={land_ms:.0f}ms; speed={telemetry.planning_speed_px_s:.0f}px/s; "
                            f"cluster={telemetry.cluster_width_px}",
                            flush=True,
                        )

                    if now - last_ui >= 0.10:
                        exit_tti = telemetry.predicted_exit_tti_dino_s
                        self.messages.put(
                            (
                                "live",
                                (
                                    telemetry.speed_px_s,
                                    telemetry.planning_speed_px_s,
                                    exit_tti,
                                    telemetry.envelope_duration_s,
                                    telemetry.rolling_min_envelope_s,
                                    telemetry.late_by_s,
                                    telemetry.required_extra_sensor_px,
                                    telemetry.scheduled_command_in_s,
                                    telemetry.trigger_reason,
                                    self.jump_count,
                                ),
                            )
                        )
                        last_ui = now

                    time.sleep(0.004)
        except Exception as exc:
            self.messages.put(("error", str(exc)))
        finally:
            self.stop_event.set()
            if actuator is not None:
                actuator.stop()
            self.messages.put(("finished", None))

    @staticmethod
    def _grab_gray(sct, region: dict[str, int]) -> np.ndarray:
        shot = np.asarray(sct.grab(region), dtype=np.uint8)
        b = shot[:, :, 0].astype(np.uint16)
        g = shot[:, :, 1].astype(np.uint16)
        r = shot[:, :, 2].astype(np.uint16)
        return ((29 * b + 150 * g + 77 * r) >> 8).astype(np.uint8)

    def _global_key(self, key) -> None:
        if key == STOP_KEY:
            self.stop_event.set()
            self.messages.put(("stop_requested", None))

    def _poll_messages(self) -> None:
        try:
            while True:
                kind, payload = self.messages.get_nowait()
                if kind == "live":
                    (
                        measured, planning, exit_tti, envelope_s, rolling_min_s,
                        late_by_s, extra_px, scheduled_in_s, reason, jumps,
                    ) = payload
                    if float(measured) > 0:
                        self.speed_var.set(f"{float(measured):.0f} / {float(planning):.0f} px/s")
                    else:
                        self.speed_var.set(f"learning / {float(planning):.0f} px/s")
                    self.exit_var.set(
                        f"{float(exit_tti) * 1000:.0f} ms" if exit_tti is not None else "—"
                    )
                    if envelope_s is not None:
                        rolling = f"; min {float(rolling_min_s) * 1000:.0f} ms" if rolling_min_s is not None else ""
                        self.envelope_var.set(f"{float(envelope_s) * 1000:.0f} ms{rolling}")
                    else:
                        self.envelope_var.set("learning…")
                    if float(extra_px) > 1.0:
                        self.timing_var.set(
                            f"TOO CLOSE by {float(late_by_s) * 1000:.0f} ms → move sensor +{float(extra_px):.0f}px"
                        )
                    elif scheduled_in_s is not None:
                        self.timing_var.set(f"command scheduled in {float(scheduled_in_s) * 1000:.0f} ms")
                    else:
                        self.timing_var.set("OK")
                    self.reason_var.set(str(reason) if reason != "none" else "—")
                    self.jump_var.set(str(jumps))
                elif kind == "error":
                    messagebox.showerror(APP_TITLE, f"Simulation stopped:\n\n{payload}")
                elif kind in {"finished", "stop_requested"} and self.running:
                    self.running = False
                    self._restore()
        except queue.Empty:
            pass

        try:
            self.root.after(50, self._poll_messages)
        except tk.TclError:
            pass

    def close(self) -> None:
        self.stop_event.set()
        try:
            self.listener.stop()
        except Exception:
            pass
        for marker in (self.dino_marker, self.sensor_marker):
            if marker:
                marker.destroy()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    DinoAutoPlayerApp().run()
