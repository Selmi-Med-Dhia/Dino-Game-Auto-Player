from __future__ import annotations

import ctypes
import queue
import sys
import threading
import time


def enable_dpi_awareness() -> str:
    """Keep Tk/pynput coordinates aligned with MSS physical screen pixels."""
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


DPI_MODE = enable_dpi_awareness()

import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
from mss import MSS
from pynput import keyboard
from pynput.keyboard import Controller as KeyboardController
from pynput.mouse import Button, Controller as MouseController

from detector import AutoJumpDetector, DetectorConfig

APP_TITLE = "Dino Auto-Player"
STOP_KEY = keyboard.Key.f8


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
    def __init__(self, root: tk.Tk, x: int, y: int, radius: int):
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
        canvas = tk.Canvas(self.window, width=size, height=size, bg="#ff00ff", highlightthickness=0)
        canvas.pack()
        pad = 7
        canvas.create_oval(pad, pad, size - pad, size - pad, outline="#00d8ff", width=3)
        canvas.create_line(size // 2 - 5, size // 2, size // 2 + 5, size // 2, fill="#00d8ff", width=2)
        canvas.create_line(size // 2, size // 2 - 5, size // 2, size // 2 + 5, fill="#00d8ff", width=2)
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


class PointSelector:
    def __init__(self, root: tk.Tk, on_select, on_cancel):
        self.root = root
        self.on_select = on_select
        self.on_cancel = on_cancel
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
        tk.Label(
            self.overlay,
            text="Click an EMPTY point roughly 100 px in front of the dinosaur, at cactus-body height\n(Esc to cancel)",
            fg="white",
            bg="black",
            font=("Segoe UI", 18, "bold"),
            padx=20,
            pady=12,
        ).place(relx=0.5, rely=0.08, anchor="n")
        self.overlay.focus_force()
        self.overlay.grab_set()

    def _clicked(self, event: tk.Event) -> None:
        logical_x = self.overlay.winfo_rootx() + int(event.x)
        logical_y = self.overlay.winfo_rooty() + int(event.y)
        x, y = physical_cursor_pos(logical_x, logical_y)
        self.overlay.grab_release()
        self.overlay.destroy()
        self.on_select(x, y)

    def cancel(self) -> None:
        try:
            self.overlay.grab_release()
        except tk.TclError:
            pass
        self.overlay.destroy()
        self.on_cancel()


class DinoAutoPlayerApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title(APP_TITLE)
        self.root.geometry("560x510")
        self.root.minsize(530, 490)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.selected: tuple[int, int] | None = None
        self.marker: Marker | None = None
        self.stop_event = threading.Event()
        self.running = False
        self.keyboard = KeyboardController()
        self.mouse = MouseController()
        self.listener = keyboard.Listener(on_press=self._global_key)
        self.listener.start()
        self.messages: queue.Queue[tuple[str, object]] = queue.Queue()
        self.jump_count = 0

        self.status_var = tk.StringVar(value="Choose a detector point to begin.")
        self.point_var = tk.StringVar(value="Not selected")
        self.speed_var = tk.StringVar(value="—")
        self.jump_var = tk.StringVar(value="0")
        self.occupancy_var = tk.StringVar(value="—")
        self.exit_var = tk.StringVar(value="—")

        self.lead_ms = tk.DoubleVar(value=70.0)
        self.radius = tk.IntVar(value=14)
        self.sensitivity = tk.DoubleVar(value=22.0)
        self.marker_ahead_px = tk.IntVar(value=100)

        self._build_ui()
        self.root.after(50, self._poll_messages)
        self.root.after(250, self.select_point)

    def _build_ui(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("vista")
        except tk.TclError:
            pass

        outer = ttk.Frame(self.root, padding=18)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Chrome Dino Auto-Player", font=("Segoe UI", 18, "bold")).pack(anchor="w")
        ttk.Label(
            outer,
            text="The detector now times jumps from the cactus EXIT (trailing edge), so the Dino lands just after the whole cactus/cluster clears.",
            wraplength=520,
        ).pack(anchor="w", pady=(6, 15))

        info = ttk.LabelFrame(outer, text="Last telemetry", padding=12)
        info.pack(fill="x")
        rows = (
            ("Detector point", self.point_var),
            ("Estimated speed", self.speed_var),
            ("Circle contrast", self.occupancy_var),
            ("Hazard exit", self.exit_var),
            ("Automatic jumps", self.jump_var),
        )
        for row, (name, variable) in enumerate(rows):
            ttk.Label(info, text=name).grid(row=row, column=0, sticky="w", pady=2)
            ttk.Label(info, textvariable=variable, font=("Segoe UI", 10, "bold")).grid(row=row, column=1, sticky="e", pady=2)
        info.columnconfigure(1, weight=1)

        settings = ttk.LabelFrame(outer, text="Tuning", padding=12)
        settings.pack(fill="x", pady=12)
        setting_rows = (
            ("Leading-edge safety (ms)", self.lead_ms, 25, 130),
            ("Marker ahead of Dino (px)", self.marker_ahead_px, 60, 150),
            ("Circle radius", self.radius, 8, 24),
            ("Min contrast", self.sensitivity, 12, 55),
        )
        for row, (name, variable, low, high) in enumerate(setting_rows):
            ttk.Label(settings, text=name).grid(row=row, column=0, sticky="w", pady=3)
            ttk.Scale(settings, from_=low, to=high, variable=variable, orient="horizontal").grid(row=row, column=1, sticky="ew", padx=10, pady=3)
            ttk.Label(settings, textvariable=variable, width=7).grid(row=row, column=2, sticky="e")
        settings.columnconfigure(1, weight=1)

        buttons = ttk.Frame(outer)
        buttons.pack(fill="x")
        self.select_btn = ttk.Button(buttons, text="Select point", command=self.select_point)
        self.select_btn.pack(side="left")
        self.start_btn = ttk.Button(buttons, text="Start autoplay", command=self.start)
        self.start_btn.pack(side="left", padx=8)
        self.stop_btn = ttk.Button(buttons, text="Stop (F8)", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left")

        ttk.Label(outer, textvariable=self.status_var, wraplength=520).pack(anchor="w", pady=(12, 0))
        ttk.Label(
            outer,
            text="Tip: Marker ahead should roughly match how many pixels the cyan point is in front of the Dino. Default 100 px works well for the recommended placement.",
            wraplength=520,
        ).pack(anchor="w", pady=(7, 0))

    def select_point(self) -> None:
        if self.running:
            self.stop()
            return
        self.root.withdraw()
        PointSelector(self.root, self._point_selected, self._selection_cancelled)

    def _selection_cancelled(self) -> None:
        self.root.deiconify()
        self.root.lift()
        self.status_var.set("Point selection cancelled.")

    def _point_selected(self, x: int, y: int) -> None:
        self.selected = (x, y)
        self.point_var.set(f"{x}, {y} physical px")
        if self.marker:
            self.marker.destroy()
        self.marker = Marker(self.root, x, y, int(self.radius.get()))
        self.root.deiconify()
        self.root.lift()
        self.status_var.set("Point selected. Leave the Dino game visible, then click Start autoplay.")

    def start(self) -> None:
        if self.running:
            return
        if not self.selected:
            self.select_point()
            return

        self.jump_count = 0
        self.jump_var.set("0")
        self.stop_event.clear()
        self.running = True
        self.start_btn.config(state="disabled")
        self.select_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_var.set("Running… press F8 to stop.")

        self.root.withdraw()
        if self.marker:
            self.marker.hide()

        settings = {
            "radius": int(self.radius.get()),
            "lead_ms": float(self.lead_ms.get()),
            "sensitivity": float(self.sensitivity.get()),
            "marker_ahead_px": float(self.marker_ahead_px.get()),
        }
        threading.Thread(target=self._run_detector, args=(settings,), daemon=True).start()

    def stop(self) -> None:
        if self.running:
            self.stop_event.set()

    def _restore(self) -> None:
        self.root.deiconify()
        self.root.lift()
        if self.marker:
            self.marker.show()
        self.start_btn.config(state="normal")
        self.select_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_var.set("Stopped. Adjust settings or restart.")

    def _run_detector(self, settings: dict[str, float]) -> None:
        assert self.selected is not None
        x, y = self.selected
        radius = int(settings["radius"])
        behind = max(32, radius + 12)
        lookahead = 600
        capture_h = max(48, radius * 2 + 18)

        try:
            time.sleep(0.18)
            self.mouse.position = (x, y)
            self.mouse.click(Button.left, 1)
            time.sleep(0.18)

            with MSS() as sct:
                virtual = sct.monitors[0]
                vleft, vtop = int(virtual["left"]), int(virtual["top"])
                vright = vleft + int(virtual["width"])
                vbottom = vtop + int(virtual["height"])

                left = max(vleft, x - behind)
                right = min(vright, x + lookahead)
                top = max(vtop, y - capture_h // 2)
                bottom = min(vbottom, y + capture_h // 2)
                width, height = right - left, bottom - top
                if width < 16 or height < 16:
                    raise RuntimeError("Detector point is too close to a desktop edge.")

                sensor_x, sensor_y = x - left, y - top
                region = {"left": left, "top": top, "width": width, "height": height}
                config = DetectorConfig(
                    radius=min(radius, max(4, min(width, height) // 2 - 1)),
                    lookahead_px=lookahead,
                    behind_px=behind,
                    min_pixel_delta=float(settings["sensitivity"]),
                    lead_time_s=float(settings["lead_ms"]) / 1000.0,
                    sensor_ahead_px=float(settings["marker_ahead_px"]),
                )
                detector = AutoJumpDetector(width, height, sensor_x, sensor_y, config)

                print(
                    f"[Dino] DPI={DPI_MODE}; physical point=({x},{y}); marker-ahead={config.sensor_ahead_px:.0f}px; capture=({left},{top},{width}x{height}); sensor=({sensor_x},{sensor_y})",
                    flush=True,
                )

                frames: list[np.ndarray] = []
                deadline = time.perf_counter() + 0.38
                while time.perf_counter() < deadline and not self.stop_event.is_set():
                    frames.append(self._grab_gray(sct, region))
                    time.sleep(0.012)
                if self.stop_event.is_set():
                    return

                detector.calibrate(frames)
                print(
                    f"[Dino] calibration: {len(frames)} frames, threshold={detector.threshold:.1f}, noise={detector.noise_level:.2f}",
                    flush=True,
                )

                self.keyboard.press(keyboard.Key.space)
                self.keyboard.release(keyboard.Key.space)
                time.sleep(0.12)

                last_ui = 0.0
                while not self.stop_event.is_set():
                    now = time.perf_counter()
                    telemetry = detector.process(self._grab_gray(sct, region), now)

                    if telemetry.should_jump:
                        self.keyboard.press(keyboard.Key.space)
                        self.keyboard.release(keyboard.Key.space)
                        self.jump_count += 1
                        landing_ms = telemetry.landing_error_s * 1000.0 if telemetry.landing_error_s is not None else float("nan")
                        print(
                            f"[Dino] JUMP #{self.jump_count}: entry={telemetry.obstacle_distance_px}, exit={telemetry.obstacle_exit_distance_px}, cluster={telemetry.cluster_width_px}, speed={telemetry.speed_px_s:.0f}px/s, land-after-exit={landing_ms:.0f}ms",
                            flush=True,
                        )

                    if now - last_ui >= 0.10:
                        self.messages.put(("live", (telemetry.speed_px_s, telemetry.occupancy, telemetry.obstacle_exit_distance_px, self.jump_count)))
                        last_ui = now

                    time.sleep(0.004)
        except Exception as exc:
            self.messages.put(("error", str(exc)))
        finally:
            self.stop_event.set()
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
                    speed, occupancy, exit_distance, jumps = payload
                    self.speed_var.set(f"{float(speed):,.0f} px/s" if float(speed) > 0 else "learning…")
                    self.occupancy_var.set(f"{float(occupancy) * 100:.1f}%")
                    self.exit_var.set(f"{float(exit_distance):.0f} px" if exit_distance is not None else "—")
                    self.jump_var.set(str(jumps))
                elif kind == "error":
                    messagebox.showerror(APP_TITLE, f"Autoplay stopped:\n\n{payload}")
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
        if self.marker:
            self.marker.destroy()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    DinoAutoPlayerApp().run()
