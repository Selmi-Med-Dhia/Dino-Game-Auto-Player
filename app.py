from __future__ import annotations

import ctypes
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import mss
import numpy as np
from pynput import keyboard, mouse
from pynput.keyboard import Controller as KeyboardController
from pynput.mouse import Button, Controller as MouseController

from detector import AutoJumpDetector, DetectorConfig


APP_TITLE = "Dino Auto-Player"
STOP_KEY = keyboard.Key.f8


class SensorOverlay:
    """Small click-through circle marking the selected detection point."""

    def __init__(self, root: tk.Tk, x: int, y: int, radius: int):
        self.root = root
        self.x = x
        self.y = y
        self.radius = radius
        self.size = radius * 2 + 18
        self.capture_excluded = False

        self.window = tk.Toplevel(root)
        self.window.overrideredirect(True)
        self.window.attributes("-topmost", True)
        self.window.configure(bg="#ff00ff")
        try:
            self.window.wm_attributes("-transparentcolor", "#ff00ff")
        except tk.TclError:
            pass
        left = x - self.size // 2
        top = y - self.size // 2
        self.window.geometry(f"{self.size}x{self.size}{left:+d}{top:+d}")

        canvas = tk.Canvas(self.window, width=self.size, height=self.size, bg="#ff00ff", highlightthickness=0)
        canvas.pack(fill="both", expand=True)
        pad = 7
        canvas.create_oval(pad, pad, self.size - pad, self.size - pad, outline="#00d8ff", width=3)
        canvas.create_line(self.size // 2, 2, self.size // 2, 10, fill="#00d8ff", width=2)
        canvas.create_line(self.size - 10, self.size // 2, self.size - 2, self.size // 2, fill="#00d8ff", width=2)
        canvas.create_line(self.size // 2, self.size - 10, self.size // 2, self.size - 2, fill="#00d8ff", width=2)
        canvas.create_line(2, self.size // 2, 10, self.size // 2, fill="#00d8ff", width=2)

        self.window.update_idletasks()
        self._make_click_through_and_capture_safe()

    def _make_click_through_and_capture_safe(self) -> None:
        if sys.platform != "win32":
            return
        try:
            hwnd = self.window.winfo_id()
            user32 = ctypes.windll.user32
            GWL_EXSTYLE = -20
            WS_EX_TRANSPARENT = 0x00000020
            WS_EX_LAYERED = 0x00080000
            WS_EX_TOOLWINDOW = 0x00000080
            WS_EX_NOACTIVATE = 0x08000000
            style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style | WS_EX_TRANSPARENT | WS_EX_LAYERED | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE)

            # Windows 10 2004+: keep our marker out of Desktop Duplication / screenshots.
            WDA_EXCLUDEFROMCAPTURE = 0x00000011
            self.capture_excluded = bool(user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE))
        except Exception:
            self.capture_excluded = False

    def show(self) -> None:
        self.window.deiconify()

    def hide(self) -> None:
        self.window.withdraw()

    def destroy(self) -> None:
        try:
            self.window.destroy()
        except tk.TclError:
            pass


class PointSelector:
    def __init__(self, root: tk.Tk, callback, cancel_callback=None):
        self.root = root
        self.callback = callback
        self.cancel_callback = cancel_callback
        self.overlay = tk.Toplevel(root)
        self.overlay.overrideredirect(True)
        self.overlay.attributes("-topmost", True)
        self.overlay.attributes("-alpha", 0.24)
        self.overlay.configure(bg="black")

        # Cover the whole virtual desktop on Windows, primary screen elsewhere.
        if sys.platform == "win32":
            user32 = ctypes.windll.user32
            SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN = 76, 77
            SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN = 78, 79
            vx = user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
            vy = user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
            vw = user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
            vh = user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)
        else:
            vx, vy = 0, 0
            vw, vh = root.winfo_screenwidth(), root.winfo_screenheight()

        self.vx, self.vy = vx, vy
        self.overlay.geometry(f"{vw}x{vh}{vx:+d}{vy:+d}")
        self.overlay.config(cursor="crosshair")
        self.overlay.bind("<Button-1>", self._clicked)
        self.overlay.bind("<Escape>", lambda _e: self.cancel())

        self.label = tk.Label(
            self.overlay,
            text="Click the point where you want the cactus detector circle\n(Esc to cancel)",
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
        x = self.overlay.winfo_rootx() + int(event.x)
        y = self.overlay.winfo_rooty() + int(event.y)
        self.overlay.grab_release()
        self.overlay.destroy()
        self.callback(x, y)

    def cancel(self) -> None:
        try:
            self.overlay.grab_release()
        except tk.TclError:
            pass
        self.overlay.destroy()
        if self.cancel_callback:
            self.cancel_callback()


class DinoAutoPlayerApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title(APP_TITLE)
        self.root.geometry("520x430")
        self.root.minsize(500, 410)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.selected: tuple[int, int] | None = None
        self.sensor_overlay: SensorOverlay | None = None
        self.worker: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.running = False
        self.keyboard = KeyboardController()
        self.mouse = MouseController()
        self.listener = keyboard.Listener(on_press=self._global_key)
        self.listener.start()
        self.messages: queue.Queue[tuple[str, object]] = queue.Queue()

        self.status_var = tk.StringVar(value="Choose a detector point to begin.")
        self.point_var = tk.StringVar(value="Not selected")
        self.speed_var = tk.StringVar(value="—")
        self.jump_var = tk.StringVar(value="0")
        self.occupancy_var = tk.StringVar(value="—")
        self.lead_ms = tk.DoubleVar(value=60.0)
        self.radius = tk.IntVar(value=14)
        self.sensitivity = tk.DoubleVar(value=26.0)
        self.jump_count = 0

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
            text="Select a blank point slightly in front of the dinosaur at cactus height. The app calibrates that local background and jumps when a contrasting obstacle approaches.",
            wraplength=480,
        ).pack(anchor="w", pady=(6, 16))

        info = ttk.LabelFrame(outer, text="Live status", padding=12)
        info.pack(fill="x")
        self._row(info, 0, "Detector point", self.point_var)
        self._row(info, 1, "Estimated speed", self.speed_var)
        self._row(info, 2, "Circle contrast", self.occupancy_var)
        self._row(info, 3, "Jumps", self.jump_var)

        settings = ttk.LabelFrame(outer, text="Tuning", padding=12)
        settings.pack(fill="x", pady=12)

        ttk.Label(settings, text="Predictive lead").grid(row=0, column=0, sticky="w")
        ttk.Scale(settings, from_=25, to=110, variable=self.lead_ms, orient="horizontal").grid(row=0, column=1, sticky="ew", padx=10)
        ttk.Label(settings, textvariable=self.lead_ms, width=6).grid(row=0, column=2, sticky="e")

        ttk.Label(settings, text="Circle radius").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Scale(settings, from_=8, to=24, variable=self.radius, orient="horizontal").grid(row=1, column=1, sticky="ew", padx=10, pady=(8, 0))
        ttk.Label(settings, textvariable=self.radius, width=6).grid(row=1, column=2, sticky="e", pady=(8, 0))

        ttk.Label(settings, text="Min contrast").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Scale(settings, from_=14, to=55, variable=self.sensitivity, orient="horizontal").grid(row=2, column=1, sticky="ew", padx=10, pady=(8, 0))
        ttk.Label(settings, textvariable=self.sensitivity, width=6).grid(row=2, column=2, sticky="e", pady=(8, 0))
        settings.columnconfigure(1, weight=1)

        buttons = ttk.Frame(outer)
        buttons.pack(fill="x", pady=(2, 8))
        self.select_btn = ttk.Button(buttons, text="Select point", command=self.select_point)
        self.select_btn.pack(side="left")
        self.start_btn = ttk.Button(buttons, text="Start autoplay", command=self.start)
        self.start_btn.pack(side="left", padx=8)
        self.stop_btn = ttk.Button(buttons, text="Stop (F8)", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left")

        ttk.Label(outer, textvariable=self.status_var, wraplength=480).pack(anchor="w", pady=(8, 0))
        ttk.Label(outer, text="Tip: if jumps are late at high speed, increase Predictive lead. If it jumps on noise, increase Min contrast.", wraplength=480).pack(anchor="w", pady=(8, 0))

    @staticmethod
    def _row(parent, row: int, label: str, variable: tk.StringVar) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Label(parent, textvariable=variable, font=("Segoe UI", 10, "bold")).grid(row=row, column=1, sticky="e", pady=2)
        parent.columnconfigure(1, weight=1)

    def select_point(self) -> None:
        if self.running:
            self.stop()
        self.root.withdraw()
        PointSelector(self.root, self._point_selected, self._selection_cancelled)

    def _selection_cancelled(self) -> None:
        self.root.deiconify()
        self.root.lift()
        self.status_var.set("Point selection cancelled.")

    def _point_selected(self, x: int, y: int) -> None:
        self.selected = (x, y)
        self.point_var.set(f"{x}, {y}")
        if self.sensor_overlay:
            self.sensor_overlay.destroy()
        self.sensor_overlay = SensorOverlay(self.root, x, y, int(self.radius.get()))
        self.root.deiconify()
        self.root.lift()
        self.status_var.set("Point selected. Keep the detector circle on empty background, then click Start autoplay.")

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
        self.status_var.set("Starting: focusing the game, calibrating clean background, then autoplaying…")

        # Hide the control window so it cannot overlap the capture area.
        self.root.withdraw()
        if self.sensor_overlay and not self.sensor_overlay.capture_excluded:
            # Fallback for Windows versions that cannot exclude a window from capture.
            self.sensor_overlay.hide()

        settings = {
            "radius": int(self.radius.get()),
            "lead_ms": float(self.lead_ms.get()),
            "sensitivity": float(self.sensitivity.get()),
        }
        self.worker = threading.Thread(target=self._run_detector, args=(settings,), daemon=True)
        self.worker.start()

    def stop(self) -> None:
        if not self.running:
            return
        self.stop_event.set()
        self.stop_btn.config(state="disabled")
        self.status_var.set("Stopping autoplay…")

    def _restore_after_stop(self) -> None:
        self.root.deiconify()
        self.root.lift()
        if self.sensor_overlay:
            self.sensor_overlay.show()
        self.start_btn.config(state="normal")
        self.select_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_var.set("Stopped. Adjust settings or restart. Press F8 at any time to stop autoplay.")

    def _run_detector(self, settings: dict[str, float]) -> None:
        assert self.selected is not None
        x, y = self.selected
        radius = int(settings["radius"])
        behind = max(24, radius + 8)
        lookahead = 150
        desired_height = max(34, radius * 2 + 8)

        try:
            # Focus Chrome/game safely by clicking the point the user chose.
            time.sleep(0.35)
            self.mouse.position = (x, y)
            self.mouse.click(Button.left, 1)
            time.sleep(0.18)

            with mss.mss() as sct:
                # Clamp the capture rectangle to the virtual desktop so a point
                # near a monitor edge does not make mss reject the grab.
                virtual = sct.monitors[0]
                vleft = int(virtual["left"])
                vtop = int(virtual["top"])
                vright = vleft + int(virtual["width"])
                vbottom = vtop + int(virtual["height"])

                left = max(vleft, x - behind)
                right = min(vright, x + lookahead)
                top = max(vtop, y - desired_height // 2)
                bottom = min(vbottom, y + desired_height // 2)
                width = right - left
                height = bottom - top
                if width < 8 or height < 8:
                    raise RuntimeError("The selected point is too close to the desktop edge. Choose a point farther inside the screen.")
                sensor_x = x - left
                sensor_y = y - top

                config = DetectorConfig(
                    radius=min(radius, max(4, min(width, height) // 2 - 1)),
                    lookahead_px=lookahead,
                    behind_px=behind,
                    min_pixel_delta=float(settings["sensitivity"]),
                    lead_time_s=float(settings["lead_ms"]) / 1000.0,
                )
                detector = AutoJumpDetector(width, height, sensor_x, sensor_y, config)
                region = {"left": left, "top": top, "width": width, "height": height}

                calibration = []
                deadline = time.perf_counter() + 0.38
                while time.perf_counter() < deadline and not self.stop_event.is_set():
                    calibration.append(self._grab_gray(sct, region))
                    time.sleep(0.012)
                if self.stop_event.is_set():
                    return
                detector.calibrate(calibration)

                # Space starts/restarts the dinosaur game. If it is already
                # running, this is simply an initial jump and autoplay continues.
                self.keyboard.press(keyboard.Key.space)
                self.keyboard.release(keyboard.Key.space)
                time.sleep(0.12)

                last_ui = 0.0
                while not self.stop_event.is_set():
                    now = time.perf_counter()
                    frame = self._grab_gray(sct, region)
                    telemetry = detector.process(frame, now)

                    if telemetry.should_jump:
                        self.keyboard.press(keyboard.Key.space)
                        self.keyboard.release(keyboard.Key.space)
                        self.jump_count += 1

                    if now - last_ui >= 0.10:
                        speed = telemetry.speed_px_s
                        occupancy_pct = telemetry.occupancy * 100.0
                        self.messages.put(("live", (speed, occupancy_pct, self.jump_count)))
                        last_ui = now

                    # Cap around 120 Hz. mss is fast, and this keeps CPU usage sane.
                    time.sleep(0.004)
        except Exception as exc:
            self.messages.put(("error", str(exc)))
        finally:
            self.stop_event.set()
            self.messages.put(("finished", None))

    @staticmethod
    def _grab_gray(sct: mss.mss, region: dict[str, int]) -> np.ndarray:
        shot = np.asarray(sct.grab(region), dtype=np.uint8)
        # mss is BGRA; integer luma approximation avoids OpenCV dependency.
        b = shot[:, :, 0].astype(np.uint16)
        g = shot[:, :, 1].astype(np.uint16)
        r = shot[:, :, 2].astype(np.uint16)
        gray = ((29 * b + 150 * g + 77 * r) >> 8).astype(np.uint8)
        return gray

    def _set_live_ui(self, speed: float, occupancy_pct: float, jumps: int) -> None:
        self.speed_var.set(f"{speed:,.0f} px/s" if speed > 0 else "learning…")
        self.occupancy_var.set(f"{occupancy_pct:.1f}%")
        self.jump_var.set(str(jumps))

    def _worker_error(self, message: str) -> None:
        messagebox.showerror(APP_TITLE, f"Autoplay stopped because of an error:\n\n{message}")

    def _global_key(self, key) -> None:
        if key == STOP_KEY:
            self.stop_event.set()
            self.messages.put(("stop_requested", None))

    def _poll_messages(self) -> None:
        try:
            while True:
                kind, payload = self.messages.get_nowait()
                if kind == "live":
                    speed, occupancy_pct, jumps = payload
                    self._set_live_ui(float(speed), float(occupancy_pct), int(jumps))
                elif kind == "error":
                    self._worker_error(str(payload))
                elif kind in {"finished", "stop_requested"}:
                    if self.running:
                        self.running = False
                        self._restore_after_stop()
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
        if self.sensor_overlay:
            self.sensor_overlay.destroy()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    DinoAutoPlayerApp().run()
