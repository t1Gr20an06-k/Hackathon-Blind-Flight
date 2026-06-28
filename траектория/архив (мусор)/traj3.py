import os
import csv
import time
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.interpolate import interp1d


class GeoTIFFPathApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GeoTIFF trajectory with speed")

        self.ds = None
        self.img = None
        self.points = []
        self.running = True
        self.last_update = 0.0

        self.speed_mps = 5.0
        self.dt = 0.1

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        control = tk.Frame(root)
        control.pack(fill=tk.X)

        tk.Label(control, text="Speed m/s").pack(side=tk.LEFT)
        self.speed_entry = tk.Entry(control, width=8)
        self.speed_entry.insert(0, "5.0")
        self.speed_entry.pack(side=tk.LEFT)

        tk.Label(control, text="dt s").pack(side=tk.LEFT)
        self.dt_entry = tk.Entry(control, width=8)
        self.dt_entry.insert(0, "0.1")
        self.dt_entry.pack(side=tk.LEFT)

        tk.Button(control, text="Apply", command=self.apply_params).pack(side=tk.LEFT)
        tk.Button(control, text="Open GeoTIFF", command=self.open_file).pack(side=tk.LEFT)
        tk.Button(control, text="Clear points", command=self.clear_points).pack(side=tk.LEFT)
        tk.Button(control, text="Save CSV", command=self.save_csv).pack(side=tk.LEFT)

        self.status = tk.Label(root, text="Open a GeoTIFF file", anchor="w")
        self.status.pack(fill=tk.X)

        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(100, self.refresh)

    def apply_params(self):
        try:
            self.speed_mps = float(self.speed_entry.get())
            self.dt = float(self.dt_entry.get())
            self.status.config(text=f"speed={self.speed_mps} m/s, dt={self.dt} s")
        except ValueError:
            messagebox.showerror("Error", "Wrong speed or dt")

    def open_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("GeoTIFF", "*.tif *.tiff"), ("All files", "*.*")]
        )
        if not path:
            return

        if self.ds is not None:
            self.ds.close()

        self.ds = rasterio.open(path)
        self.img = self.ds.read(1).astype(float)

        nodata = self.ds.nodata
        if nodata is not None:
            self.img[self.img == nodata] = np.nan

        self.points = []
        self.status.config(text=f"Loaded: {os.path.basename(path)} | left click to add points")
        self.redraw()

    def clear_points(self):
        self.points = []
        self.redraw()

    def on_click(self, event):
        if self.ds is None:
            return
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        x = int(round(event.xdata))
        y = int(round(event.ydata))

        if y < 0 or x < 0 or y >= self.img.shape[0] or x >= self.img.shape[1]:
            return

        self.points.append((x, y))
        self.redraw()

    def sample_elevation_px(self, x, y):
        if self.img is None:
            return None
        if y < 0 or x < 0 or y >= self.img.shape[0] or x >= self.img.shape[1]:
            return None
        z = self.img[y, x]
        if np.isnan(z):
            return None
        return float(z)

    def build_path(self):
        if len(self.points) < 2:
            return None

        pts = np.array(self.points, dtype=float)
        d = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
        s = np.concatenate([[0.0], np.cumsum(d)])

        if s[-1] <= 0:
            return None

        total_len_px = s[-1]
        pixel_per_second = self.speed_mps
        if pixel_per_second <= 0:
            return None

        duration = total_len_px / pixel_per_second
        t_new = np.arange(0.0, duration + self.dt, self.dt)

        s_new = np.minimum(t_new * pixel_per_second, total_len_px)

        fx = interp1d(s, pts[:, 0], kind="linear")
        fy = interp1d(s, pts[:, 1], kind="linear")

        x_new = fx(s_new)
        y_new = fy(s_new)
        z_new = np.array([self.sample_elevation_px(int(round(x)), int(round(y))) for x, y in zip(x_new, y_new)])
        return t_new, x_new, y_new, z_new

    def redraw(self):
        self.ax.clear()

        if self.img is not None:
            self.ax.imshow(self.img, cmap="terrain", origin="upper")

        if self.points:
            xs = [p[0] for p in self.points]
            ys = [p[1] for p in self.points]
            self.ax.plot(xs, ys, "r--", linewidth=1.5, alpha=0.8)
            self.ax.scatter(xs, ys, c="yellow", s=35, zorder=3)

            for i, (x, y) in enumerate(self.points):
                z = self.sample_elevation_px(x, y)
                if z is not None:
                    self.ax.text(
                        x, y, f"{i+1}:{z:.1f}",
                        color="white", fontsize=8,
                        bbox=dict(facecolor="black", alpha=0.5, pad=1)
                    )

        path = self.build_path()
        if path is not None:
            t_new, x_new, y_new, z_new = path
            valid = np.isfinite(z_new)
            self.ax.plot(x_new, y_new, color="cyan", linewidth=2.5, zorder=2)
            if np.any(valid):
                self.ax.scatter(x_new[valid], y_new[valid], c=z_new[valid], cmap="viridis", s=12, zorder=4)
                self.status.config(
                    text=f"Points: {len(self.points)} | samples: {len(t_new)} | speed: {self.speed_mps} | dt: {self.dt}"
                )

        self.ax.set_title("Left click to build a trajectory")
        self.ax.set_axis_off()
        self.canvas.draw_idle()

    def refresh(self):
        if self.running and self.ds is not None and len(self.points) >= 2:
            now = time.time()
            if now - self.last_update >= self.dt:
                self.last_update = now
                self.redraw()
        if self.running:
            self.root.after(100, self.refresh)

    def save_csv(self):
        path_data = self.build_path()
        if path_data is None:
            messagebox.showinfo("Info", "Need at least 2 points")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")]
        )
        if not path:
            return

        t_new, x_new, y_new, z_new = path_data
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "x_px", "y_px", "elevation"])
            for t, x, y, z in zip(t_new, x_new, y_new, z_new):
                if z is None or (isinstance(z, float) and np.isnan(z)):
                    z = ""
                w.writerow([round(float(t), 3), round(float(x), 2), round(float(y), 2), z])

        self.status.config(text=f"Saved: {path}")

    def on_close(self):
        self.running = False
        try:
            if self.ds is not None:
                self.ds.close()
        except Exception:
            pass
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = GeoTIFFPathApp(root)
    root.mainloop()
