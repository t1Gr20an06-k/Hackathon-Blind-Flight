import os
import csv
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import interp1d

plt.style.use('dark_background')


class GeoTIFFPathApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Геотрасса с высотами")
        self.root.geometry("880x760")
        self.root.minsize(750, 650)

        # Настройка тёмной темы ttk
        style = ttk.Style()
        style.theme_use('clam')

        bg_dark = '#1a1a1a'
        bg_medium = '#2b2b2b'
        bg_light = '#3a3a3a'
        fg_light = '#e0e0e0'
        fg_bright = '#ffffff'
        accent = '#4a9eff'

        font_button = ('Segoe UI', 11, 'bold')
        font_label = ('Segoe UI', 11)
        font_entry = ('Segoe UI', 11)
        font_status = ('Segoe UI', 10)
        font_info = ('Segoe UI', 14, 'bold')

        style.configure('.', background=bg_dark, foreground=fg_light, fieldbackground=bg_medium)
        style.configure('TLabel', background=bg_dark, foreground=fg_light, font=font_label)
        style.configure('TButton', background=bg_medium, foreground=fg_light,
                        borderwidth=2, focusthickness=3, focuscolor=accent,
                        font=font_button, padding=6)
        style.map('TButton',
                  background=[('active', bg_light), ('pressed', accent)],
                  foreground=[('active', fg_bright)])
        style.configure('TEntry', fieldbackground=bg_medium, foreground=fg_light,
                        insertcolor=fg_light, borderwidth=2, font=font_entry)
        style.configure('TLabelframe', background=bg_dark, foreground=fg_light,
                        bordercolor=bg_light, borderwidth=2)
        style.configure('TLabelframe.Label', background=bg_dark, foreground=fg_bright,
                        font=('Segoe UI', 12, 'bold'))
        style.configure('TFrame', background=bg_dark)

        self.root.configure(bg=bg_dark)

        # Данные
        self.ds = None
        self.img = None
        self.points = []
        self.running = True
        self.last_update = 0.0

        self.speed_mps = 5.0
        self.dt = 0.1

        # Для управления цветовой шкалой
        self.im = None
        self.cbar = None
        self.cax = None  # храним ось шкалы, чтобы не пересоздавать

        # Основной контейнер
        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Панель управления
        control_panel = ttk.Frame(main_frame)
        control_panel.pack(fill=tk.X, pady=(0, 10))

        param_frame = ttk.LabelFrame(control_panel, text="⚙️ Параметры", padding=8)
        param_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        ttk.Label(param_frame, text="Скорость (м/с):").grid(row=0, column=0, padx=6, pady=4, sticky='e')
        self.speed_entry = ttk.Entry(param_frame, width=8)
        self.speed_entry.insert(0, "5.0")
        self.speed_entry.grid(row=0, column=1, padx=6, pady=4, sticky='w')

        ttk.Label(param_frame, text="Шаг (с):").grid(row=0, column=2, padx=6, pady=4, sticky='e')
        self.dt_entry = ttk.Entry(param_frame, width=8)
        self.dt_entry.insert(0, "0.1")
        self.dt_entry.grid(row=0, column=3, padx=6, pady=4, sticky='w')

        self.apply_btn = ttk.Button(param_frame, text="✅", command=self.apply_params, width=4)
        self.apply_btn.grid(row=0, column=4, padx=8, pady=4)

        action_frame = ttk.Frame(control_panel)
        action_frame.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Button(action_frame, text="📂 Открыть", command=self.open_file).pack(side=tk.LEFT, padx=3)
        ttk.Button(action_frame, text="🗑️ Очистить", command=self.clear_points).pack(side=tk.LEFT, padx=3)
        ttk.Button(action_frame, text="💾 CSV", command=self.save_csv).pack(side=tk.LEFT, padx=3)

        # Холст
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.fig.patch.set_facecolor('#1a1a1a')
        self.ax.set_facecolor('#1a1a1a')
        self.ax.set_axis_off()

        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Подсказка
        hint = ttk.Label(main_frame, text="🖱️ Кликните на карте, чтобы добавить точки",
                         foreground='#aaaaaa', font=('Segoe UI', 11))
        hint.pack(anchor='w', pady=(5, 0))

        self.info_label = ttk.Label(main_frame, text="📍 Точек: 0  |  📊 Измерений: 0",
                                    font=font_info, foreground='#4a9eff', background='#1a1a1a')
        self.info_label.pack(anchor='w', pady=(4, 0))

        self.status = ttk.Label(main_frame, text="Откройте GeoTIFF-файл", relief='sunken', anchor='w',
                                background='#2b2b2b', foreground='#cccccc', font=font_status)
        self.status.pack(fill=tk.X, pady=(6, 0), ipady=4)

        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.root.after(100, self.refresh)

    def apply_params(self):
        try:
            new_speed = float(self.speed_entry.get())
            new_dt = float(self.dt_entry.get())
            if new_speed <= 0 or new_dt <= 0:
                raise ValueError
            self.speed_mps = new_speed
            self.dt = new_dt
            self.status.config(text=f"✅ Параметры обновлены: {self.speed_mps} м/с, {self.dt} с")
            self.redraw()
        except ValueError:
            messagebox.showerror("Ошибка", "Введите положительные числа для скорости и шага.")

    def open_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("GeoTIFF", "*.tif *.tiff"), ("Все файлы", "*.*")]
        )
        if not path:
            return

        if self.ds is not None:
            self.ds.close()

        try:
            self.ds = rasterio.open(path)
            self.img = self.ds.read(1).astype(float)
            nodata = self.ds.nodata
            if nodata is not None:
                self.img[self.img == nodata] = np.nan
            self.points = []
            # Удаляем старую шкалу, если есть
            if self.cbar is not None:
                self.cbar.remove()
                self.cbar = None
                self.cax = None
            self.im = None
            self.redraw()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось открыть файл:\n{e}")

    def clear_points(self):
        self.points = []
        self.redraw()
        self.status.config(text="🗑️ Точки очищены")

    def on_click(self, event):
        if self.ds is None:
            self.status.config(text="❌ Сначала загрузите GeoTIFF")
            return
        if event.inaxes != self.ax:
            self.status.config(text="⚠️ Кликните по карте, а не по шкале")
            return
        if event.xdata is None or event.ydata is None:
            return

        x = int(round(event.xdata))
        y = int(round(event.ydata))

        if y < 0 or x < 0 or y >= self.img.shape[0] or x >= self.img.shape[1]:
            self.status.config(text=f"❌ Точка ({x}, {y}) за пределами карты")
            return

        self.points.append((x, y))
        self.status.config(text=f"✅ Добавлена точка {len(self.points)}: ({x}, {y})")
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
        self.ax.set_facecolor('#1a1a1a')
        self.ax.set_axis_off()

        # Отображаем карту и создаём цветовую шкалу (только если её ещё нет)
        if self.img is not None:
            self.im = self.ax.imshow(self.img, cmap="terrain", origin="upper")
            # Создаём шкалу, если ещё не создана
            if self.cbar is None:
                divider = make_axes_locatable(self.ax)
                self.cax = divider.append_axes("right", size="5%", pad=0.05)
                self.cbar = self.fig.colorbar(self.im, cax=self.cax)
                self.cbar.set_label('Высота, м', color='white', fontsize=10)
                self.cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')
            else:
                # Обновляем данные шкалы (если карта изменилась)
                self.cbar.update_normal(self.im)

        # Рисуем точки и траекторию поверх
        if self.points:
            xs = [p[0] for p in self.points]
            ys = [p[1] for p in self.points]
            self.ax.plot(xs, ys, color='#ff6b6b', linestyle='--', linewidth=2, alpha=0.9)
            self.ax.scatter(xs, ys, c='#ffdd57', s=60, zorder=3, edgecolors='#ffffff', linewidth=0.8)

            for i, (x, y) in enumerate(self.points):
                z = self.sample_elevation_px(x, y)
                if z is not None:
                    self.ax.text(
                        x, y, f"{i+1}:{z:.1f}",
                        color='white', fontsize=9,
                        bbox=dict(facecolor='#222222', alpha=0.8, pad=3, edgecolor='none')
                    )

        n_samples = 0
        path = self.build_path()
        if path is not None:
            t_new, x_new, y_new, z_new = path
            n_samples = len(t_new)
            valid = np.isfinite(z_new)
            self.ax.plot(x_new, y_new, color='#4a9eff', linewidth=3, zorder=2)
            if np.any(valid):
                self.ax.scatter(x_new[valid], y_new[valid], c=z_new[valid],
                                cmap='viridis', s=25, zorder=4, edgecolors='#ffffff', linewidth=0.4)

        self.ax.set_title("Построение траектории по кликам", color='#ffffff', fontsize=14)
        self.canvas.draw_idle()

        # Обновляем информацию
        self.info_label.config(text=f"📍 Точек: {len(self.points)}  |  📊 Измерений: {n_samples}")

        status_parts = []
        if self.ds is not None:
            status_parts.append(f"📁 {os.path.basename(self.ds.name)}")
        status_parts.append(f"⚡ {self.speed_mps} м/с")
        status_parts.append(f"⏱️ {self.dt} с")
        self.status.config(text=" | ".join(status_parts))

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
            messagebox.showinfo("Информация", "Нужно как минимум 2 точки для сохранения.")
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

        self.status.config(text=f"💾 CSV сохранён: {path}")

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
