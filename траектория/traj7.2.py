import os
import csv
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import rasterio
from rasterio.warp import transform
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
        
        # Координаты углов в широте/долготе
        self.lon_min = None
        self.lon_max = None
        self.lat_min = None
        self.lat_max = None
        self.is_geographic = False

        self.speed_mps = 5.0
        self.dt = 0.1

        # Для управления цветовой шкалой
        self.im = None
        self.cbar = None
        self.cax = None

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

        self.info_label = ttk.Label(main_frame, text="📍 Точек: 0  |  📊 Измерений: 0  |  📏 Длина: 0 м",
                                    font=font_info, foreground='#4a9eff', background='#1a1a1a')
        self.info_label.pack(anchor='w', pady=(4, 0))

        self.status = ttk.Label(main_frame, text="Откройте GeoTIFF-файл", relief='sunken', anchor='w',
                                background='#2b2b2b', foreground='#cccccc', font=font_status)
        self.status.pack(fill=tk.X, pady=(6, 0), ipady=4)

        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.root.after(100, self.refresh)

    # ---------- Вспомогательные геометрические функции ----------
    @staticmethod
    def haversine(lon1, lat1, lon2, lat2):
        """Расстояние между двумя точками на сфере (в метрах) по формуле гаверсинуса."""
        R = 6371000
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)
        a = np.sin(delta_phi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(delta_lambda/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c

    @staticmethod
    def bearing(lon1, lat1, lon2, lat2):
        """Начальный азимут (в градусах, 0 = север) от точки 1 к точке 2."""
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        delta_lambda = np.radians(lon2 - lon1)
        x = np.sin(delta_lambda) * np.cos(phi2)
        y = np.cos(phi1)*np.sin(phi2) - np.sin(phi1)*np.cos(phi2)*np.cos(delta_lambda)
        theta = np.arctan2(x, y)
        bearing = np.degrees(theta)
        return (bearing + 360) % 360

    # ---------- Остальные методы ----------
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

    def pixel_to_lonlat(self, x, y):
        """Преобразование пиксельных координат в широту/долготу."""
        if self.ds is None:
            return None, None
        
        xs, ys = self.ds.xy([y], [x])  # (row, col)
        
        if self.is_geographic:
            return float(xs[0]), float(ys[0])
        
        lon, lat = transform(self.ds.crs, 'EPSG:4326', xs, ys)
        return float(lon[0]), float(lat[0])

    def get_corners_lonlat(self):
        """Получить широту/долготу для углов изображения."""
        if self.ds is None:
            return None, None, None, None
        
        h, w = self.img.shape
        corners = [(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)]
        lons = []
        lats = []
        
        for y, x in corners:
            lon, lat = self.pixel_to_lonlat(x, y)
            if lon is not None and lat is not None:
                lons.append(lon)
                lats.append(lat)
        
        if lons and lats:
            return min(lons), max(lons), min(lats), max(lats)
        return None, None, None, None

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
            
            if self.ds.crs and self.ds.crs.is_geographic:
                self.is_geographic = True
            else:
                self.is_geographic = False
            
            self.lon_min, self.lon_max, self.lat_min, self.lat_max = self.get_corners_lonlat()
            
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
        
        lon, lat = self.pixel_to_lonlat(x, y)
        coord_str = ""
        if lon is not None and lat is not None:
            coord_str = f"  🌍 {lat:.6f}°, {lon:.6f}°"
        
        self.status.config(text=f"✅ Добавлена точка {len(self.points)}: ({x}, {y}){coord_str}")
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
        """Построение интерполированной траектории с равномерным шагом по времени."""
        if len(self.points) < 2:
            return None

        pts = np.array(self.points, dtype=float)
        # Вычисляем расстояния в метрах между соседними точками
        dists_m = []
        for i in range(len(pts)-1):
            x1, y1 = pts[i]
            x2, y2 = pts[i+1]
            lon1, lat1 = self.pixel_to_lonlat(int(round(x1)), int(round(y1)))
            lon2, lat2 = self.pixel_to_lonlat(int(round(x2)), int(round(y2)))
            if lon1 is None or lat1 is None or lon2 is None or lat2 is None:
                # Если геокоординаты недоступны, используем пиксельное расстояние с приближением 30 м/пиксель
                # Но лучше вернуть None, чтобы не искажать результаты
                return None
            d = self.haversine(lon1, lat1, lon2, lat2)
            dists_m.append(d)
        total_len_m = sum(dists_m)
        if total_len_m <= 0:
            return None

        # Накопленная длина в метрах
        s_m = np.concatenate([[0.0], np.cumsum(dists_m)])

        # Время прохождения всей трассы
        duration = total_len_m / self.speed_mps
        if duration <= 0:
            return None

        # Массив времени с шагом dt
        n_steps = int(np.ceil(duration / self.dt)) + 1
        t_new = np.linspace(0, duration, n_steps)
        # Длины, соответствующие каждому моменту времени
        s_new = self.speed_mps * t_new
        s_new = np.minimum(s_new, total_len_m)  # ограничиваем

        # Интерполяция пиксельных координат как функция от длины в метрах
        fx = interp1d(s_m, pts[:, 0], kind='linear', fill_value='extrapolate')
        fy = interp1d(s_m, pts[:, 1], kind='linear', fill_value='extrapolate')
        x_new = fx(s_new)
        y_new = fy(s_new)

        # Высоты
        z_new = np.array([self.sample_elevation_px(int(round(x)), int(round(y))) for x, y in zip(x_new, y_new)])
        return t_new, x_new, y_new, z_new

    def get_total_length(self):
        """Вычисляет общую длину траектории в метрах."""
        if len(self.points) < 2:
            return 0.0
        total = 0.0
        for i in range(len(self.points)-1):
            x1, y1 = self.points[i]
            x2, y2 = self.points[i+1]
            lon1, lat1 = self.pixel_to_lonlat(x1, y1)
            lon2, lat2 = self.pixel_to_lonlat(x2, y2)
            if lon1 is not None and lat1 is not None and lon2 is not None and lat2 is not None:
                total += self.haversine(lon1, lat1, lon2, lat2)
        return total

    def redraw(self):
        self.ax.clear()
        self.ax.set_facecolor('#1a1a1a')
        self.ax.set_axis_off()

        # Отображаем карту и цветовую шкалу
        if self.img is not None:
            self.im = self.ax.imshow(self.img, cmap="terrain", origin="upper")
            
            if self.lon_min is not None and self.lon_max is not None and \
               self.lat_min is not None and self.lat_max is not None:
                
                self.ax.set_axis_on()
                
                x_ticks = [0, self.img.shape[1]//2, self.img.shape[1]-1]
                x_labels = [f"{self.lon_min:.4f}°", f"{(self.lon_min + self.lon_max)/2:.4f}°", f"{self.lon_max:.4f}°"]
                
                y_ticks = [0, self.img.shape[0]//2, self.img.shape[0]-1]
                y_labels = [f"{self.lat_max:.4f}°", f"{(self.lat_min + self.lat_max)/2:.4f}°", f"{self.lat_min:.4f}°"]
                
                self.ax.set_xticks(x_ticks)
                self.ax.set_xticklabels(x_labels, color='white', fontsize=10)
                self.ax.set_yticks(y_ticks)
                self.ax.set_yticklabels(y_labels, color='white', fontsize=10)
                
                self.ax.set_xlabel('Долгота', color='white', fontsize=11)
                self.ax.set_ylabel('Широта', color='white', fontsize=11)
                
                self.ax.tick_params(axis='x', colors='white', which='both')
                self.ax.tick_params(axis='y', colors='white', which='both')
                for spine in self.ax.spines.values():
                    spine.set_color('white')
                    spine.set_linewidth(0.5)
                
                self.ax.grid(True, color='gray', alpha=0.3, linestyle='--', linewidth=0.5)
            else:
                self.ax.set_axis_off()

            if self.cbar is None:
                divider = make_axes_locatable(self.ax)
                self.cax = divider.append_axes("right", size="5%", pad=0.05)
                self.cbar = self.fig.colorbar(self.im, cax=self.cax)
                self.cbar.set_label('Высота, м', color='white', fontsize=10)
                self.cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')
            else:
                self.cbar.update_normal(self.im)

        # ---- Отрисовка точек и подписей к ним ----
        if self.points:
            xs = [p[0] for p in self.points]
            ys = [p[1] for p in self.points]
            self.ax.plot(xs, ys, color='#ff6b6b', linestyle='--', linewidth=2, alpha=0.9)
            self.ax.scatter(xs, ys, c='#ffdd57', s=60, zorder=3, edgecolors='#ffffff', linewidth=0.8)

            for i, (x, y) in enumerate(self.points):
                lon, lat = self.pixel_to_lonlat(x, y)
                if lon is not None and lat is not None:
                    label = f"{i+1}\n{lat:.5f}°\n{lon:.5f}°"
                else:
                    label = f"{i+1}"
                self.ax.text(
                    x, y, label,
                    color='white', fontsize=9, weight='bold',
                    bbox=dict(facecolor='#000000', alpha=0.85, pad=4,
                              edgecolor='#ffdd57', linewidth=1.5),
                    horizontalalignment='center',
                    verticalalignment='bottom',
                    zorder=10
                )

        # ---- Подписи для отрезков: длина и азимут на середине ----
        if len(self.points) >= 2:
            for i in range(len(self.points)-1):
                x1, y1 = self.points[i]
                x2, y2 = self.points[i+1]
                lon1, lat1 = self.pixel_to_lonlat(x1, y1)
                lon2, lat2 = self.pixel_to_lonlat(x2, y2)
                if lon1 is not None and lat1 is not None and lon2 is not None and lat2 is not None:
                    dist = self.haversine(lon1, lat1, lon2, lat2)
                    bear = self.bearing(lon1, lat1, lon2, lat2)
                    
                    mx = (x1 + x2) / 2.0
                    my = (y1 + y2) / 2.0
                    
                    label = f"L={dist:.0f} м\nA={bear:.1f}°"
                    self.ax.text(
                        mx, my, label,
                        color='white', fontsize=9, weight='bold',
                        bbox=dict(facecolor='#000000', alpha=0.85, pad=4,
                                  edgecolor='#4a9eff', linewidth=1.5),
                        horizontalalignment='center',
                        verticalalignment='center',
                        zorder=10
                    )

        # ---- Траектория (интерполяция) ----
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

        # ---- Обновление информационной панели с длиной ----
        total_len = self.get_total_length()
        if total_len >= 1000:
            len_str = f"{total_len/1000:.1f} км"
        else:
            len_str = f"{total_len:.0f} м"
        self.info_label.config(text=f"📍 Точек: {len(self.points)}  |  📊 Измерений: {n_samples}  |  📏 Длина: {len_str}")

        status_parts = []
        if self.ds is not None:
            status_parts.append(f"📁 {os.path.basename(self.ds.name)}")
        status_parts.append(f"⚡ {self.speed_mps} м/с")
        status_parts.append(f"⏱️ {self.dt} с")
        if self.lon_min is not None and self.lat_min is not None:
            status_parts.append(f"🌍 {self.lat_min:.3f}°-{self.lat_max:.3f}°, {self.lon_min:.3f}°-{self.lon_max:.3f}°")
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
            w.writerow(["t_s", "x_px", "y_px", "elevation", "longitude", "latitude"])
            for t, x, y, z in zip(t_new, x_new, y_new, z_new):
                if z is None or (isinstance(z, float) and np.isnan(z)):
                    z = ""
                lon, lat = self.pixel_to_lonlat(int(round(x)), int(round(y)))
                lon_str = f"{lon:.6f}" if lon is not None else ""
                lat_str = f"{lat:.6f}" if lat is not None else ""
                w.writerow([round(float(t), 3), round(float(x), 2), round(float(y), 2), z, lon_str, lat_str])

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
