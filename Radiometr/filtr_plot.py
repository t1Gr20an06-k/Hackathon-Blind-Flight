"""
Единый скрипт обработки данных радиовысотомера с барометрической коррекцией.

Объединяет:
1. Парсинг NMEA-сообщений радиоальтиметра (filtr_radio.py)
2. Генерацию зашумленных барометрических данных (random_BMP_1500.py)
3. Фильтрацию радиоальтиметра (медианный + ФНЧ с нулевой фазой)
4. Фильтрацию барометра (медианный + Калмановский фильтр)
5. Вычисление абсолютной высоты рельефа
6. Визуализацию качества фильтрации и устойчивости к шуму

Выходные данные:
- Отфильтрованная высота радиоальтиметра (H_radar)
- Отфильтрованная барометрическая высота (H_baro)
- Абсолютная высота рельефа (H_terrain = H_baro - H_radar)
"""

import sys
import os
import re
from pathlib import Path
from collections import deque
from typing import Optional, List, Tuple, Dict, Union

import numpy as np
from scipy.signal import butter, filtfilt

# Импортируем matplotlib только если нужна визуализация
try:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# ============================================================================
# БЛОК 1: ПАРСИНГ NMEA (из filtr_radio.py)
# ============================================================================

def parse_gpgga_altitude(sentence: str) -> Optional[float]:
    """
    Извлекает высоту антенны (Altitude, поле 9) из $GPGGA.
    Контрольная сумма НЕ проверяется — заглушка.
    """
    sentence = sentence.strip()

    if not sentence.startswith("$GPGGA"):
        return None

    # Отрезаем чексумму (всё после '*')
    body = sentence.split("*")[0]
    fields = body.split(",")

    if len(fields) < 11:
        return None

    try:
        alt_str = fields[9]
        if alt_str == "":
            return None
        return float(alt_str)
    except (ValueError, IndexError):
        return None


# ============================================================================
# БЛОК 2: ФИЛЬТР РАДИОАЛЬТИМЕТРА (из filtr_radio.py)
# ============================================================================

class MedianFilter:
    """Медианный фильтр с фиксированным окном (нечётным)."""

    def __init__(self, window_size: int = 5):
        if window_size % 2 == 0:
            raise ValueError("Окно медианного фильтра должно быть нечётным")
        self._buffer: deque = deque(maxlen=window_size)

    def update(self, value: float) -> Optional[float]:
        self._buffer.append(value)
        if len(self._buffer) < self._buffer.maxlen:
            return None
        return float(np.median(self._buffer))

    @property
    def ready(self) -> bool:
        return len(self._buffer) == self._buffer.maxlen

    def reset(self):
        self._buffer.clear()


class ZeroPhaseButterworth:
    """
    Низкочастотный Баттерворт с filtfilt на скользящем окне.
    """

    def __init__(
        self,
        fs: float = 10.0,
        cutoff: float = 2.0,
        window_duration: float = 4.0,
        order: int = 2,
    ):
        if cutoff <= 0:
            raise ValueError("Частота среза должна быть > 0")
        if cutoff >= fs / 2:
            raise ValueError(
                f"Частота среза ({cutoff} Гц) должна быть < Найквиста ({fs/2} Гц)"
            )
        self.fs = fs
        self.cutoff = cutoff
        self.order = order
        self.window_duration = window_duration

        nyq = 0.5 * fs
        self._b, self._a = butter(order, cutoff / nyq, btype='low')

        self._min_samples = max(4 * order, 10)
        self._buffer: deque = deque()
        self._max_samples = max(
            int(window_duration * fs),
            self._min_samples + 1
        )
        self._last_value: Optional[float] = None

    def update(self, value: float) -> Optional[float]:
        self._buffer.append(value)

        while len(self._buffer) > self._max_samples:
            self._buffer.popleft()

        if len(self._buffer) < self._min_samples:
            self._last_value = value
            return None

        arr = np.array(self._buffer, dtype=np.float64)

        if np.any(np.isnan(arr)):
            clean = arr[~np.isnan(arr)]
            if len(clean) < self._min_samples:
                self._last_value = float(value)
                return self._last_value
            arr = clean

        try:
            filtered = filtfilt(self._b, self._a, arr)
            self._last_value = float(filtered[-1])
        except ValueError:
            if self._last_value is None:
                self._last_value = float(value)

        return self._last_value

    @property
    def ready(self) -> bool:
        return self._last_value is not None

    def reset(self):
        self._buffer.clear()
        self._last_value = None


class AltimeterPreFilter:
    """
    Предварительный фильтр данных радиовысотомера.
    """

    def __init__(
        self,
        fs: float = 10.0,
        cutoff: float = 2.0,
        median_window: int = 5,
        window_duration: float = 4.0,
    ):
        self.median = MedianFilter(window_size=median_window)
        self.butter = ZeroPhaseButterworth(
            fs=fs, cutoff=cutoff, window_duration=window_duration
        )
        self.raw_count = 0
        self.error_count = 0
        self.filtered_count = 0
        self.last_output: Optional[float] = None

    def process(self, nmea_sentence: str) -> Optional[float]:
        self.raw_count += 1

        altitude = parse_gpgga_altitude(nmea_sentence)
        if altitude is None:
            self.error_count += 1
            return None

        med = self.median.update(altitude)
        if med is None:
            return None

        out = self.butter.update(med)
        if out is None:
            return None

        self.filtered_count += 1
        self.last_output = out
        return out

    @property
    def ready(self) -> bool:
        return self.median.ready and self.butter.ready

    def reset(self):
        self.median.reset()
        self.butter.reset()
        self.raw_count = 0
        self.error_count = 0
        self.filtered_count = 0
        self.last_output = None


# ============================================================================
# БЛОК 3: ФИЛЬТР БАРОМЕТРА (из filtr1_1.py)
# ============================================================================

class BaroAltitudeFilter:
    """
    Каскадный фильтр высоты: Медиана (3) + Калман (1D).
    """

    def __init__(self, start_altitude: float, q: float = 0.05, r: float = 1.2):
        self._buf = [start_altitude] * 3
        self._buf_idx = 0
        self._x_hat = start_altitude
        self._p = 1.0
        self._q = q
        self._r = r

    def _median3(self, a: float, b: float, c: float) -> float:
        if a > b:
            a, b = b, a
        if b > c:
            b, c = c, b
        if a > b:
            a, b = b, a
        return b

    def update(self, raw_altitude: float) -> float:
        self._buf[self._buf_idx] = raw_altitude
        self._buf_idx = (self._buf_idx + 1) % 3

        clean_measurement = self._median3(self._buf[0], self._buf[1], self._buf[2])

        self._p = self._p + self._q
        k_gain = self._p / (self._p + self._r)
        self._x_hat = self._x_hat + k_gain * (clean_measurement - self._x_hat)
        self._p = (1.0 - k_gain) * self._p

        return self._x_hat

    def reset(self, altitude: float):
        self._buf = [altitude] * 3
        self._x_hat = altitude
        self._p = 1.0


# ============================================================================
# БЛОК 4: ГЕНЕРАЦИЯ БАРОМЕТРИЧЕСКИХ ДАННЫХ (из random_BMP_1500.py)
# ============================================================================

def generate_baro_data_for_radar(
    n_samples: int,
    base_altitude: float = 1500.0,
    freq: float = 10.0,
    noise_std: float = 1.5,
    spike_prob: float = 0.01,
    spike_amplitude: float = 8.0,
    wind_amplitude: float = 5.0,
    drift_rate: float = 0.02,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Генерирует барометрические данные, синхронизированные с данными радиоальтиметра.
    Возвращает массив значений барометрической высоты.
    """
    if seed is not None:
        np.random.seed(seed)

    time = np.arange(n_samples) / freq

    # Базовая высота 1500 м
    true_alt = np.full(n_samples, base_altitude)

    # Ветровые колебания
    wind = (
        wind_amplitude * 0.5 * np.sin(2 * np.pi * 0.05 * time) +
        wind_amplitude * 0.3 * np.sin(2 * np.pi * 0.15 * time + 1.2) +
        wind_amplitude * 0.2 * np.cos(2 * np.pi * 0.3 * time + 0.7)
    )

    # Коррекция PID (авторегрессия)
    correction = np.zeros(n_samples)
    for i in range(1, n_samples):
        correction[i] = correction[i-1] * 0.95 + np.random.normal(0, 0.3)
        correction[i] = np.clip(correction[i], -3.0, 3.0)

    true_alt = base_altitude + wind + correction

    # Шум
    noise = np.random.normal(0, noise_std, n_samples)

    # Импульсные выбросы
    spikes_mask = np.random.rand(n_samples) < spike_prob
    spike_values = np.random.exponential(spike_amplitude / 2, n_samples)
    spike_signs = np.random.choice([-1, 1], n_samples)
    spikes = np.where(spikes_mask, spike_values * spike_signs, 0)

    # Температурный дрейф
    drift = drift_rate * np.sin(2 * np.pi * 0.001 * time) * np.random.normal(1, 0.3)

    # Высокочастотная вибрация
    vibration = (
        0.5 * np.sin(2 * np.pi * 2.3 * time) * np.random.normal(1, 0.3, n_samples) +
        0.3 * np.sin(2 * np.pi * 5.7 * time) * np.random.normal(1, 0.2, n_samples)
    )

    # Скачки давления
    pressure_jumps = np.zeros(n_samples)
    jump_positions = np.random.choice(n_samples, size=int(n_samples * 0.005), replace=False)
    pressure_jumps[jump_positions] = np.random.normal(0, 2.0, len(jump_positions))

    # Сырой сигнал
    raw_alt = true_alt + noise + spikes + drift + vibration + pressure_jumps

    return raw_alt


# ============================================================================
# БЛОК 5: ВИЗУАЛИЗАЦИЯ И АНАЛИЗ УСТОЙЧИВОСТИ
# ============================================================================

class FilterVisualizer:
    """
    Визуализация качества фильтрации и анализ устойчивости алгоритма.
    """

    @staticmethod
    def plot_filter_comparison(
        raw_data: np.ndarray,
        filtered_data: np.ndarray,
        baro_raw: np.ndarray,
        baro_filtered: np.ndarray,
        terrain: np.ndarray,
        fs: float,
        title: str = "Результаты фильтрации"
    ):
        """
        Сравнительный график сырых и отфильтрованных данных.
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib не установлен. Пропуск визуализации.")
            return

        time = np.arange(len(raw_data)) / fs

        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.25)

        # 1. Радиоальтиметр: сырые и фильтрованные
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(time, raw_data, 'gray', alpha=0.4, linewidth=0.8, label='Сырые данные')
        ax1.plot(time, filtered_data, 'b-', linewidth=1.5, label='Фильтрованные')
        ax1.set_ylabel('Высота [м]')
        ax1.set_title('Радиоальтиметр')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # 2. Барометр: сырые и фильтрованные
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(time, baro_raw, 'gray', alpha=0.4, linewidth=0.8, label='Сырые данные')
        ax2.plot(time, baro_filtered, 'g-', linewidth=1.5, label='Фильтрованные')
        ax2.set_ylabel('Высота [м]')
        ax2.set_title('Барометр')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)

        # 3. Террейн: высота рельефа
        ax3 = fig.add_subplot(gs[1, :])
        ax3.plot(time, terrain, 'r-', linewidth=1.5, label='Высота рельефа')
        ax3.axhline(y=np.mean(terrain), color='k', linestyle='--', 
                    linewidth=0.8, alpha=0.5, label=f'Среднее: {np.mean(terrain):.2f} м')
        ax3.fill_between(time, 
                         np.mean(terrain) - np.std(terrain),
                         np.mean(terrain) + np.std(terrain),
                         alpha=0.15, color='r', label='±1σ')
        ax3.set_xlabel('Время [с]')
        ax3.set_ylabel('Высота рельефа [м]')
        ax3.set_title('Абсолютная высота рельефа (H_baro - H_radar)')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)

        # 4. Детализация (первые 200 точек)
        zoom = min(200, len(time))
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.plot(time[:zoom], raw_data[:zoom], 'gray', alpha=0.5, linewidth=0.8, label='Сырые')
        ax4.plot(time[:zoom], filtered_data[:zoom], 'b-', linewidth=1.8, label='Фильтр')
        ax4.set_xlabel('Время [с]')
        ax4.set_ylabel('Высота [м]')
        ax4.set_title(f'Радиоальтиметр (первые {zoom} отсчётов)')
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3)

        # 5. Гистограмма ошибок фильтрации (разница между сырым и фильтрованным)
        ax5 = fig.add_subplot(gs[2, 1])
        error_radar = raw_data - filtered_data
        ax5.hist(error_radar, bins=50, color='blue', alpha=0.6, edgecolor='black', linewidth=0.5)
        ax5.axvline(x=0, color='r', linestyle='--', linewidth=1)
        ax5.axvline(x=np.mean(error_radar), color='k', linestyle='-', linewidth=1,
                    label=f'Среднее: {np.mean(error_radar):.3f} м')
        ax5.set_xlabel('Ошибка [м]')
        ax5.set_ylabel('Частота')
        ax5.set_title('Распределение ошибки радиоальтиметра')
        ax5.legend(loc='best')
        ax5.grid(True, alpha=0.3)

        # 6. Спектральный анализ (PSD) - сравнение частотных характеристик
        ax6 = fig.add_subplot(gs[3, 0])
        if len(raw_data) > 100:
            # Вычисляем периодограмму
            f_psd, psd_raw = plt.psd(raw_data, Fs=fs, NFFT=min(256, len(raw_data)//2),
                                     pad_to=512, scale_by_freq=True, return_line=False)
            _, psd_filt = plt.psd(filtered_data, Fs=fs, NFFT=min(256, len(raw_data)//2),
                                  pad_to=512, scale_by_freq=True, return_line=False)
            
            # Перерисовываем на отдельном графике
            ax6.clear()
            f_psd = np.fft.rfftfreq(min(512, len(raw_data)), d=1/fs)
            
            # Используем Welch для лучшей оценки
            from scipy.signal import welch
            f_raw, psd_raw = welch(raw_data, fs=fs, nperseg=min(128, len(raw_data)//4))
            f_filt, psd_filt = welch(filtered_data, fs=fs, nperseg=min(128, len(raw_data)//4))
            
            ax6.loglog(f_raw, psd_raw, 'gray', alpha=0.6, label='Сырые')
            ax6.loglog(f_filt, psd_filt, 'b-', linewidth=1.5, label='Фильтрованные')
            ax6.axvline(x=2.0, color='r', linestyle='--', linewidth=0.8, alpha=0.7, label='Частота среза')
            ax6.set_xlabel('Частота [Гц]')
            ax6.set_ylabel('Спектральная плотность')
            ax6.set_title('Спектральный анализ')
            ax6.legend(loc='best')
            ax6.grid(True, alpha=0.3, which='both', linestyle='--', linewidth=0.5)

        # 7. Статистика фильтрации
        ax7 = fig.add_subplot(gs[3, 1])
        ax7.axis('off')
        
        stats_text = (
            f"Статистика фильтрации:\n"
            f"{'='*40}\n"
            f"Радиоальтиметр:\n"
            f"  Сырые: μ={np.mean(raw_data):.2f} м, σ={np.std(raw_data):.2f} м\n"
            f"  Фильтр: μ={np.mean(filtered_data):.2f} м, σ={np.std(filtered_data):.2f} м\n"
            f"  Снижение шума: {(1 - np.std(filtered_data)/np.std(raw_data))*100:.1f}%\n"
            f"\nБарометр:\n"
            f"  Сырые: μ={np.mean(baro_raw):.2f} м, σ={np.std(baro_raw):.2f} м\n"
            f"  Фильтр: μ={np.mean(baro_filtered):.2f} м, σ={np.std(baro_filtered):.2f} м\n"
            f"  Снижение шума: {(1 - np.std(baro_filtered)/np.std(baro_raw))*100:.1f}%\n"
            f"\nВысота рельефа:\n"
            f"  μ={np.mean(terrain):.2f} м, σ={np.std(terrain):.2f} м\n"
            f"  min={np.min(terrain):.2f} м, max={np.max(terrain):.2f} м"
        )
        ax7.text(0.05, 0.95, stats_text, transform=ax7.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_stability_analysis(
        noise_levels: List[float],
        metrics: Dict[str, List[float]],
        title: str = "Анализ устойчивости к шуму"
    ):
        """
        График устойчивости алгоритма при различных уровнях шума.
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib не установлен. Пропуск визуализации.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(title, fontsize=14, fontweight='bold')

        # 1. Ошибка vs уровень шума
        ax1 = axes[0, 0]
        ax1.plot(noise_levels, metrics.get('radar_error', []), 'bo-', 
                linewidth=1.5, markersize=6, label='Радиоальтиметр')
        ax1.plot(noise_levels, metrics.get('baro_error', []), 'gs-', 
                linewidth=1.5, markersize=6, label='Барометр')
        ax1.set_xlabel('Уровень шума (σ) [м]')
        ax1.set_ylabel('Средняя ошибка [м]')
        ax1.set_title('Зависимость ошибки от уровня шума')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # 2. Снижение шума vs уровень шума
        ax2 = axes[0, 1]
        ax2.plot(noise_levels, metrics.get('radar_noise_reduction', []), 'bo-', 
                linewidth=1.5, markersize=6, label='Радиоальтиметр')
        ax2.plot(noise_levels, metrics.get('baro_noise_reduction', []), 'gs-', 
                linewidth=1.5, markersize=6, label='Барометр')
        ax2.set_xlabel('Уровень шума (σ) [м]')
        ax2.set_ylabel('Снижение шума [%]')
        ax2.set_title('Эффективность подавления шума')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)

        # 3. Стабильность рельефа vs уровень шума
        ax3 = axes[1, 0]
        ax3.plot(noise_levels, metrics.get('terrain_stability', []), 'r^-', 
                linewidth=1.5, markersize=6)
        ax3.set_xlabel('Уровень шума (σ) [м]')
        ax3.set_ylabel('Стабильность рельефа (1/σ_terrain)')
        ax3.set_title('Стабильность оценки высоты рельефа')
        ax3.grid(True, alpha=0.3)

        # 4. Общая оценка качества
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        if metrics:
            avg_radar_reduction = np.mean(metrics.get('radar_noise_reduction', [0]))
            avg_baro_reduction = np.mean(metrics.get('baro_noise_reduction', [0]))
            
            stats_text = (
                f"Общая оценка качества фильтрации:\n"
                f"{'='*45}\n"
                f"Среднее снижение шума:\n"
                f"  Радиоальтиметр: {avg_radar_reduction:.1f}%\n"
                f"  Барометр:       {avg_baro_reduction:.1f}%\n"
                f"\nУстойчивость алгоритма:\n"
                f"  Ошибка растет {'медленно' if metrics.get('radar_error', [0])[-1] / metrics.get('radar_error', [0])[0] < 3 else 'быстро'}\n"
                f"  Фильтр {'стабилен' if metrics.get('terrain_stability', [0])[-1] > 0.1 * metrics.get('terrain_stability', [0])[0] else 'теряет эффективность'}\n"
                f"\nРекомендации:\n"
                f"  {'Алгоритм устойчив' if metrics.get('terrain_stability', [0])[-1] > 0.05 else 'Требуется адаптация фильтра'}"
            )
            ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        plt.tight_layout()
        plt.show()

    @staticmethod
    def analyze_noise_stability(
        processor_class,
        nmea_lines: List[str],
        fs: float = 10.0,
        noise_levels: List[float] = None,
        n_runs: int = 5
    ) -> Dict[str, List[float]]:
        """
        Анализирует устойчивость алгоритма при различных уровнях шума.
        
        Args:
            processor_class: Класс TerrainAltitudeProcessor
            nmea_lines: Список NMEA-строк
            fs: Частота дискретизации
            noise_levels: Список уровней шума для тестирования
            n_runs: Количество прогонов для каждого уровня шума
            
        Returns:
            Словарь с метриками для визуализации
        """
        if noise_levels is None:
            noise_levels = [0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0]
        
        metrics = {
            'noise_levels': [],
            'radar_error': [],
            'baro_error': [],
            'radar_noise_reduction': [],
            'baro_noise_reduction': [],
            'terrain_stability': []
        }
        
        # Базовая конфигурация (без шума барометра)
        base_processor = processor_class(
            fs=fs,
            baro_base_altitude=1500.0,
            generate_baro=False
        )
        
        # Получаем базовую высоту рельефа без шума
        base_results = []
        for line in nmea_lines:
            res = base_processor.process_sentence(line)
            if res is not None:
                base_results.append(res)
        
        if not base_results:
            print("Ошибка: нет результатов для базовой конфигурации")
            return metrics
        
        base_terrain = np.array([r['terrain_altitude'] for r in base_results])
        base_radar = np.array([r['radar_filtered'] for r in base_results])
        
        for noise_std in noise_levels:
            metrics['noise_levels'].append(noise_std)
            
            radar_errors = []
            baro_errors = []
            radar_reductions = []
            baro_reductions = []
            terrain_stabilities = []
            
            for run in range(n_runs):
                # Создаем процессор с текущим уровнем шума
                processor = processor_class(
                    fs=fs,
                    baro_base_altitude=1500.0,
                    baro_noise_std=noise_std,
                    baro_seed=run
                )
                
                results = []
                for line in nmea_lines:
                    res = processor.process_sentence(line)
                    if res is not None:
                        results.append(res)
                
                if not results:
                    continue
                
                radar_filt = np.array([r['radar_filtered'] for r in results])
                baro_filt = np.array([r['baro_filtered'] for r in results])
                terrain = np.array([r['terrain_altitude'] for r in results])
                
                # Ошибка фильтрации (отклонение от базового значения)
                radar_error = np.std(radar_filt - base_radar[:len(radar_filt)])
                baro_error = np.std(baro_filt - 1500.0)  # Относительно базовой высоты
                
                # Снижение шума (относительно сырых данных)
                radar_raw = np.array([r['radar_raw'] for r in results])
                baro_raw = np.array([r['baro_raw'] for r in results])
                
                # Для радиоальтиметра используем сырые данные из NMEA
                # Для барометра используем сгенерированные сырые данные
                radar_reduction = (1 - np.std(radar_filt) / max(np.std(radar_raw), 0.001)) * 100
                baro_reduction = (1 - np.std(baro_filt) / max(np.std(baro_raw), 0.001)) * 100
                
                # Стабильность оценки рельефа (обратная дисперсия)
                terrain_stability = 1.0 / max(np.std(terrain - base_terrain[:len(terrain)]), 0.001)
                
                radar_errors.append(radar_error)
                baro_errors.append(baro_error)
                radar_reductions.append(radar_reduction)
                baro_reductions.append(baro_reduction)
                terrain_stabilities.append(terrain_stability)
            
            # Усредняем по запускам
            metrics['radar_error'].append(np.mean(radar_errors))
            metrics['baro_error'].append(np.mean(baro_errors))
            metrics['radar_noise_reduction'].append(np.mean(radar_reductions))
            metrics['baro_noise_reduction'].append(np.mean(baro_reductions))
            metrics['terrain_stability'].append(np.mean(terrain_stabilities))
        
        return metrics


# ============================================================================
# ОСНОВНОЙ ОБРАБОТЧИК
# ============================================================================

class TerrainAltitudeProcessor:
    """
    Обработчик данных радиовысотомера и барометра.
    Вычисляет абсолютную высоту рельефа.
    """

    def __init__(
        self,
        fs: float = 10.0,
        baro_base_altitude: float = 1500.0,
        radar_cutoff: float = 2.0,
        radar_median_window: int = 5,
        radar_window_duration: float = 4.0,
        baro_q: float = 0.05,
        baro_r: float = 1.2,
        baro_noise_std: float = 1.5,
        baro_spike_prob: float = 0.01,
        baro_spike_amplitude: float = 8.0,
        baro_wind_amplitude: float = 5.0,
        baro_drift_rate: float = 0.02,
        baro_seed: Optional[int] = None,
        generate_baro: bool = True,
    ):
        """
        Args:
            fs: Частота сообщений (Гц)
            baro_base_altitude: Базовая барометрическая высота (м)
            radar_cutoff: Частота среза ФНЧ для радиоальтиметра (Гц)
            radar_median_window: Размер окна медианного фильтра радиоальтиметра
            radar_window_duration: Длительность окна ФНЧ радиоальтиметра (с)
            baro_q: Шум процесса для фильтра Калмана барометра
            baro_r: Шум измерения для фильтра Калмана барометра
            baro_noise_std: СКО шума барометра
            baro_spike_prob: Вероятность выбросов барометра
            baro_spike_amplitude: Амплитуда выбросов барометра
            baro_wind_amplitude: Амплитуда ветровых колебаний барометра
            baro_drift_rate: Скорость температурного дрейфа барометра
            baro_seed: Зерно для генерации барометрических данных
            generate_baro: Если False, барометрические данные не генерируются
        """
        self.fs = fs
        self.baro_base_altitude = baro_base_altitude
        self.generate_baro = generate_baro

        # Фильтр радиоальтиметра
        self.radar_filter = AltimeterPreFilter(
            fs=fs,
            cutoff=radar_cutoff,
            median_window=radar_median_window,
            window_duration=radar_window_duration,
        )

        # Фильтр барометра (будет инициализирован после первого измерения)
        self.baro_filter: Optional[BaroAltitudeFilter] = None
        self.baro_params = {
            'q': baro_q,
            'r': baro_r,
        }

        # Параметры генерации барометрических данных
        self.baro_gen_params = {
            'base_altitude': baro_base_altitude,
            'freq': fs,
            'noise_std': baro_noise_std,
            'spike_prob': baro_spike_prob,
            'spike_amplitude': baro_spike_amplitude,
            'wind_amplitude': baro_wind_amplitude,
            'drift_rate': baro_drift_rate,
            'seed': baro_seed,
        }

        # Для хранения предварительно сгенерированных барометрических данных
        self._baro_raw_data: Optional[np.ndarray] = None
        self._baro_index: int = 0
        self._baro_raw_values: List[float] = []

        # Результаты
        self.results: List[Dict[str, Union[float, None]]] = []

    def _get_baro_measurement(self, idx: int) -> float:
        """
        Получает барометрическое измерение (реальное или сгенерированное).
        """
        if self._baro_raw_data is not None and idx < len(self._baro_raw_data):
            return float(self._baro_raw_data[idx])
        return self.baro_base_altitude + np.random.normal(0, 0.5)

    def process_sentence(self, nmea_sentence: str, sample_index: int = -1) -> Optional[Dict[str, float]]:
        """
        Обрабатывает одно NMEA-сообщение.

        Returns:
            Dict с ключами:
                - radar_raw: сырая высота радиоальтиметра
                - radar_filtered: отфильтрованная высота радиоальтиметра
                - baro_raw: сырая барометрическая высота
                - baro_filtered: отфильтрованная барометрическая высота
                - terrain_altitude: абсолютная высота рельефа (baro - radar)
                - ready: готовность фильтров
        """
        # 1. Обработка радиоальтиметра
        radar_raw = parse_gpgga_altitude(nmea_sentence)
        if radar_raw is None:
            return None

        radar_filtered = self.radar_filter.process(nmea_sentence)
        if radar_filtered is None:
            return None

        # 2. Получение барометрического измерения
        if self.generate_baro:
            # Используем сгенерированные данные
            if self._baro_raw_data is not None:
                idx = sample_index if sample_index >= 0 else len(self.results)
                if idx < len(self._baro_raw_data):
                    baro_raw = float(self._baro_raw_data[idx])
                else:
                    baro_raw = self._get_baro_measurement(idx)
            else:
                # Генерируем все данные сразу
                n_samples = 10000  # Запас
                self._baro_raw_data = generate_baro_data_for_radar(
                    n_samples=n_samples,
                    **self.baro_gen_params
                )
                idx = sample_index if sample_index >= 0 else len(self.results)
                if idx < len(self._baro_raw_data):
                    baro_raw = float(self._baro_raw_data[idx])
                else:
                    baro_raw = self._get_baro_measurement(idx)
        else:
            # Используем базовую высоту (без шума)
            baro_raw = self.baro_base_altitude

        # 3. Инициализация фильтра барометра при первом измерении
        if self.baro_filter is None:
            self.baro_filter = BaroAltitudeFilter(
                start_altitude=baro_raw,
                q=self.baro_params['q'],
                r=self.baro_params['r']
            )

        # 4. Фильтрация барометра
        baro_filtered = self.baro_filter.update(baro_raw)

        # 5. Вычисление высоты рельефа
        terrain_altitude = baro_filtered - radar_filtered

        result = {
            'radar_raw': radar_raw,
            'radar_filtered': radar_filtered,
            'baro_raw': baro_raw,
            'baro_filtered': baro_filtered,
            'terrain_altitude': terrain_altitude,
            'ready': self.radar_filter.ready,
        }

        self.results.append(result)
        return result

    def process_file(self, filepath: str, verbose: bool = True) -> List[Dict[str, float]]:
        """
        Обрабатывает файл с NMEA-сообщениями.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {filepath}")

        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        if verbose:
            print(f"Обработка файла: {path.name}")
            print(f"Строк: {len(lines)}")
            print(f"Частота: {self.fs} Гц")
            print("-" * 60)

        # Генерируем барометрические данные заранее
        if self.generate_baro:
            self._baro_raw_data = generate_baro_data_for_radar(
                n_samples=len(lines),
                **self.baro_gen_params
            )
            if verbose:
                print(f"Сгенерировано {len(self._baro_raw_data)} барометрических отсчётов")

        results = []
        for i, line in enumerate(lines):
            result = self.process_sentence(line, i)
            if result is not None:
                results.append(result)

        if verbose:
            print("-" * 60)
            print(f"Обработано: {len(results)} значений")
            print(f"Радиоальтиметр готов: {'да' if self.radar_filter.ready else 'нет'}")
            if self.baro_filter is not None:
                print(f"Барометр прогрет: да")
            if results:
                last = results[-1]
                print(f"Последняя высота рельефа: {last['terrain_altitude']:.2f} м")
                print(f"Последняя радиовысота: {last['radar_filtered']:.2f} м")
                print(f"Последняя барометрическая высота: {last['baro_filtered']:.2f} м")

        return results

    def get_terrain_profile(self) -> np.ndarray:
        """Возвращает массив вычисленных высот рельефа."""
        return np.array([r['terrain_altitude'] for r in self.results])

    def get_radar_profile(self) -> np.ndarray:
        """Возвращает массив отфильтрованных высот радиоальтиметра."""
        return np.array([r['radar_filtered'] for r in self.results])

    def get_baro_profile(self) -> np.ndarray:
        """Возвращает массив отфильтрованных барометрических высот."""
        return np.array([r['baro_filtered'] for r in self.results])
    
    def get_radar_raw(self) -> np.ndarray:
        """Возвращает массив сырых высот радиоальтиметра."""
        return np.array([r['radar_raw'] for r in self.results])
    
    def get_baro_raw(self) -> np.ndarray:
        """Возвращает массив сырых барометрических высот."""
        return np.array([r['baro_raw'] for r in self.results])

    def get_statistics(self) -> Dict[str, float]:
        """Возвращает статистику обработки."""
        if not self.results:
            return {}

        terrain = self.get_terrain_profile()
        radar = self.get_radar_profile()
        baro = self.get_baro_profile()
        radar_raw = self.get_radar_raw()
        baro_raw = self.get_baro_raw()

        return {
            'n_samples': len(self.results),
            'terrain_mean': float(np.mean(terrain)),
            'terrain_std': float(np.std(terrain)),
            'terrain_min': float(np.min(terrain)),
            'terrain_max': float(np.max(terrain)),
            'radar_mean': float(np.mean(radar)),
            'radar_std': float(np.std(radar)),
            'baro_mean': float(np.mean(baro)),
            'baro_std': float(np.std(baro)),
            'radar_raw_std': float(np.std(radar_raw)),
            'baro_raw_std': float(np.std(baro_raw)),
            'radar_noise_reduction': (1 - np.std(radar) / max(np.std(radar_raw), 0.001)) * 100,
            'baro_noise_reduction': (1 - np.std(baro) / max(np.std(baro_raw), 0.001)) * 100,
            'ready': self.radar_filter.ready and self.baro_filter is not None,
        }

    def reset(self):
        """Полный сброс состояния."""
        self.radar_filter.reset()
        self.baro_filter = None
        self._baro_raw_data = None
        self._baro_index = 0
        self.results = []


# ============================================================================
# ТОЧКА ВХОДА
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Обработка данных радиовысотомера с барометрической коррекцией и визуализацией"
    )
    parser.add_argument(
        "file",
        nargs="?",
        default="nmea_data.txt",
        help="Файл с NMEA-сообщениями (по умолчанию nmea_data.txt)"
    )
    parser.add_argument("--fs", type=float, default=10.0, help="Частота сообщений, Гц")
    parser.add_argument("--baro-base", type=float, default=1500.0, help="Базовая барометрическая высота, м")

    # Параметры фильтра радиоальтиметра
    parser.add_argument("--radar-cutoff", type=float, default=2.0, help="Частота среза ФНЧ радиоальтиметра, Гц")
    parser.add_argument("--radar-median", type=int, default=5, help="Размер окна медианного фильтра радиоальтиметра")
    parser.add_argument("--radar-window", type=float, default=4.0, help="Длительность окна ФНЧ радиоальтиметра, с")

    # Параметры фильтра барометра
    parser.add_argument("--baro-q", type=float, default=0.05, help="Шум процесса Калмана (Q)")
    parser.add_argument("--baro-r", type=float, default=1.2, help="Шум измерения Калмана (R)")

    # Параметры генерации барометрических данных
    parser.add_argument("--baro-noise", type=float, default=1.5, help="СКО шума барометра, м")
    parser.add_argument("--baro-spike-prob", type=float, default=0.01, help="Вероятность выбросов барометра")
    parser.add_argument("--baro-spike-amp", type=float, default=8.0, help="Амплитуда выбросов барометра, м")
    parser.add_argument("--baro-wind-amp", type=float, default=5.0, help="Амплитуда ветровых колебаний, м")
    parser.add_argument("--baro-drift", type=float, default=0.02, help="Скорость температурного дрейфа, м/с")
    parser.add_argument("--baro-seed", type=int, default=None, help="Зерно для генерации барометрических данных")

    parser.add_argument("--no-barogen", action="store_true", help="Отключить генерацию барометрических данных")
    parser.add_argument("--quiet", action="store_true", help="Минимальный вывод")
    parser.add_argument("--stats", action="store_true", help="Вывести статистику")
    
    # Параметры визуализации
    parser.add_argument("--plot", action="store_true", help="Показать графики фильтрации")
    parser.add_argument("--stability", action="store_true", help="Провести анализ устойчивости к шуму")
    parser.add_argument("--save-plot", type=str, default=None, help="Сохранить график в файл (путь)")

    args = parser.parse_args()

    # Проверка наличия matplotlib
    if (args.plot or args.stability) and not MATPLOTLIB_AVAILABLE:
        print("Ошибка: для визуализации требуется matplotlib")
        print("Установите: pip install matplotlib")
        sys.exit(1)

    # Создание процессора
    processor = TerrainAltitudeProcessor(
        fs=args.fs,
        baro_base_altitude=args.baro_base,
        radar_cutoff=args.radar_cutoff,
        radar_median_window=args.radar_median,
        radar_window_duration=args.radar_window,
        baro_q=args.baro_q,
        baro_r=args.baro_r,
        baro_noise_std=args.baro_noise,
        baro_spike_prob=args.baro_spike_prob,
        baro_spike_amplitude=args.baro_spike_amp,
        baro_wind_amplitude=args.baro_wind_amp,
        baro_drift_rate=args.baro_drift,
        baro_seed=args.baro_seed,
        generate_baro=not args.no_barogen,
    )

    # Обработка файла
    results = processor.process_file(args.file, verbose=not args.quiet)

    # Вывод результатов
    if results and not args.quiet:
        print("\n" + "=" * 60)
        print("РЕЗУЛЬТАТЫ ОБРАБОТКИ")
        print("=" * 60)

        # Первые 5
        print("\nПервые 5 измерений:")
        print(f"{'№':>4} {'Radar_raw':>10} {'Radar_filt':>10} {'Baro_raw':>10} {'Baro_filt':>10} {'Terrain':>10}")
        for i, r in enumerate(results[:5]):
            print(f"{i:4d} {r['radar_raw']:10.2f} {r['radar_filtered']:10.2f} {r['baro_raw']:10.2f} {r['baro_filtered']:10.2f} {r['terrain_altitude']:10.2f}")

        # Последние 5
        print("\nПоследние 5 измерений:")
        for i, r in enumerate(results[-5:], len(results)-5):
            print(f"{i:4d} {r['radar_raw']:10.2f} {r['radar_filtered']:10.2f} {r['baro_raw']:10.2f} {r['baro_filtered']:10.2f} {r['terrain_altitude']:10.2f}")

    # Статистика
    if args.stats or args.quiet:
        stats = processor.get_statistics()
        if stats:
            print("\n" + "=" * 60)
            print("СТАТИСТИКА")
            print("=" * 60)
            print(f"Всего отсчетов:          {stats['n_samples']}")
            print(f"Высота рельефа (сред.):  {stats['terrain_mean']:.2f} м")
            print(f"Высота рельефа (σ):      {stats['terrain_std']:.2f} м")
            print(f"Высота рельефа (min):    {stats['terrain_min']:.2f} м")
            print(f"Высота рельефа (max):    {stats['terrain_max']:.2f} м")
            print(f"Радиовысота (сред.):     {stats['radar_mean']:.2f} м")
            print(f"Радиовысота (σ):         {stats['radar_std']:.2f} м")
            print(f"Барометрическая (сред.): {stats['baro_mean']:.2f} м")
            print(f"Барометрическая (σ):     {stats['baro_std']:.2f} м")
            print(f"Снижение шума (радар):   {stats['radar_noise_reduction']:.1f}%")
            print(f"Снижение шума (барометр): {stats['baro_noise_reduction']:.1f}%")
            print(f"Фильтры готовы:          {'да' if stats['ready'] else 'нет'}")

    # Визуализация результатов фильтрации
    if args.plot and results:
        print("\nПостроение графиков фильтрации...")
        
        radar_raw = processor.get_radar_raw()
        radar_filt = processor.get_radar_profile()
        baro_raw = processor.get_baro_raw()
        baro_filt = processor.get_baro_profile()
        terrain = processor.get_terrain_profile()
        
        visualizer = FilterVisualizer()
        visualizer.plot_filter_comparison(
            raw_data=radar_raw,
            filtered_data=radar_filt,
            baro_raw=baro_raw,
            baro_filtered=baro_filt,
            terrain=terrain,
            fs=args.fs,
            title=f"Фильтрация данных (fs={args.fs} Гц, cutoff={args.radar_cutoff} Гц)"
        )
        
        if args.save_plot:
            plt.savefig(args.save_plot, dpi=150, bbox_inches='tight')
            print(f"График сохранен в: {args.save_plot}")

    # Анализ устойчивости к шуму
    if args.stability:
        print("\nАнализ устойчивости алгоритма к шуму...")
        
        # Читаем NMEA строки из файла
        path = Path(args.file)
        if not path.exists():
            print(f"Ошибка: файл не найден {args.file}")
        else:
            with open(path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            
            if lines:
                visualizer = FilterVisualizer()
                metrics = visualizer.analyze_noise_stability(
                    processor_class=TerrainAltitudeProcessor,
                    nmea_lines=lines,
                    fs=args.fs,
                    noise_levels=[0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0]
                )
                visualizer.plot_stability_analysis(
                    noise_levels=metrics['noise_levels'],
                    metrics=metrics,
                    title="Анализ устойчивости к шуму"
                )
            else:
                print("Ошибка: не удалось прочитать NMEA строки из файла")

    # Сохранение результатов в файл
    if results:
        output_file = Path(args.file).stem + "_terrain_profile.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Результаты обработки данных радиовысотомера\n")
            f.write("# Формат: radar_raw, radar_filtered, baro_raw, baro_filtered, terrain_altitude\n")
            for r in results:
                f.write(f"{r['radar_raw']:.4f}, {r['radar_filtered']:.4f}, {r['baro_raw']:.4f}, {r['baro_filtered']:.4f}, {r['terrain_altitude']:.4f}\n")
        print(f"\nРезультаты сохранены в файл: {output_file}")

    return results


if __name__ == "__main__":
    main()