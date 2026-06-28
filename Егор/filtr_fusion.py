"""
Единый скрипт обработки данных радиовысотомера с барометрической коррекцией.

Объединяет:
1. Парсинг NMEA-сообщений радиоальтиметра (filtr_radio.py)
2. Генерацию зашумленных барометрических данных (random_BMP_1500.py)
3. Фильтрацию радиоальтиметра (медианный + ФНЧ с нулевой фазой)
4. Фильтрацию барометра (медианный + Калмановский фильтр)
5. Вычисление абсолютной высоты рельефа

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

    def get_statistics(self) -> Dict[str, float]:
        """Возвращает статистику обработки."""
        if not self.results:
            return {}

        terrain = self.get_terrain_profile()
        radar = self.get_radar_profile()
        baro = self.get_baro_profile()

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
        description="Обработка данных радиовысотомера с барометрической коррекцией"
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

    args = parser.parse_args()

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
            print(f"Всего отсчетов:       {stats['n_samples']}")
            print(f"Высота рельефа (сред.): {stats['terrain_mean']:.2f} м")
            print(f"Высота рельефа (σ):      {stats['terrain_std']:.2f} м")
            print(f"Высота рельефа (min):    {stats['terrain_min']:.2f} м")
            print(f"Высота рельефа (max):    {stats['terrain_max']:.2f} м")
            print(f"Радиовысота (сред.):    {stats['radar_mean']:.2f} м")
            print(f"Барометрическая (сред.): {stats['baro_mean']:.2f} м")
            print(f"Фильтры готовы:         {'да' if stats['ready'] else 'нет'}")

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