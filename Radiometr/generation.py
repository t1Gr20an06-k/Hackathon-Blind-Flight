"""
Генератор реалистичных данных радиовысотомера.

Читает CSV с эталонным профилем рельефа (временные метки, координаты, высота),
вычисляет истинные показания радиовысотомера как:
    H_radio = H_baro - H_terrain

Накладывает комплексный шум, характерный для реальной системы:
  1. Вибрационный шум (двигатель, винты) — узкополосный, 50-200 Гц
  2. Турбулентная болтанка — низкочастотный окрашенный шум, 0.1-2 Гц
  3. Шум квантования АЦП — белый, равномерный
  4. Случайные выбросы (отражения от птиц, пропуски данных)
  5. Медленный дрейф чувствительности (температурный уход)

Формат входного CSV (профиль трассы):
  t_s,x_px,y_px,elevation
  0.0,302.0,663.0,26.0
  0.1,305.68,664.57,27.0
  ...

Выходной файл (nmea_data.txt):
  $GPGGA,000000.000,,,,,,,,545.4,M,46.9,M,,*47
  $GPGGA,000100.000,,,,,,,,544.7,M,46.9,M,,*4A
  ...
"""

import csv
import math
from pathlib import Path
from typing import Tuple, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Параметры шумов (реалистичные, подобраны под лёгкий БПЛА)
# ---------------------------------------------------------------------------

class NoiseConfig:
    """
    Конфигурация шумов радиовысотомера.

    Все значения основаны на типовых характеристиках лёгких БПЛА
    (взлётная масса 10-30 кг, поршневой/электрический двигатель).
    """

    # Вибрационный шум (двигатель + винты)
    vibration_freq: float = 80.0       # Гц (типовые обороты 4800 об/мин)
    vibration_amplitude: float = 0.25  # метры
    vibration_harmonics: int = 3       # число гармоник

    # Турбулентность (болтанка)
    turbulence_cutoff: float = 0.8     # Гц
    turbulence_amplitude: float = 1.5  # метры

    # Шум квантования АЦП
    adc_resolution: float = 0.01       # метры (1 см)

    # Выбросы (птицы, переотражения, кратковременные сбои)
    outlier_probability: float = 0.02
    outlier_amplitude_mean: float = 0.0
    outlier_amplitude_std: float = 15.0

    # Температурный дрейф
    drift_rate: float = 0.01           # м/с
    drift_max: float = 2.0             # метры


# ---------------------------------------------------------------------------
# Генератор шума
# ---------------------------------------------------------------------------

class AltimeterNoiseGenerator:
    """
    Генерирует реалистичный шум радиовысотомера.

    Использует аддитивную модель: к истинному значению добавляются
    независимые компоненты шума, каждый со своей спектральной характеристикой.
    """

    def __init__(self, fs: float, config: NoiseConfig = None, seed: int = 42):
        """
        Параметры
        ---------
        fs : float
            Частота дискретизации (Гц).
        config : NoiseConfig
            Конфигурация шумов.
        seed : int
            Зерно для воспроизводимости.
        """
        self.fs = fs
        self.config = config or NoiseConfig()
        self.rng = np.random.default_rng(seed)

        # Состояние генераторов
        self._time = 0.0
        self._drift_accumulated = 0.0
        self._turbulence_state = None

    def _generate_vibration(self, n_samples: int) -> np.ndarray:
        """Генерирует узкополосный вибрационный шум (двигатель + винты)."""
        cfg = self.config
        t = np.arange(n_samples) / self.fs + self._time

        signal = np.zeros(n_samples, dtype=np.float64)
        for k in range(1, cfg.vibration_harmonics + 1):
            freq = cfg.vibration_freq * k
            amp = cfg.vibration_amplitude / k
            phase = self.rng.uniform(0, 2 * np.pi)
            signal += amp * np.sin(2 * np.pi * freq * t + phase)

        return signal

    def _generate_turbulence(self, n_samples: int) -> np.ndarray:
        """
        Генерирует окрашенный шум турбулентности.

        Метод: фильтрация белого шума через ФНЧ 1-го порядка
        с частотой среза turbulence_cutoff.
        """
        cfg = self.config

        # Белый шум
        white = self.rng.normal(0, 1, n_samples)

        # Фильтр 1-го порядка: y[n] = alpha * y[n-1] + (1-alpha) * x[n]
        alpha = math.exp(-2 * math.pi * cfg.turbulence_cutoff / self.fs)

        colored = np.zeros(n_samples, dtype=np.float64)

        # Используем предыдущее состояние, если есть
        if self._turbulence_state is not None:
            colored[0] = alpha * self._turbulence_state + (1 - alpha) * white[0]
        else:
            colored[0] = white[0]

        for i in range(1, n_samples):
            colored[i] = alpha * colored[i - 1] + (1 - alpha) * white[i]

        # Сохраняем состояние для следующего вызова
        self._turbulence_state = colored[-1]

        # Нормируем на заданную амплитуду
        colored *= cfg.turbulence_amplitude

        return colored

    def _generate_adc_noise(self, n_samples: int) -> np.ndarray:
        """Равномерный шум квантования АЦП."""
        cfg = self.config
        half_step = cfg.adc_resolution / 2
        return self.rng.uniform(-half_step, half_step, n_samples)

    def _generate_outliers(self, n_samples: int) -> np.ndarray:
        """Генерирует случайные выбросы."""
        cfg = self.config
        outliers = np.zeros(n_samples, dtype=np.float64)

        for i in range(n_samples):
            if self.rng.random() < cfg.outlier_probability:
                amp = self.rng.normal(cfg.outlier_amplitude_mean, cfg.outlier_amplitude_std)
                outliers[i] = amp

        return outliers

    def _generate_drift(self, n_samples: int) -> np.ndarray:
        """Медленный температурный дрейф смещения."""
        cfg = self.config
        dt = 1.0 / self.fs

        drift = np.zeros(n_samples, dtype=np.float64)
        for i in range(n_samples):
            step = self.rng.normal(0, cfg.drift_rate * math.sqrt(dt))
            self._drift_accumulated += step
            self._drift_accumulated = np.clip(
                self._drift_accumulated, -cfg.drift_max, cfg.drift_max
            )
            drift[i] = self._drift_accumulated

        return drift

    def apply_noise(self, true_signal: np.ndarray) -> np.ndarray:
        """
        Добавляет все компоненты шума к истинному сигналу.

        Параметры
        ---------
        true_signal : np.ndarray
            Истинные значения радиовысотомера (без шума).

        Возвращает
        ----------
        np.ndarray — зашумлённый сигнал той же длины.
        """
        n = len(true_signal)

        # Генерируем компоненты шума
        vibration = self._generate_vibration(n)
        turbulence = self._generate_turbulence(n)
        adc = self._generate_adc_noise(n)
        drift = self._generate_drift(n)
        outliers = self._generate_outliers(n)

        # Суммируем все шумы
        total_noise = vibration + turbulence + adc + drift + outliers

        # Добавляем к сигналу
        noisy = true_signal + total_noise

        # Обновляем внутреннее время
        self._time += n / self.fs

        return noisy

    def reset(self):
        """Сбрасывает состояние генератора."""
        self._time = 0.0
        self._drift_accumulated = 0.0
        self._turbulence_state = None


# ---------------------------------------------------------------------------
# Чтение эталонного CSV
# ---------------------------------------------------------------------------

def read_terrain_csv(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Читает CSV с эталонным рельефом в формате временных меток и координат.

    Формат CSV:
      t_s,x_px,y_px,elevation
      0.0,302.0,663.0,26.0
      0.1,305.68,664.57,27.0
      ...

    Возвращает:
      time : np.ndarray — время в секундах
      x : np.ndarray — координата x в пикселях
      y : np.ndarray — координата y в пикселях
      elevation : np.ndarray — абсолютная высота рельефа (м)
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"CSV-файл не найден: {filepath}")

    time_list = []
    x_list = []
    y_list = []
    elev_list = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)

        # Проверяем, есть ли заголовок
        if header:
            first_val = header[0].strip()
            # Если первая строка содержит строковые заголовки, пропускаем её
            if not first_val.replace('.', '').replace('-', '').replace('_', '').isdigit():
                # Это заголовок, уже прочитали
                pass
            else:
                # Первая строка была данными — обрабатываем
                try:
                    time_list.append(float(header[0]))
                    x_list.append(float(header[1]))
                    y_list.append(float(header[2]))
                    elev_list.append(float(header[3]))
                except (ValueError, IndexError):
                    pass

        # Читаем остальные строки
        for row in reader:
            if len(row) < 4:
                continue
            try:
                time_list.append(float(row[0].strip()))
                x_list.append(float(row[1].strip()))
                y_list.append(float(row[2].strip()))
                elev_list.append(float(row[3].strip()))
            except ValueError as e:
                print(f"Ошибка парсинга строки: {row} -> {e}")
                continue

    if not time_list:
        raise ValueError(f"Не удалось прочитать данные из файла: {filepath}")

    return (
        np.array(time_list, dtype=np.float64),
        np.array(x_list, dtype=np.float64),
        np.array(y_list, dtype=np.float64),
        np.array(elev_list, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Формирование NMEA-строк
# ---------------------------------------------------------------------------

def nmea_checksum(body_without_dollar: str) -> str:
    """
    Вычисляет контрольную сумму NMEA (XOR всех байт между $ и *).

    Параметры
    ---------
    body_without_dollar : str
        Тело сообщения без '$' (например, "GPGGA,123519.111,...")

    Возвращает
    ----------
    str — двузначное шестнадцатеричное число (например, "47").
    """
    checksum = 0
    for ch in body_without_dollar:
        checksum ^= ord(ch)
    return f"{checksum:02X}"


def format_gpgga(
    time_s: float,
    altitude_m: float,
    geoid_separation_m: float = 46.9,
) -> str:
    """
    Формирует строку $GPGGA с реалистичными полями.

    Параметры
    ---------
    time_s : float
        Время измерения в секундах от начала суток (дробное).
    altitude_m : float
        Высота антенны над геоидом (метры).
    geoid_separation_m : float
        Разница между геоидом и WGS-84 эллипсоидом (метры).

    Возвращает
    ----------
    str — валидная NMEA-строка с контрольной суммой.
    """
    # Преобразуем время в формат HHMMSS.SSS
    hours = int(time_s // 3600)
    minutes = int((time_s % 3600) // 60)
    seconds = time_s % 60
    time_str = f"{hours:02d}{minutes:02d}{seconds:06.3f}"

    # Собираем тело сообщения (без '$' и '*' в начале)
    body = (
        f"GPGGA,{time_str},,,,,,,,"
        f"{altitude_m:.1f},M,"
        f"{geoid_separation_m:.1f},M,,"
    )

    # Контрольная сумма
    cs = nmea_checksum(body)

    return f"${body}*{cs}"


# ---------------------------------------------------------------------------
# Главный генератор
# ---------------------------------------------------------------------------

def generate_nmea_file(
    terrain_csv: str,
    output_file: str,
    baro_altitude: float = 1500.0,
    fs: float = None,
    noise_config: NoiseConfig = None,
    seed: int = 42,
    max_time_offset: float = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Генерирует файл с зашумлёнными NMEA-данными радиовысотомера.

    Параметры
    ---------
    terrain_csv : str
        Путь к CSV с эталонным рельефом (t_s,x_px,y_px,elevation).
    output_file : str
        Путь к выходному файлу.
    baro_altitude : float
        Абсолютная высота БПЛА по барометру (метры).
    fs : float, optional
        Частота дискретизации (Гц). Если не указана, вычисляется из временных меток.
    noise_config : NoiseConfig
        Параметры шума. None — использовать значения по умолчанию.
    seed : int
        Зерно для воспроизводимости шума.
    max_time_offset : float, optional
        Максимальное смещение времени для имитации джиттера (секунды).

    Возвращает
    ----------
    Tuple[np.ndarray, np.ndarray, np.ndarray] — 
        (временные метки, зашумленный сигнал, истинный сигнал)
    """
    # Читаем эталонный рельеф
    time_s, x_coords, y_coords, terrain_h = read_terrain_csv(terrain_csv)
    n_points = len(terrain_h)
    print(f"Загружено точек рельефа: {n_points}")
    print(f"Диапазон времени: {time_s.min():.2f} – {time_s.max():.2f} с")
    print(f"Диапазон координат x: {x_coords.min():.1f} – {x_coords.max():.1f} px")
    print(f"Диапазон высот рельефа: {terrain_h.min():.1f} – {terrain_h.max():.1f} м")

    # Вычисляем истинные показания радиовысотомера
    true_radio_h = baro_altitude - terrain_h

    # Проверяем физическую корректность
    if (true_radio_h < 0).any():
        min_h = true_radio_h.min()
        raise ValueError(
            f"Отрицательная высота радиовысотомера ({min_h:.1f} м). "
            f"БПЛА ниже рельефа! Увеличьте baro_altitude (сейчас {baro_altitude} м)."
        )

    print(f"Истинная высота по радио: {true_radio_h.min():.1f} – {true_radio_h.max():.1f} м")

    # Определяем частоту дискретизации
    if fs is None:
        # Вычисляем из временных меток
        dt = np.mean(np.diff(time_s))
        if dt <= 0:
            raise ValueError("Временные метки должны быть строго возрастающими")
        fs = 1.0 / dt
        print(f"Частота дискретизации (по данным): {fs:.2f} Гц")
    else:
        print(f"Частота дискретизации (задана): {fs:.2f} Гц")

    # Генерируем шум
    noise_gen = AltimeterNoiseGenerator(fs=fs, config=noise_config, seed=seed)
    noisy_radio_h = noise_gen.apply_noise(true_radio_h)

    # Статистика шума
    noise_only = noisy_radio_h - true_radio_h
    print(f"\nСтатистика добавленного шума:")
    print(f"  СКО шума:           {np.std(noise_only):.3f} м")
    print(f"  Макс. отклонение:   {np.abs(noise_only).max():.3f} м")
    print(f"  Среднее смещение:   {np.mean(noise_only):.4f} м")

    # Формируем NMEA-строки
    lines = []
    for i in range(n_points):
        # Используем время из CSV для NMEA-строки
        t = time_s[i]
        
        # Добавляем джиттер, если указан
        if max_time_offset is not None and max_time_offset > 0:
            jitter = np.random.uniform(-max_time_offset, max_time_offset)
            t = max(0, t + jitter)  # Время не может быть отрицательным
        
        nmea = format_gpgga(time_s=t, altitude_m=noisy_radio_h[i])
        lines.append(nmea)

    # Пишем файл
    output_path = Path(output_file)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nВыходной файл: {output_path.absolute()}")
    print(f"Записано строк: {len(lines)}")

    return time_s, noisy_radio_h, true_radio_h


def generate_csv_output(
    terrain_csv: str,
    output_file: str,
    baro_altitude: float = 1500.0,
    fs: float = None,
    noise_config: NoiseConfig = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Генерирует CSV-файл с добавленными колонками истинного и зашумлённого сигнала.

    Параметры
    ---------
    terrain_csv : str
        Путь к CSV с эталонным рельефом (t_s,x_px,y_px,elevation).
    output_file : str
        Путь к выходному CSV-файлу.
    baro_altitude : float
        Абсолютная высота БПЛА по барометру (метры).
    fs : float, optional
        Частота дискретизации (Гц).
    noise_config : NoiseConfig
        Параметры шума.
    seed : int
        Зерно для воспроизводимости шума.

    Возвращает
    ----------
    Tuple[np.ndarray, np.ndarray, np.ndarray] — 
        (временные метки, зашумленный сигнал, истинный сигнал)
    """
    # Читаем эталонный рельеф
    time_s, x_coords, y_coords, terrain_h = read_terrain_csv(terrain_csv)
    n_points = len(terrain_h)
    print(f"Загружено точек рельефа: {n_points}")
    print(f"Диапазон времени: {time_s.min():.2f} – {time_s.max():.2f} с")
    print(f"Диапазон высот рельефа: {terrain_h.min():.1f} – {terrain_h.max():.1f} м")

    # Вычисляем истинные показания радиовысотомера
    true_radio_h = baro_altitude - terrain_h

    if (true_radio_h < 0).any():
        min_h = true_radio_h.min()
        raise ValueError(
            f"Отрицательная высота радиовысотомера ({min_h:.1f} м). "
            f"Увеличьте baro_altitude (сейчас {baro_altitude} м)."
        )

    print(f"Истинная высота по радио: {true_radio_h.min():.1f} – {true_radio_h.max():.1f} м")

    # Определяем частоту дискретизации
    if fs is None:
        dt = np.mean(np.diff(time_s))
        if dt <= 0:
            raise ValueError("Временные метки должны быть строго возрастающими")
        fs = 1.0 / dt
        print(f"Частота дискретизации (по данным): {fs:.2f} Гц")

    # Генерируем шум
    noise_gen = AltimeterNoiseGenerator(fs=fs, config=noise_config, seed=seed)
    noisy_radio_h = noise_gen.apply_noise(true_radio_h)

    # Статистика шума
    noise_only = noisy_radio_h - true_radio_h
    print(f"\nСтатистика добавленного шума:")
    print(f"  СКО шума:           {np.std(noise_only):.3f} м")
    print(f"  Макс. отклонение:   {np.abs(noise_only).max():.3f} м")
    print(f"  Среднее смещение:   {np.mean(noise_only):.4f} м")

    # Сохраняем в CSV
    output_path = Path(output_file)
    with open(output_path, "w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        # Заголовок
        writer.writerow(['t_s', 'x_px', 'y_px', 'elevation', 'radio_true', 'radio_noisy'])
        
        # Данные
        for i in range(n_points):
            writer.writerow([
                f"{time_s[i]:.4f}",
                f"{x_coords[i]:.2f}",
                f"{y_coords[i]:.2f}",
                f"{terrain_h[i]:.1f}",
                f"{true_radio_h[i]:.3f}",
                f"{noisy_radio_h[i]:.3f}",
            ])

    print(f"\nВыходной CSV-файл: {output_path.absolute()}")
    print(f"Записано строк: {n_points}")

    return time_s, noisy_radio_h, true_radio_h


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Генератор реалистичных данных радиовысотомера"
    )
    parser.add_argument(
        "terrain_csv",
        help="Путь к CSV-файлу с эталонным рельефом (t_s,x_px,y_px,elevation)",
    )
    parser.add_argument(
        "-o", "--output",
        default="nmea_data.txt",
        help="Имя выходного файла (по умолчанию nmea_data.txt)",
    )
    parser.add_argument(
        "--format",
        choices=["nmea", "csv"],
        default="nmea",
        help="Формат выходного файла: nmea или csv (по умолчанию nmea)",
    )
    parser.add_argument(
        "--baro",
        type=float,
        default=1500.0,
        help="Абсолютная высота БПЛА по барометру, метры (по умолчанию 1500)",
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=None,
        help="Частота дискретизации, Гц (по умолчанию вычисляется из данных)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Зерно для воспроизводимости шума",
    )
    parser.add_argument(
        "--vibration",
        type=float,
        default=0.25,
        help="Амплитуда вибраций, метры (по умолчанию 0.25)",
    )
    parser.add_argument(
        "--turbulence",
        type=float,
        default=1.5,
        help="Амплитуда турбулентности, метры (по умолчанию 1.5)",
    )
    parser.add_argument(
        "--outliers",
        type=float,
        default=0.02,
        help="Вероятность выброса (0-1, по умолчанию 0.02)",
    )
    parser.add_argument(
        "--jitter",
        type=float,
        default=None,
        help="Максимальный джиттер времени, секунды (по умолчанию None)",
    )
    args = parser.parse_args()

    # Настраиваем шум
    config = NoiseConfig()
    config.vibration_amplitude = args.vibration
    config.turbulence_amplitude = args.turbulence
    config.outlier_probability = args.outliers

    # Генерируем в нужном формате
    if args.format == "csv":
        generate_csv_output(
            terrain_csv=args.terrain_csv,
            output_file=args.output,
            baro_altitude=args.baro,
            fs=args.fs,
            noise_config=config,
            seed=args.seed,
        )
    else:  # nmea
        generate_nmea_file(
            terrain_csv=args.terrain_csv,
            output_file=args.output,
            baro_altitude=args.baro,
            fs=args.fs,
            noise_config=config,
            seed=args.seed,
            max_time_offset=args.jitter,
        )