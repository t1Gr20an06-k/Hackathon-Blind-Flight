"""
Предварительный фильтр данных радиовысотомера.

Читает NMEA-сообщения из текстового файла (одна строка — одно сообщение),
извлекает высоту, прогоняет через медианный фильтр и ФНЧ с нулевой фазой
на скользящем окне. Возвращает текущее отфильтрованное значение.

Каскад:
  1. Парсинг $GPGGA (контрольная сумма — заглушка, данные заведомо корректны)
  2. Медианный фильтр (окно 5 отсчётов)
  3. ФНЧ Баттерворта 2-го порядка, нулевая фаза (скользящее окно)
"""

from collections import deque
from pathlib import Path
from typing import Optional, List

import numpy as np
from scipy.signal import butter, filtfilt


# ---------------------------------------------------------------------------
# Парсинг NMEA
# ---------------------------------------------------------------------------

def parse_gpgga_altitude(sentence: str) -> Optional[float]:
    """
    Извлекает высоту антенны (Altitude, поле 9) из $GPGGA.

    Контрольная сумма НЕ проверяется — заглушка.
    В реальной системе здесь должен быть расчёт XOR и сравнение с полем после '*'.

    Возвращает:
        float — высота в метрах, или None при ошибке парсинга.
    """
    sentence = sentence.strip()

    if not sentence.startswith("$GPGGA"):
        return None

    # --- ЗАГЛУШКА: пропускаем проверку контрольной суммы ---
    # TODO: реализовать nmea_checksum_ok(sentence)
    # if not nmea_checksum_ok(sentence):
    #     return None
    # --------------------------------------------------------

    # Отрезаем чексумму (всё после '*') — нам нужны только поля
    body = sentence.split("*")[0]
    fields = body.split(",")

    # Поле 9 (индекс 9) — Altitude, units в поле 10 (индекс 10)
    if len(fields) < 11:
        return None

    try:
        alt_str = fields[9]
        if alt_str == "":
            return None
        return float(alt_str)
    except (ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# Медианный фильтр
# ---------------------------------------------------------------------------

class MedianFilter:
    """Медианный фильтр с фиксированным окном (нечётным)."""

    def __init__(self, window_size: int = 5):
        if window_size % 2 == 0:
            raise ValueError("Окно медианного фильтра должно быть нечётным")
        self._buffer: deque = deque(maxlen=window_size)

    def update(self, value: float) -> Optional[float]:
        """
        Добавляет значение в окно.
        Возвращает медиану, если буфер заполнен, иначе None.
        """
        self._buffer.append(value)
        if len(self._buffer) < self._buffer.maxlen:
            return None
        return float(np.median(self._buffer))

    @property
    def ready(self) -> bool:
        return len(self._buffer) == self._buffer.maxlen

    def reset(self):
        self._buffer.clear()


# ---------------------------------------------------------------------------
# ФНЧ Баттерворта с нулевой фазой (скользящее окно)
# ---------------------------------------------------------------------------

class ZeroPhaseButterworth:
    """
    Низкочастотный Баттерворт с filtfilt на скользящем окне.

    Не копит жёсткие пачки — хранит последние N секунд данных.
    При каждом новом отсчёте фильтрует весь буфер и возвращает последнее
    значение. Нет артефактов на стыках, устойчив к плавающей частоте.
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

        # Коэффициенты фильтра
        nyq = 0.5 * fs
        self._b, self._a = butter(order, cutoff / nyq, btype='low')

        # Минимальная длина для filtfilt: padlen = 3 * order для всех типов pad
        # Добавляем запас — 4 * order
        self._min_samples = max(4 * order, 10)

        # Кольцевой буфер
        self._buffer: deque = deque()
        self._max_samples = max(
            int(window_duration * fs),
            self._min_samples + 1  # гарантируем, что окно не меньше минимального
        )

        # Предыдущее значение (для случая, когда filtfilt не может выдать результат)
        self._last_value: Optional[float] = None

    def update(self, value: float) -> Optional[float]:
        """
        Добавляет значение, возвращает фильтрованное (или None).

        Если данных недостаточно для полноценной filtfilt, но буфер
        уже не пуст, возвращает текущее значение без фильтрации —
        это позволяет системе запуститься и постепенно накопить данные.
        """
        self._buffer.append(value)

        # Удаляем слишком старые отсчёты
        while len(self._buffer) > self._max_samples:
            self._buffer.popleft()

        # Если данных совсем мало — копим дальше
        if len(self._buffer) < self._min_samples:
            self._last_value = value
            return None

        # Фильтруем весь буфер
        arr = np.array(self._buffer, dtype=np.float64)

        # Дополнительная проверка: вдруг есть NaN
        if np.any(np.isnan(arr)):
            # Удаляем NaN и пробуем снова
            clean = arr[~np.isnan(arr)]
            if len(clean) < self._min_samples:
                self._last_value = float(value)
                return self._last_value
            arr = clean

        try:
            filtered = filtfilt(self._b, self._a, arr)
            self._last_value = float(filtered[-1])
        except ValueError as e:
            # Если filtfilt всё равно упал — возвращаем последнее валидное
            print(f"[WARN] filtfilt error: {e}, returning last valid value")
            if self._last_value is None:
                self._last_value = float(value)

        return self._last_value

    @property
    def ready(self) -> bool:
        return self._last_value is not None

    def reset(self):
        self._buffer.clear()
        self._last_value = None


# ---------------------------------------------------------------------------
# Комбинированный фильтр
# ---------------------------------------------------------------------------

class AltimeterPreFilter:
    """
    Предварительный фильтр данных радиовысотомера.

    На вход — NMEA-строка ($GPGGA).
    На выход — float (отфильтрованная высота, метры) или None.

    Параметры
    ---------
    fs : float
        Номинальная частота сообщений (Гц).
    cutoff : float
        Частота среза ФНЧ (Гц).
    median_window : int
        Размер окна медианного фильтра.
    window_duration : float
        Длительность скользящего окна ФНЧ (секунды).
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

        # Статистика
        self.raw_count = 0
        self.error_count = 0
        self.filtered_count = 0
        self.last_output: Optional[float] = None

    def process(self, nmea_sentence: str) -> Optional[float]:
        """
        Обрабатывает одно NMEA-сообщение.

        Возвращает:
            float — отфильтрованная высота (метры), или None.
        """
        self.raw_count += 1

        # 1. Парсинг
        altitude = parse_gpgga_altitude(nmea_sentence)
        if altitude is None:
            self.error_count += 1
            return None

        # 2. Медиана
        med = self.median.update(altitude)
        if med is None:
            return None

        # 3. Баттерворт с нулевой фазой
        out = self.butter.update(med)
        if out is None:
            return None

        self.filtered_count += 1
        self.last_output = out
        return out

    @property
    def ready(self) -> bool:
        """Оба каскада прогреты и выдают данные."""
        return self.median.ready and self.butter.ready

    def reset(self):
        """Полный сброс состояния."""
        self.median.reset()
        self.butter.reset()
        self.raw_count = 0
        self.error_count = 0
        self.filtered_count = 0
        self.last_output = None


# ---------------------------------------------------------------------------
# Обработка файла
# ---------------------------------------------------------------------------

def process_file(
    filepath: str,
    filt: AltimeterPreFilter,
    verbose: bool = True,
) -> List[float]:
    """
    Читает NMEA-сообщения из файла, пропускает через фильтр.

    Параметры
    ---------
    filepath : str
        Путь к файлу (одна строка — одно сообщение).
    filt : AltimeterPreFilter
        Экземпляр фильтра.
    verbose : bool
        Печатать ли каждое значение.

    Возвращает
    ----------
    list[float] — массив отфильтрованных значений.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {filepath}")

    results = []

    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"Обработка файла: {path.name}")
    print(f"Строк: {len(lines)}")
    print(f"Частота среза ФНЧ: {filt.butter.cutoff:.2f} Гц")
    print(f"Окно ФНЧ: {filt.butter.window_duration:.1f} с "
          f"({filt.butter._max_samples} отсчётов)")
    print(f"Мин. отсчётов для filtfilt: {filt.butter._min_samples}")
    print("-" * 50)

    for i, line in enumerate(lines):
        out = filt.process(line)

        if out is not None:
            results.append(out)
            if verbose:
                print(f"[{i:4d}] {out:8.2f} м")
        elif verbose and i < 20:
            # Первые 20 строк показываем статус прогрева
            alt = parse_gpgga_altitude(line)
            if alt is not None:
                print(f"[{i:4d}] (прогрев фильтра...) сырая: {alt:.2f} м")

    print("-" * 50)
    print(f"Принято строк:      {filt.raw_count}")
    print(f"Ошибок парсинга:    {filt.error_count}")
    print(f"Отфильтровано:      {filt.filtered_count}")
    print(f"Фильтр прогрет:     {'да' if filt.ready else 'нет'}")
    if filt.last_output is not None:
        print(f"Последнее значение: {filt.last_output:.2f} м")
    print(f"Результат:          {len(results)} значений")

    return results


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Фильтр радиовысотомера: NMEA -> чистая высота"
    )
    parser.add_argument(
        "file",
        nargs="?",
        default="nmea_data.txt",
        help="Файл с NMEA-сообщениями (по умолчанию nmea_data.txt в папке скрипта)",
    )
    parser.add_argument("--fs", type=float, default=10.0, help="Частота сообщений, Гц")
    parser.add_argument("--cutoff", type=float, default=2.0, help="Частота среза ФНЧ, Гц")
    parser.add_argument("--window", type=float, default=4.0, help="Окно ФНЧ, секунды")
    parser.add_argument("--quiet", action="store_true", help="Только статистика")
    args = parser.parse_args()

    filt = AltimeterPreFilter(
        fs=args.fs,
        cutoff=args.cutoff,
        window_duration=args.window,
    )

    results = process_file(args.file, filt, verbose=not args.quiet)

    if results:
        print(f"\nПервые 5: {[f'{v:.2f}' for v in results[:5]]}")
        print(f"Последние 5: {[f'{v:.2f}' for v in results[-5:]]}")