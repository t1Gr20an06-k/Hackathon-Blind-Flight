import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


class BaroAltitudeFilter:
    """
    Каскадный фильтр высоты: Медиана (3) + Калман (1D).
    Полностью автономен, не требует внешних библиотек кроме numpy.
    """
    def __init__(self, start_altitude: float, q: float = 0.05, r: float = 1.2):
        """
        Args:
            start_altitude: Начальная высота в метрах.
            q: Ковариация шума процесса (чувствительность к движению).
            r: Ковариация шума измерения (степень фильтрации).
        """
        # Буфер для медианного фильтра (3 последних значения)
        self._buf = [start_altitude] * 3
        self._buf_idx = 0
        
        # Состояние фильтра Калмана
        self._x_hat = start_altitude  # Оценка высоты
        self._p = 1.0                 # Ковариация ошибки
        self._q = q                   # Шум процесса
        self._r = r                   # Шум измерения
        
    def _median3(self, a: float, b: float, c: float) -> float:
        """Медиана трех чисел (оптимизированная, без сортировки массива)."""
        if a > b:
            a, b = b, a
        if b > c:
            b, c = c, b
        if a > b:
            a, b = b, a
        return b
    
    def update(self, raw_altitude: float) -> float:
        """
        Обработать одно сырое измерение и вернуть отфильтрованную высоту.
        
        Args:
            raw_altitude: Сырое значение высоты с барометра [м].
        Returns:
            Отфильтрованное значение высоты [м].
        """
        # === Шаг 1: Медианный фильтр ===
        self._buf[self._buf_idx] = raw_altitude
        self._buf_idx = (self._buf_idx + 1) % 3
        
        # Вычисляем медиану текущего окна
        clean_measurement = self._median3(self._buf[0], self._buf[1], self._buf[2])
        
        # === Шаг 2: Фильтр Калмана ===
        # Предсказание
        self._p = self._p + self._q
        
        # Коррекция
        k_gain = self._p / (self._p + self._r)
        self._x_hat = self._x_hat + k_gain * (clean_measurement - self._x_hat)
        self._p = (1.0 - k_gain) * self._p
        
        return self._x_hat
    
    def reset(self, altitude: float):
        """Сброс фильтра на новую высоту."""
        self._buf = [altitude] * 3
        self._x_hat = altitude
        self._p = 1.0


def parse_data_file(filepath: str) -> tuple:
    """
    Чтение данных из txt-файла.
    
    Поддерживаемые форматы:
    1. Один столбец — значения высоты (сырые данные).
    2. Два столбца — время и высота.
    3. Три столбца — время, истинная высота, сырая высота.
    
    Строки, начинающиеся с # или //, считаются комментариями.
    
    Args:
        filepath: Путь к файлу с данными.
    
    Returns:
        raw_alt: Массив сырых значений высоты.
        true_alt: Массив истинных значений (если есть), иначе None.
        freq: Частота дискретизации (определяется из заголовка или вычисляется).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Файл '{filepath}' не найден.")
    
    data = []
    freq = None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Пропускаем пустые строки
            if not line:
                continue
            
            # Проверяем комментарии на наличие информации о частоте
            if line.startswith('#') or line.startswith('//'):
                if 'Frequency:' in line or 'frequency:' in line:
                    # Пытаемся извлечь частоту из строки типа "Frequency: 10.0Hz"
                    import re
                    match = re.search(r'(\d+\.?\d*)\s*Hz', line, re.IGNORECASE)
                    if match:
                        freq = float(match.group(1))
                continue
            
            # Пробуем разные разделители
            for sep in [',', '\t', ' ']:
                if sep in line:
                    parts = [p.strip() for p in line.split(sep) if p.strip()]
                    break
            else:
                # Если разделителей нет — одно значение
                parts = [line]
            
            # Конвертируем в числа
            try:
                numbers = [float(p) for p in parts]
                data.append(numbers)
            except ValueError:
                print(f"Предупреждение: пропущена строка (не числовые данные): '{line}'")
                continue
    
    if not data:
        raise ValueError("Файл не содержит числовых данных.")
    
    # Преобразуем в numpy массив
    data = np.array(data)
    
    if data.ndim == 1 or data.shape[1] == 1:
        # Один столбец — только сырые данные
        raw_alt = data.flatten()
        true_alt = None
        print(f"Загружено {len(raw_alt)} значений (один столбец: сырая высота).")
        
    elif data.shape[1] == 2:
        # Два столбца — время и сырая высота
        raw_alt = data[:, 1]
        true_alt = None
        print(f"Загружено {len(raw_alt)} значений (два столбца: время + сырая высота).")
        
    elif data.shape[1] >= 3:
        # Три столбца — время, истинная высота, сырая высота
        true_alt = data[:, 1]
        raw_alt = data[:, 2]
        print(f"Загружено {len(raw_alt)} значений (три столбца: время + истинная + сырая).")
    
    else:
        raise ValueError(f"Неизвестный формат данных: {data.shape[1]} столбцов.")
    
    # Если частота не определена из заголовка, вычисляем по времени
    if freq is None and data.shape[1] >= 2:
        time = data[:, 0]
        if len(time) > 1:
            freq = 1.0 / (time[1] - time[0])
            print(f"Частота определена по данным: {freq:.1f} Гц")
    
    return raw_alt, true_alt, freq


def get_file_path() -> str:
    """
    Запрашивает у пользователя путь к файлу с данными.
    Проверяет существование файла.
    """
    print("=" * 60)
    print("ФИЛЬТРАЦИЯ БАРОМЕТРИЧЕСКИХ ДАННЫХ")
    print("Каскадный фильтр: Медиана(3) + Калман(1D)")
    print("=" * 60)
    
    while True:
        print("\nВведите путь к txt-файлу с данными:")
        print("  Пример: C:\\Users\\79182\\Desktop\\hack\\baro_flight_60s_50hz.txt")
        print("  Можно перетащить файл в окно терминала")
        
        filepath = input("\nПуть к файлу: ").strip()
        
        # Убираем кавычки, если пользователь их добавил
        filepath = filepath.strip('"').strip("'")
        
        if not filepath:
            print("✗ Путь не может быть пустым.")
            continue
        
        if not os.path.exists(filepath):
            print(f"✗ Файл не найден: {filepath}")
            print("  Проверьте путь и попробуйте снова.")
            continue
        
        if not filepath.lower().endswith('.txt'):
            print("⚠ Предупреждение: файл не имеет расширения .txt")
            confirm = input("  Продолжить? (y/n): ").strip().lower()
            if confirm != 'y':
                continue
        
        print(f"✓ Файл найден: {filepath}")
        return filepath


def get_filter_parameters() -> tuple:
    """
    Запрашивает параметры фильтра Калмана.
    """
    print("\n" + "-" * 60)
    print("Настройка параметров фильтра (нажмите Enter для значений по умолчанию)")
    print("-" * 60)
    
    q = 0.05
    r = 1.2
    
    # Q - шум процесса
    while True:
        user_input = input(f"Шум процесса Q (по умолчанию {q}): ").strip()
        if not user_input:
            break
        try:
            q = float(user_input)
            if q <= 0:
                print("✗ Q должен быть положительным числом.")
                continue
            break
        except ValueError:
            print("✗ Введите число.")
    
    # R - шум измерения
    while True:
        user_input = input(f"Шум измерения R (по умолчанию {r}): ").strip()
        if not user_input:
            break
        try:
            r = float(user_input)
            if r <= 0:
                print("✗ R должен быть положительным числом.")
                continue
            break
        except ValueError:
            print("✗ Введите число.")
    
    print(f"\nПараметры фильтра: Q={q}, R={r}")
    print("  • Увеличьте Q — фильтр быстрее реагирует на изменения")
    print("  • Увеличьте R — фильтр сильнее сглаживает шум")
    
    return q, r


def get_frequency() -> float:
    """
    Запрашивает частоту дискретизации, если она не определена из файла.
    """
    print("\n" + "-" * 60)
    print("Частота дискретизации не определена из файла.")
    
    while True:
        user_input = input("Введите частоту дискретизации [Гц] (например, 50): ").strip()
        try:
            freq = float(user_input)
            if freq <= 0:
                print("✗ Частота должна быть положительным числом.")
                continue
            return freq
        except ValueError:
            print("✗ Введите число.")


def plot_results(raw_alt: np.ndarray, filtered_alt: np.ndarray, 
                 true_alt: Optional[np.ndarray] = None, freq: float = 1.0):
    """
    Построение графиков результатов фильтрации.
    """
    n_samples = len(raw_alt)
    time = np.arange(n_samples) / freq
    
    # Определяем количество подграфиков
    if true_alt is not None:
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    else:
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # График 1: Полный сигнал
    ax1 = axes[0]
    ax1.plot(time, raw_alt, 'gray', alpha=0.3, linewidth=0.5, label='Сырые данные')
    if true_alt is not None:
        ax1.plot(time, true_alt, 'b-', linewidth=1.5, label='Истинная высота')
    ax1.plot(time, filtered_alt, 'r-', linewidth=1.2, label='Отфильтрованная')
    ax1.set_ylabel('Высота [м]')
    ax1.set_title('Результаты фильтрации (полный интервал)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # График 2: Детализация (первые 20% или 500 точек)
    ax2 = axes[1]
    zoom_samples = min(int(n_samples * 0.2), 500)
    zoom_end = max(zoom_samples, 50)
    
    ax2.plot(time[:zoom_end], raw_alt[:zoom_end], 'gray', alpha=0.3, linewidth=0.8, label='Сырые')
    if true_alt is not None:
        ax2.plot(time[:zoom_end], true_alt[:zoom_end], 'b-', linewidth=1.5, label='Истинная')
    ax2.plot(time[:zoom_end], filtered_alt[:zoom_end], 'r-', linewidth=1.2, label='Фильтр')
    ax2.set_xlabel('Время [с]')
    ax2.set_ylabel('Высота [м]')
    ax2.set_title(f'Детализация (первые {zoom_end} отсчетов)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # График 3: Ошибка (если есть истинные данные)
    if true_alt is not None:
        ax3 = axes[2]
        error_raw = raw_alt - true_alt
        error_filtered = filtered_alt - true_alt
        
        ax3.plot(time, error_raw, 'gray', alpha=0.3, linewidth=0.5, label='Ошибка сырых')
        ax3.plot(time, error_filtered, 'r-', linewidth=1.0, label='Ошибка фильтрованных')
        ax3.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        ax3.set_xlabel('Время [с]')
        ax3.set_ylabel('Ошибка [м]')
        ax3.set_title('Ошибка относительно истинной высоты')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def print_statistics(raw_alt: np.ndarray, filtered_alt: np.ndarray, 
                     true_alt: Optional[np.ndarray] = None):
    """Вывод статистики фильтрации."""
    print("\n" + "=" * 60)
    print("СТАТИСТИКА ФИЛЬТРАЦИИ")
    print("=" * 60)
    
    print(f"Количество отсчетов:     {len(raw_alt)}")
    print(f"Диапазон сырых данных:   [{np.min(raw_alt):.3f}, {np.max(raw_alt):.3f}] м")
    print(f"Диапазон фильтрованных:  [{np.min(filtered_alt):.3f}, {np.max(filtered_alt):.3f}] м")
    print(f"СКО сырых данных:        {np.std(raw_alt):.3f} м")
    print(f"СКО фильтрованных:       {np.std(filtered_alt):.3f} м")
    
    # Сглаживание (дрожание между соседними отсчетами)
    raw_jitter = np.std(np.diff(raw_alt))
    filt_jitter = np.std(np.diff(filtered_alt))
    print(f"Дрожание (jitter) сырых:  {raw_jitter:.4f} м/отсчет")
    print(f"Дрожание (jitter) фильтр: {filt_jitter:.4f} м/отсчет")
    
    if true_alt is not None:
        mae_raw = np.mean(np.abs(raw_alt - true_alt))
        mae_filt = np.mean(np.abs(filtered_alt - true_alt))
        rmse_raw = np.sqrt(np.mean((raw_alt - true_alt)**2))
        rmse_filt = np.sqrt(np.mean((filtered_alt - true_alt)**2))
        
        improvement_mae = (1 - mae_filt / mae_raw) * 100 if mae_raw > 0 else 0
        improvement_rmse = (1 - rmse_filt / rmse_raw) * 100 if rmse_raw > 0 else 0
        
        print(f"\nСравнение с истинной высотой:")
        print(f"  MAE сырых:        {mae_raw:.4f} м")
        print(f"  MAE фильтрованных: {mae_filt:.4f} м")
        print(f"  RMSE сырых:        {rmse_raw:.4f} м")
        print(f"  RMSE фильтрованных: {rmse_filt:.4f} м")
        print(f"  Улучшение по MAE:  {improvement_mae:.1f}%")
        print(f"  Улучшение по RMSE: {improvement_rmse:.1f}%")
    
    # Обнаружение выбросов в сырых данных
    threshold = 3 * np.std(raw_alt)
    outliers = np.sum(np.abs(raw_alt - np.mean(raw_alt)) > threshold)
    if outliers > 0:
        print(f"\nОбнаружено выбросов (>3σ): {outliers} шт. ({outliers/len(raw_alt)*100:.1f}%)")
    
    print("=" * 60)


def main():
    """Основная функция."""
    
    # Шаг 1: Запрашиваем путь к файлу
    filepath = get_file_path()
    
    # Шаг 2: Читаем данные
    print("\n" + "-" * 60)
    print("Чтение данных...")
    
    try:
        raw_alt, true_alt, freq = parse_data_file(filepath)
    except (FileNotFoundError, ValueError) as e:
        print(f"✗ Ошибка при чтении файла: {e}")
        sys.exit(1)
    
    # Шаг 3: Запрашиваем частоту, если не определена
    if freq is None:
        freq = get_frequency()
    else:
        print(f"Частота дискретизации: {freq:.1f} Гц")
    
    # Шаг 4: Запрашиваем параметры фильтра
    q, r = get_filter_parameters()
    
    # Шаг 5: Инициализация и запуск фильтра
    print("\n" + "-" * 60)
    print("Фильтрация данных...")
    
    baro_filter = BaroAltitudeFilter(
        start_altitude=raw_alt[0],
        q=q,
        r=r
    )
    
    filtered_alt = np.zeros_like(raw_alt)
    for i in range(len(raw_alt)):
        filtered_alt[i] = baro_filter.update(raw_alt[i])
    
    print(f"✓ Обработано {len(raw_alt)} отсчетов")
    
    # Шаг 6: Статистика
    print_statistics(raw_alt, filtered_alt, true_alt)
    
    # Шаг 7: Визуализация
    print("\nПостроение графиков...")
    plot_results(raw_alt, filtered_alt, true_alt, freq)
    
    print("\n" + "=" * 60)
    print("Готово!")
    print("=" * 60)


if __name__ == "__main__":
    main()