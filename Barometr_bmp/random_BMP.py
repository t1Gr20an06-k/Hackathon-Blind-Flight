import sys
import os
import numpy as np
from typing import Optional


def get_output_dir():
    """
    Запрашивает у пользователя путь к папке для сохранения.
    Создаёт папку, если её нет.
    Возвращает путь к папке.
    """
    print("=" * 60)
    print("ГЕНЕРАТОР ТЕСТОВЫХ ДАННЫХ ДЛЯ БАРОМЕТРИЧЕСКОГО ФИЛЬТРА")
    print("=" * 60)
    
    while True:
        print("\nКуда сохранить сгенерированные файлы?")
        print("  • Можно ввести полный путь: C:\\Users\\79182\\Desktop\\hack")
        print("  • Можно ввести относительно текущей папки: data или ./output")
        print("  • Можно нажать Enter — сохранит в текущую папку")
        
        user_input = input("\nПуть к папке: ").strip()
        
        # Если ничего не ввели — используем текущую папку
        if not user_input:
            output_dir = os.getcwd()
            print(f"Выбрана текущая папка: {output_dir}")
        else:
            output_dir = user_input
        
        # Пробуем создать папку
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"✓ Папка готова: {output_dir}")
            return output_dir
        except Exception as e:
            print(f"✗ Ошибка: не удалось создать папку '{output_dir}'")
            print(f"  {e}")
            print("  Попробуйте другой путь.")


def generate_baro_data(
    duration: float = 60.0,
    freq: float = 50.0,
    noise_std: float = 0.3,
    spike_prob: float = 0.02,
    spike_amplitude: float = 5.0,
    drift_rate: float = 0.01,
    seed: Optional[int] = None
) -> tuple:
    """
    Генератор реалистичных данных барометра.
    
    Args:
        duration: Длительность симуляции [с].
        freq: Частота опроса датчика [Гц].
        noise_std: СКО шума измерений [м].
        spike_prob: Вероятность импульсного выброса на одном отсчете.
        spike_amplitude: Амплитуда выбросов [м].
        drift_rate: Скорость температурного дрейфа [м/с].
        seed: Зерно для воспроизводимости (None = случайно).
    
    Returns:
        time: Массив времени [с].
        true_alt: Истинная высота (без шума) [м].
        raw_alt: Сырые данные с датчика (с шумом и выбросами) [м].
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_samples = int(duration * freq)
    time = np.arange(n_samples) / freq
    
    # === Истинная высота: сложная траектория ===
    true_alt = np.zeros(n_samples)
    
    # Фаза 1: Взлет (0-20% времени)
    phase1_end = int(n_samples * 0.2)
    true_alt[:phase1_end] = np.linspace(0, 15, phase1_end)
    vibration = 0.5 * np.sin(2 * np.pi * 3.0 * time[:phase1_end])
    true_alt[:phase1_end] += vibration
    
    # Фаза 2: Набор высоты с ускорением (20-35%)
    phase2_start = phase1_end
    phase2_end = int(n_samples * 0.35)
    phase2_t = time[phase2_start:phase2_end] - time[phase2_start]
    phase2_duration = time[phase2_end] - time[phase2_start]
    true_alt[phase2_start:phase2_end] = 15 + 10 * (phase2_t / phase2_duration)**2
    
    # Фаза 3: Плавное снижение (35-50%)
    phase3_start = phase2_end
    phase3_end = int(n_samples * 0.5)
    phase3_t = time[phase3_start:phase3_end] - time[phase3_start]
    phase3_duration = time[phase3_end] - time[phase3_start]
    true_alt[phase3_start:phase3_end] = 25 - 5 * np.sin(np.pi * phase3_t / phase3_duration)
    
    # Фаза 4: Крейсерский полет с синусоидальными колебаниями (50-85%)
    phase4_start = phase3_end
    phase4_end = int(n_samples * 0.85)
    phase4_t = time[phase4_start:phase4_end]
    true_alt[phase4_start:phase4_end] = (
        20.0 
        + 3.0 * np.sin(2 * np.pi * 0.15 * phase4_t)
        + 1.5 * np.cos(2 * np.pi * 0.4 * phase4_t)
        + 0.8 * np.sin(2 * np.pi * 0.05 * phase4_t)
    )
    
    # Фаза 5: Посадка (85-100%)
    phase5_start = phase4_end
    phase5_t = time[phase5_start:] - time[phase5_start]
    phase5_duration = time[-1] - time[phase5_start]
    descent = 20 * np.exp(-3 * phase5_t / phase5_duration)
    flutter = 0.3 * np.sin(2 * np.pi * 8.0 * phase5_t) * np.exp(-2 * phase5_t / phase5_duration)
    true_alt[phase5_start:] = descent + flutter
    
    # === Добавляем шум и артефакты ===
    noise = np.random.normal(0, noise_std, n_samples)
    spikes_mask = np.random.rand(n_samples) < spike_prob
    spike_values = np.random.normal(0, spike_amplitude, n_samples)
    spike_signs = np.random.choice([-1, 1], n_samples)
    spikes = np.where(spikes_mask, spike_values * spike_signs, 0)
    drift = drift_rate * time * np.random.normal(0, 1)
    vibration_noise = 0.15 * np.sin(2 * np.pi * 17.3 * time) * np.random.normal(1, 0.2, n_samples)
    
    raw_alt = true_alt + noise + spikes + drift + vibration_noise
    
    return time, true_alt, raw_alt


def generate_flat_data(
    duration: float = 30.0,
    freq: float = 50.0,
    altitude: float = 100.0,
    noise_std: float = 0.2,
    spike_prob: float = 0.01,
    spike_amplitude: float = 3.0
) -> tuple:
    """
    Генератор данных с постоянной высотой (для тестирования стабильности).
    """
    n_samples = int(duration * freq)
    time = np.arange(n_samples) / freq
    true_alt = np.full(n_samples, altitude)
    noise = np.random.normal(0, noise_std, n_samples)
    spikes_mask = np.random.rand(n_samples) < spike_prob
    spike_values = np.random.normal(0, spike_amplitude, n_samples)
    spikes = np.where(spikes_mask, spike_values, 0)
    raw_alt = true_alt + noise + spikes
    return time, true_alt, raw_alt


def generate_step_data(
    duration: float = 20.0,
    freq: float = 50.0,
    steps: list = None,
    noise_std: float = 0.2,
    spike_prob: float = 0.01,
    spike_amplitude: float = 2.0
) -> tuple:
    """
    Генератор ступенчатых изменений высоты (тест на отклик фильтра).
    """
    if steps is None:
        steps = [(0, 0), (3, 10), (8, 10), (8.5, 3), (15, 3), (16, 8), (20, 8)]
    
    n_samples = int(duration * freq)
    time = np.arange(n_samples) / freq
    true_alt = np.zeros(n_samples)
    
    for i, (t_start, alt) in enumerate(steps):
        if i < len(steps) - 1:
            t_end = steps[i + 1][0]
        else:
            t_end = duration
        idx_start = int(t_start * freq)
        idx_end = min(int(t_end * freq), n_samples)
        if idx_start < n_samples:
            true_alt[idx_start:idx_end] = alt
    
    noise = np.random.normal(0, noise_std, n_samples)
    spikes_mask = np.random.rand(n_samples) < spike_prob
    spike_values = np.random.normal(0, spike_amplitude, n_samples)
    spikes = np.where(spikes_mask, spike_values, 0)
    raw_alt = true_alt + noise + spikes
    
    return time, true_alt, raw_alt


def save_data_file(
    filepath: str,
    time: np.ndarray,
    true_alt: np.ndarray,
    raw_alt: np.ndarray,
    format_type: int = 3,
    delimiter: str = ',',
    add_header: bool = True
):
    """
    Сохраняет сгенерированные данные в txt-файл.
    """
    output_dir = os.path.dirname(filepath)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    n_samples = len(time)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        if add_header:
            f.write(f"# Generated barometer data\n")
            f.write(f"# Samples: {n_samples}\n")
            f.write(f"# Duration: {time[-1]:.1f}s, Frequency: {1/(time[1]-time[0]):.1f}Hz\n")
            if format_type == 1:
                f.write(f"# Format: raw_altitude\n")
            elif format_type == 2:
                f.write(f"# Format: time, raw_altitude\n")
            else:
                f.write(f"# Format: time, true_altitude, raw_altitude\n")
            f.write(f"#\n")
        
        for i in range(n_samples):
            if format_type == 1:
                f.write(f"{raw_alt[i]:.4f}\n")
            elif format_type == 2:
                f.write(f"{time[i]:.6f}{delimiter} {raw_alt[i]:.4f}\n")
            else:
                f.write(f"{time[i]:.6f}{delimiter} {true_alt[i]:.4f}{delimiter} {raw_alt[i]:.4f}\n")
    
    print(f"\n✓ Данные сохранены в: {filepath}")
    print(f"  Отсчетов: {n_samples}")
    print(f"  Формат: тип {format_type} ({'сырые' if format_type==1 else 'время+сырые' if format_type==2 else 'время+истинные+сырые'})")
    print(f"  Разделитель: '{delimiter}'")


def choose_data_type():
    """Запрашивает у пользователя тип данных."""
    print("\n" + "-" * 60)
    print("Выберите тип тестовых данных:")
    print("  1. flight — Реалистичный полет (взлет, крейсер, посадка)")
    print("  2. flat   — Постоянная высота (тест стабильности)")
    print("  3. step   — Ступенчатые изменения (тест отклика)")
    print("-" * 60)
    
    while True:
        choice = input("Введите номер (1-3) или название: ").strip().lower()
        
        if choice in ['1', 'flight']:
            return 'flight'
        elif choice in ['2', 'flat']:
            return 'flat'
        elif choice in ['3', 'step']:
            return 'step'
        else:
            print("✗ Неверный выбор. Введите 1, 2, 3 или flight, flat, step.")


def choose_parameters(data_type: str) -> dict:
    """Запрашивает параметры генерации в зависимости от типа данных."""
    
    params = {
        'duration': 60.0,
        'freq': 50.0,
        'noise': 0.3,
        'spike_prob': 0.02,
        'spike_ampl': 5.0,
        'altitude': 100.0,
        'seed': None,
        'format': 3,
        'sep': ','
    }
    
    print("\n" + "-" * 60)
    print("Настройка параметров генерации (нажмите Enter для значений по умолчанию)")
    print("-" * 60)
    
    # Длительность
    while True:
        user_input = input(f"Длительность [с] (по умолчанию {params['duration']}): ").strip()
        if not user_input:
            break
        try:
            params['duration'] = float(user_input)
            if params['duration'] <= 0:
                print("✗ Длительность должна быть положительным числом.")
                continue
            break
        except ValueError:
            print("✗ Введите число.")
    
    # Частота
    while True:
        user_input = input(f"Частота дискретизации [Гц] (по умолчанию {params['freq']}): ").strip()
        if not user_input:
            break
        try:
            params['freq'] = float(user_input)
            if params['freq'] <= 0:
                print("✗ Частота должна быть положительным числом.")
                continue
            break
        except ValueError:
            print("✗ Введите число.")
    
    # СКО шума
    while True:
        user_input = input(f"СКО шума [м] (по умолчанию {params['noise']}): ").strip()
        if not user_input:
            break
        try:
            params['noise'] = float(user_input)
            if params['noise'] < 0:
                print("✗ СКО шума не может быть отрицательным.")
                continue
            break
        except ValueError:
            print("✗ Введите число.")
    
    # Для flat запрашиваем высоту
    if data_type == 'flat':
        while True:
            user_input = input(f"Постоянная высота [м] (по умолчанию {params['altitude']}): ").strip()
            if not user_input:
                break
            try:
                params['altitude'] = float(user_input)
                break
            except ValueError:
                print("✗ Введите число.")
    
    # Зерно (seed)
    user_input = input(f"Зерно ГПСЧ для воспроизводимости (Enter = случайно): ").strip()
    if user_input:
        try:
            params['seed'] = int(user_input)
        except ValueError:
            print("✗ Введите целое число. Будет использовано случайное зерно.")
    
    # Формат вывода
    print("\nФормат выходного файла:")
    print("  1 — Только сырая высота")
    print("  2 — Время + сырая высота")
    print("  3 — Время + истинная + сырая (по умолчанию)")
    while True:
        user_input = input("Выберите формат (1-3, Enter = 3): ").strip()
        if not user_input:
            break
        if user_input in ['1', '2', '3']:
            params['format'] = int(user_input)
            break
        else:
            print("✗ Введите 1, 2 или 3.")
    
    return params


def main():
    """Основная функция генератора."""
    
    # Шаг 1: Запрашиваем путь к папке
    output_dir = get_output_dir()
    
    # Шаг 2: Выбираем тип данных
    data_type = choose_data_type()
    
    # Шаг 3: Настраиваем параметры
    params = choose_parameters(data_type)
    
    # Формируем имя файла и полный путь
    if params['seed'] is not None:
        filename = f"baro_{data_type}_{int(params['duration'])}s_{int(params['freq'])}hz_seed{params['seed']}.txt"
    else:
        filename = f"baro_{data_type}_{int(params['duration'])}s_{int(params['freq'])}hz.txt"
    
    output_path = os.path.join(output_dir, filename)
    
    print("\n" + "=" * 60)
    print("ГЕНЕРАЦИЯ ДАННЫХ")
    print("=" * 60)
    print(f"Тип данных:     {data_type}")
    print(f"Длительность:   {params['duration']} с")
    print(f"Частота:        {params['freq']} Гц")
    print(f"СКО шума:       {params['noise']} м")
    if data_type == 'flat':
        print(f"Высота:         {params['altitude']} м")
    if params['seed'] is not None:
        print(f"Seed:           {params['seed']}")
    print(f"Выходной файл:  {output_path}")
    print("-" * 60)
    
    # Генерация в зависимости от типа
    print("Генерация данных...")
    
    if data_type == 'flight':
        time, true_alt, raw_alt = generate_baro_data(
            duration=params['duration'],
            freq=params['freq'],
            noise_std=params['noise'],
            spike_prob=params['spike_prob'],
            spike_amplitude=params['spike_ampl'],
            seed=params['seed']
        )
    elif data_type == 'flat':
        time, true_alt, raw_alt = generate_flat_data(
            duration=params['duration'],
            freq=params['freq'],
            altitude=params['altitude'],
            noise_std=params['noise'],
            spike_prob=params['spike_prob'],
            spike_amplitude=params['spike_ampl']
        )
    elif data_type == 'step':
        time, true_alt, raw_alt = generate_step_data(
            duration=params['duration'],
            freq=params['freq'],
            noise_std=params['noise'],
            spike_prob=params['spike_prob'],
            spike_amplitude=params['spike_ampl']
        )
    
    # Сохраняем в файл
    save_data_file(
        filepath=output_path,
        time=time,
        true_alt=true_alt,
        raw_alt=raw_alt,
        format_type=params['format'],
        delimiter=params['sep']
    )
    
    # Краткая статистика
    print(f"\nСтатистика сгенерированных данных:")
    print(f"  Истинная высота:   [{true_alt.min():.2f}, {true_alt.max():.2f}] м")
    print(f"  Сырая высота:      [{raw_alt.min():.2f}, {raw_alt.max():.2f}] м")
    print(f"  СКО шума (оценка): {np.std(raw_alt - true_alt):.3f} м")
    print(f"  Выбросов:          {np.sum(np.abs(raw_alt - true_alt) > 3*params['noise'])} шт.")
    
    print(f"\nДля фильтрации выполните:")
    print(f"  python baro_filter.py \"{output_path}\" {params['freq']}")
    
    print("\n" + "=" * 60)
    print("Готово!")
    print("=" * 60)


if __name__ == "__main__":
    main()