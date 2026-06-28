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
    print("ГЕНЕРАТОР ДАННЫХ БАРОМЕТРА ДЛЯ ДРОНА (1500 м)")
    print("=" * 60)
    
    while True:
        print("\nКуда сохранить сгенерированные файлы?")
        print("  • Можно ввести полный путь: C:\\Users\\79182\\Desktop\\hack")
        print("  • Можно ввести относительно текущей папки: data или ./output")
        print("  • Можно нажать Enter — сохранит в текущую папку")
        
        user_input = input("\nПуть к папке: ").strip()
        
        if not user_input:
            output_dir = os.getcwd()
            print(f"Выбрана текущая папка: {output_dir}")
        else:
            output_dir = user_input
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"✓ Папка готова: {output_dir}")
            return output_dir
        except Exception as e:
            print(f"✗ Ошибка: не удалось создать папку '{output_dir}'")
            print(f"  {e}")
            print("  Попробуйте другой путь.")


def generate_drone_cruise_data(
    duration: float = 120.0,
    freq: float = 10.0,
    base_altitude: float = 1500.0,
    noise_std: float = 1.5,
    spike_prob: float = 0.01,
    spike_amplitude: float = 8.0,
    wind_amplitude: float = 5.0,
    drift_rate: float = 0.02,
    seed: Optional[int] = None
) -> tuple:
    """
    Генератор данных барометра для дрона в режиме зависания на 1500 м.
    
    Особенности:
    - Базовая высота 1500 м с естественными колебаниями от ветра
    - Повышенный уровень шума (на высоте барометр менее точен)
    - Редкие выбросы от порывов ветра
    - Медленный температурный дрейф
    - Реалистичная вибрация от пропеллеров
    
    Args:
        duration: Длительность симуляции [с].
        freq: Частота опроса датчика [Гц].
        base_altitude: Базовая высота полета [м] (по умолчанию 1500).
        noise_std: СКО шума измерений [м].
        spike_prob: Вероятность импульсного выброса.
        spike_amplitude: Амплитуда выбросов [м].
        wind_amplitude: Амплитуда ветровых колебаний [м].
        drift_rate: Скорость температурного дрейфа [м/с].
        seed: Зерно для воспроизводимости.
    
    Returns:
        time: Массив времени [с].
        true_alt: Истинная высота [м].
        raw_alt: Сырые данные с датчика [м].
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_samples = int(duration * freq)
    time = np.arange(n_samples) / freq
    
    # === Истинная высота: 1500 м с естественными колебаниями ===
    
    # Базовая высота
    true_alt = np.full(n_samples, base_altitude)
    
    # Ветровые колебания (несколько частот для реалистичности)
    wind = (
        wind_amplitude * 0.5 * np.sin(2 * np.pi * 0.05 * time) +      # Очень медленные волны
        wind_amplitude * 0.3 * np.sin(2 * np.pi * 0.15 * time + 1.2) + # Средние колебания
        wind_amplitude * 0.2 * np.cos(2 * np.pi * 0.3 * time + 0.7)    # Быстрые порывы
    )
    
    # Случайные микро-коррекции высоты (система удержания дрона)
    # Имитация работы PID-регулятора: небольшие перелеты и коррекции
    correction = np.zeros(n_samples)
    correction[0] = 0
    for i in range(1, n_samples):
        # Авторегрессионный процесс 1-го порядка (медленный возврат к базовой высоте)
        correction[i] = correction[i-1] * 0.95 + np.random.normal(0, 0.3)
        # Ограничиваем коррекцию
        correction[i] = np.clip(correction[i], -3.0, 3.0)
    
    true_alt = base_altitude + wind + correction
    
    # === Добавляем шум и артефакты ===
    
    # 1. Белый гауссов шум (на высоте больше из-за турбулентности)
    noise = np.random.normal(0, noise_std, n_samples)
    
    # 2. Импульсные выбросы (порывы ветра, помехи)
    spikes_mask = np.random.rand(n_samples) < spike_prob
    spike_values = np.random.exponential(spike_amplitude / 2, n_samples)
    spike_signs = np.random.choice([-1, 1], n_samples)
    spikes = np.where(spikes_mask, spike_values * spike_signs, 0)
    
    # 3. Температурный дрейф барометра (медленное изменение показаний)
    drift = drift_rate * np.sin(2 * np.pi * 0.001 * time) * np.random.normal(1, 0.3)
    
    # 4. Высокочастотная вибрация от пропеллеров
    # Частота вращения ~150 Гц, но алиасинг дает низкочастотные биения
    vibration = (
        0.5 * np.sin(2 * np.pi * 2.3 * time) * np.random.normal(1, 0.3, n_samples) +
        0.3 * np.sin(2 * np.pi * 5.7 * time) * np.random.normal(1, 0.2, n_samples)
    )
    
    # 5. Случайные скачки давления (микропорывы)
    pressure_jumps = np.zeros(n_samples)
    jump_positions = np.random.choice(n_samples, size=int(n_samples * 0.005), replace=False)
    pressure_jumps[jump_positions] = np.random.normal(0, 2.0, len(jump_positions))
    
    # Собираем сырой сигнал
    raw_alt = true_alt + noise + spikes + drift + vibration + pressure_jumps
    
    return time, true_alt, raw_alt


def generate_drone_flight_data(
    duration: float = 180.0,
    freq: float = 10.0,
    seed: Optional[int] = None
) -> tuple:
    """
    Генератор полного полета дрона: взлет на 1500 м, зависание, посадка.
    
    Args:
        duration: Длительность [с].
        freq: Частота [Гц].
        seed: Зерно для воспроизводимости.
    
    Returns:
        time, true_alt, raw_alt
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_samples = int(duration * freq)
    time = np.arange(n_samples) / freq
    
    # Фазы полета
    # 1. Взлет до 1500 м (0-15% времени)
    # 2. Зависание на 1500 м (15-85% времени)
    # 3. Посадка (85-100% времени)
    
    phase1_end = int(n_samples * 0.15)
    phase2_end = int(n_samples * 0.85)
    
    true_alt = np.zeros(n_samples)
    
    # Фаза 1: Взлет (экспоненциальный набор высоты)
    phase1_t = time[:phase1_end] / time[phase1_end]
    true_alt[:phase1_end] = 1500 * (1 - np.exp(-5 * phase1_t)) / (1 - np.exp(-5))
    
    # Фаза 2: Зависание с колебаниями
    phase2_t = time[phase1_end:phase2_end] - time[phase1_end]
    wind = (
        5.0 * np.sin(2 * np.pi * 0.03 * phase2_t) +
        3.0 * np.cos(2 * np.pi * 0.12 * phase2_t) +
        2.0 * np.sin(2 * np.pi * 0.25 * phase2_t)
    )
    true_alt[phase1_end:phase2_end] = 1500 + wind
    
    # Фаза 3: Посадка
    phase3_t = time[phase2_end:] - time[phase2_end]
    phase3_duration = time[-1] - time[phase2_end]
    true_alt[phase2_end:] = 1500 * np.exp(-4 * phase3_t / phase3_duration)
    
    # Добавляем шум
    noise_std = 1.5
    spike_prob = 0.01
    spike_amplitude = 8.0
    
    noise = np.random.normal(0, noise_std, n_samples)
    spikes_mask = np.random.rand(n_samples) < spike_prob
    spike_values = np.random.exponential(spike_amplitude / 2, n_samples)
    spike_signs = np.random.choice([-1, 1], n_samples)
    spikes = np.where(spikes_mask, spike_values * spike_signs, 0)
    vibration = 0.5 * np.sin(2 * np.pi * 2.3 * time) * np.random.normal(1, 0.3, n_samples)
    
    raw_alt = true_alt + noise + spikes + vibration
    
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
            f.write(f"# Generated drone barometer data\n")
            f.write(f"# Altitude: ~1500 m\n")
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
    print(f"  Формат: тип {format_type}")
    print(f"  Разделитель: '{delimiter}'")


def choose_scenario():
    """Выбор сценария полета."""
    print("\n" + "-" * 60)
    print("Выберите сценарий полета дрона:")
    print("  1. cruise — Зависание на 1500 м (дрон держит высоту)")
    print("  2. flight — Полный полет (взлет на 1500, зависание, посадка)")
    print("-" * 60)
    
    while True:
        choice = input("Введите номер (1-2): ").strip()
        if choice == '1':
            return 'cruise'
        elif choice == '2':
            return 'flight'
        else:
            print("✗ Введите 1 или 2.")


def choose_parameters(scenario: str) -> dict:
    """Запрашивает параметры генерации."""
    
    if scenario == 'cruise':
        params = {
            'duration': 120.0,
            'freq': 10.0,
            'noise_std': 1.5,
            'spike_prob': 0.01,
            'spike_ampl': 8.0,
            'wind_ampl': 5.0,
            'drift_rate': 0.02,
            'seed': None,
            'format': 3,
            'sep': ','
        }
    else:
        params = {
            'duration': 180.0,
            'freq': 10.0,
            'seed': None,
            'format': 3,
            'sep': ','
        }
    
    print("\n" + "-" * 60)
    print("Настройка параметров (нажмите Enter для значений по умолчанию)")
    print("-" * 60)
    
    # Длительность
    while True:
        user_input = input(f"Длительность записи [с] (по умолчанию {params['duration']}): ").strip()
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
    
    # Частота опроса
    while True:
        user_input = input(f"Частота опроса датчика [Гц] (по умолчанию {params['freq']}): ").strip()
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
    
    # Для cruise - дополнительные параметры
    if scenario == 'cruise':
        while True:
            user_input = input(f"СКО шума [м] (по умолчанию {params['noise_std']}): ").strip()
            if not user_input:
                break
            try:
                params['noise_std'] = float(user_input)
                if params['noise_std'] < 0:
                    print("✗ СКО не может быть отрицательным.")
                    continue
                break
            except ValueError:
                print("✗ Введите число.")
        
        while True:
            user_input = input(f"Амплитуда ветровых колебаний [м] (по умолчанию {params['wind_ampl']}): ").strip()
            if not user_input:
                break
            try:
                params['wind_ampl'] = float(user_input)
                if params['wind_ampl'] < 0:
                    print("✗ Амплитуда не может быть отрицательной.")
                    continue
                break
            except ValueError:
                print("✗ Введите число.")
    
    # Зерно
    user_input = input(f"Зерно ГПСЧ для воспроизводимости (Enter = случайно): ").strip()
    if user_input:
        try:
            params['seed'] = int(user_input)
        except ValueError:
            print("✗ Введите целое число. Будет использовано случайное зерно.")
    
    # Формат
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
    """Основная функция."""
    
    # Шаг 1: Путь сохранения
    output_dir = get_output_dir()
    
    # Шаг 2: Сценарий
    scenario = choose_scenario()
    
    # Шаг 3: Параметры
    params = choose_parameters(scenario)
    
    # Имя файла
    if params['seed'] is not None:
        filename = f"drone_{scenario}_1500m_{int(params['duration'])}s_{int(params['freq'])}hz_seed{params['seed']}.txt"
    else:
        filename = f"drone_{scenario}_1500m_{int(params['duration'])}s_{int(params['freq'])}hz.txt"
    
    output_path = os.path.join(output_dir, filename)
    
    print("\n" + "=" * 60)
    print("ГЕНЕРАЦИЯ ДАННЫХ ДРОНА")
    print("=" * 60)
    print(f"Сценарий:      {scenario}")
    print(f"Высота:        ~1500 м")
    print(f"Длительность:  {params['duration']} с")
    print(f"Частота:       {params['freq']} Гц")
    if scenario == 'cruise':
        print(f"СКО шума:      {params['noise_std']} м")
        print(f"Ветровые колеб: ±{params['wind_ampl']} м")
    if params['seed'] is not None:
        print(f"Seed:          {params['seed']}")
    print(f"Файл:          {output_path}")
    print("-" * 60)
    
    print("Генерация данных...")
    
    if scenario == 'cruise':
        time, true_alt, raw_alt = generate_drone_cruise_data(
            duration=params['duration'],
            freq=params['freq'],
            noise_std=params.get('noise_std', 1.5),
            spike_prob=params.get('spike_prob', 0.01),
            spike_amplitude=params.get('spike_ampl', 8.0),
            wind_amplitude=params.get('wind_ampl', 5.0),
            drift_rate=params.get('drift_rate', 0.02),
            seed=params['seed']
        )
    else:
        time, true_alt, raw_alt = generate_drone_flight_data(
            duration=params['duration'],
            freq=params['freq'],
            seed=params['seed']
        )
    
    # Сохранение
    save_data_file(
        filepath=output_path,
        time=time,
        true_alt=true_alt,
        raw_alt=raw_alt,
        format_type=params['format'],
        delimiter=params['sep']
    )
    
    # Статистика
    print(f"\nСтатистика данных:")
    print(f"  Истинная высота:   [{true_alt.min():.1f}, {true_alt.max():.1f}] м")
    print(f"  Сырая высота:      [{raw_alt.min():.1f}, {raw_alt.max():.1f}] м")
    print(f"  Средняя высота:    {np.mean(true_alt):.1f} м")
    print(f"  СКО шума (оценка): {np.std(raw_alt - true_alt):.2f} м")
    
    print(f"\nДля фильтрации выполните:")
    print(f"  python baro_filter.py")
    print(f"  и укажите путь: {output_path}")
    
    print("\n" + "=" * 60)
    print("Готово!")
    print("=" * 60)


if __name__ == "__main__":
    main()