import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

class BaroAltitudeFilter:
    """
    Каскадный фильтр высоты: Медиана (3) + Калман (1D).
    Полностью автономен, не требует внешних библиотек кроме numpy (опционально для тестов).
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
        # Циклический буфер: заменяем самое старое значение новым
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
        """Сброс фильтра на новую высоту (например, при посадке)."""
        self._buf = [altitude] * 3
        self._x_hat = altitude
        self._p = 1.0


def generate_baro_data(duration: float = 60.0, freq: float = 50.0, 
                       noise_std: float = 0.3, spike_prob: float = 0.02, 
                       spike_amplitude: float = 5.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Генератор реалистичных данных барометра для тестирования.
    
    Args:
        duration: Длительность симуляции [с].
        freq: Частота опроса датчика [Гц].
        noise_std: СКО шума измерений [м].
        spike_prob: Вероятность импульсного выброса на одном отсчете.
        spike_amplitude: Амплитуда выбросов [м].
    
    Returns:
        time: Массив времени [с].
        true_alt: Истинная высота (без шума) [м].
        raw_alt: Сырые данные с датчика (с шумом и выбросами) [м].
    """
    n_samples = int(duration * freq)
    time = np.linspace(0, duration, n_samples)
    
    # Истинная высота: полет по сложной траектории
    true_alt = np.zeros(n_samples)
    # Взлет на 10 м за 5 секунд
    idx_climb = int(5 * freq)
    true_alt[:idx_climb] = np.linspace(0, 10, idx_climb)
    # Полет на 10 м с плавными колебаниями
    t_mid = time[idx_climb:]
    true_alt[idx_climb:] = 10.0 + 2.0 * np.sin(2 * np.pi * 0.2 * t_mid) + \
                           1.0 * np.cos(2 * np.pi * 0.5 * t_mid)
    
    # Добавляем белый гауссов шум
    raw_alt = true_alt + np.random.normal(0, noise_std, n_samples)
    
    # Добавляем импульсные выбросы (случайные скачки)
    spikes = np.random.rand(n_samples) < spike_prob
    spike_values = np.random.normal(0, spike_amplitude, n_samples)
    raw_alt[spikes] += spike_values[spikes]
    
    return time, true_alt, raw_alt


def test_filter():
    """Демонстрация работы фильтра с визуализацией."""
    
    # Генерируем тестовые данные (60 секунд, 50 Гц)
    time, true_alt, raw_alt = generate_baro_data(
        duration=60.0, 
        freq=50.0,
        noise_std=0.3,
        spike_prob=0.02,
        spike_amplitude=5.0
    )
    
    # Инициализируем фильтр
    baro_filter = BaroAltitudeFilter(start_altitude=raw_alt[0], q=0.05, r=1.2)
    
    # Прогоняем все данные через фильтр
    filtered_alt = np.zeros_like(raw_alt)
    for i in range(len(raw_alt)):
        filtered_alt[i] = baro_filter.update(raw_alt[i])
    
    # Анализ результатов
    error_raw = np.std(raw_alt - true_alt)
    error_filtered = np.std(filtered_alt - true_alt)
    improvement = (1 - error_filtered / error_raw) * 100
    
    print(f"=== Результаты фильтрации ===")
    print(f"СКО сырых данных:    {error_raw:.3f} м")
    print(f"СКО фильтрованных:   {error_filtered:.3f} м")
    print(f"Улучшение точности:  {improvement:.1f}%")
    
    # Визуализация
    plt.figure(figsize=(12, 10))
    
    # Полный сигнал
    plt.subplot(3, 1, 1)
    plt.plot(time, raw_alt, 'gray', alpha=0.6, linewidth=0.5, label='Сырые данные')
    plt.plot(time, true_alt, 'b-', linewidth=2, label='Истинная высота')
    plt.plot(time, filtered_alt, 'r-', linewidth=1.5, label='Отфильтрованная')
    plt.ylabel('Высота [м]')
    plt.title('Сравнение сигналов (полный интервал)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Увеличенный фрагмент: взлет (первые 10 секунд)
    plt.subplot(3, 1, 2)
    idx_zoom1 = int(10 * 50)  # 50 Гц
    plt.plot(time[:idx_zoom1], raw_alt[:idx_zoom1], 'gray', alpha=0.5, linewidth=0.8, label='Сырые')
    plt.plot(time[:idx_zoom1], true_alt[:idx_zoom1], 'b-', linewidth=2, label='Истинная')
    plt.plot(time[:idx_zoom1], filtered_alt[:idx_zoom1], 'r-', linewidth=1.5, label='Фильтр')
    plt.ylabel('Высота [м]')
    plt.title('Взлет (первые 10 секунд)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Увеличенный фрагмент: участок с сильным шумом
    plt.subplot(3, 1, 3)
    idx_zoom2_start = int(30 * 50)
    idx_zoom2_end = int(35 * 50)
    plt.plot(time[idx_zoom2_start:idx_zoom2_end], 
             raw_alt[idx_zoom2_start:idx_zoom2_end], 'gray', alpha=0.5, linewidth=0.8, label='Сырые')
    plt.plot(time[idx_zoom2_start:idx_zoom2_end], 
             true_alt[idx_zoom2_start:idx_zoom2_end], 'b-', linewidth=2, label='Истинная')
    plt.plot(time[idx_zoom2_start:idx_zoom2_end], 
             filtered_alt[idx_zoom2_start:idx_zoom2_end], 'r-', linewidth=1.5, label='Фильтр')
    plt.xlabel('Время [с]')
    plt.ylabel('Высота [м]')
    plt.title('Установившийся режим (30-35 секунд)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return baro_filter


# Пример использования в реальном проекте
def example_usage():
    """Минимальный пример встраивания в ваш код."""
    # Создаем фильтр один раз
    alt_filter = BaroAltitudeFilter(start_altitude=100.0, q=0.05, r=1.2)
    
    # В цикле опроса датчика:
    # raw = read_barometer()  # Ваша функция чтения
    raw = 100.3  # Пример сырого значения
    filtered = alt_filter.update(raw)
    print(f"Сырое: {raw:.2f} м, Фильтрованное: {filtered:.2f} м")


if __name__ == "__main__":
    # Запуск теста с визуализацией
    test_filter()
    
    # Простой пример использования
    print("\n" + "="*50)
    example_usage()