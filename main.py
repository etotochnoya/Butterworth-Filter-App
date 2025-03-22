import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# --- 1. Теоретические расчеты (описаны в коде) ---

# --- 2. Функция расчета АЧХ ---

def butterworth_lpf_freq_response(order, cutoff_frequency, sampling_frequency, num_points=1000):
    """
    Рассчитывает АЧХ фильтра нижних частот Баттерворта.

    Args:
        order (int): Порядок фильтра.
        cutoff_frequency (float): Частота среза (Гц).
        sampling_frequency (float): Частота дискретизации (Гц).
        num_points (int): Количество точек для расчета АЧХ.

    Returns:
        tuple: Кортеж, содержащий:
            - frequencies (numpy.ndarray): Массив частот (Гц).
            - amplitude_db (numpy.ndarray): Массив амплитуд АЧХ в децибелах.
    """

    # Нормализованная частота среза (относительно частоты Найквиста)
    normalized_cutoff = cutoff_frequency / (sampling_frequency / 2)

    # Создание фильтра Баттерворта
    b, a = signal.butter(order, normalized_cutoff, btype='low', analog=False)

    # Расчет АЧХ
    w, h = signal.freqz(b, a, worN=num_points)

    # Преобразование частоты из радианов в Гц
    frequencies = (sampling_frequency * w) / (2 * np.pi)

    # Амплитуда АЧХ в децибелах
    amplitude_db = 20 * np.log10(abs(h))

    return frequencies, amplitude_db


# --- 3. Функция для создания тестовых сигналов ---

def generate_test_signals(sampling_frequency, duration, harmonic_frequencies, noise_amplitude=0):
    """
    Генерирует тестовые сигналы: сумму гармоник и гармонику с шумом.

    Args:
        sampling_frequency (float): Частота дискретизации (Гц).
        duration (float): Длительность сигнала (секунды).
        harmonic_frequencies (list): Список частот гармоник (Гц).
        noise_amplitude (float): Амплитуда случайного шума.

    Returns:
        tuple: Кортеж, содержащий:
            - time (numpy.ndarray): Массив временных отсчетов.
            - sum_of_harmonics (numpy.ndarray): Сигнал, представляющий собой сумму гармоник.
            - harmonic_with_noise (numpy.ndarray): Сигнал, представляющий собой одну гармонику с шумом.
    """

    time = np.arange(0, duration, 1/sampling_frequency)

    sum_of_harmonics = np.zeros_like(time)
    for freq in harmonic_frequencies:
        sum_of_harmonics += np.sin(2 * np.pi * freq * time)

    harmonic_with_noise = np.sin(2 * np.pi * harmonic_frequencies[0] * time) + noise_amplitude * np.random.randn(len(time))

    return time, sum_of_harmonics, harmonic_with_noise

# --- 4. Функция применения фильтра ---

def apply_filter(signal_to_filter, order, cutoff_frequency, sampling_frequency):
    """
    Применяет фильтр Баттерворта к входному сигналу.

    Args:
        signal_to_filter (numpy.ndarray): Входной сигнал.
        order (int): Порядок фильтра.
        cutoff_frequency (float): Частота среза (Гц).
        sampling_frequency (float): Частота дискретизации (Гц).

    Returns:
        numpy.ndarray: Отфильтрованный сигнал.
    """
    normalized_cutoff = cutoff_frequency / (sampling_frequency / 2)
    b, a = signal.butter(order, normalized_cutoff, btype='low', analog=False)
    filtered_signal = signal.filtfilt(b, a, signal_to_filter) # Использование filtfilt для нулевой фазы
    return filtered_signal


# --- 5. Основная часть программы ---

if __name__ == "__main__":
    # Параметры сигнала
    sampling_frequency = 44100  # Гц
    duration = 1  # секунда
    harmonic_frequencies = [500, 2000, 5000]  # Гц
    noise_amplitude = 0.5

    # Генерация тестовых сигналов
    time, sum_of_harmonics, harmonic_with_noise = generate_test_signals(
        sampling_frequency, duration, harmonic_frequencies, noise_amplitude
    )

    # Параметры фильтра (для экспериментов)
    orders = [2, 4, 8]  # Различные порядки фильтра
    cutoff_frequencies = [1000, 3000]  # Различные частоты среза

    # --- 6.  Сравнение АЧХ при различных параметрах ---
    plt.figure(figsize=(12, 6))
    plt.title('Сравнение АЧХ ФНЧ Баттерворта при различных параметрах')
    plt.xlabel('Частота (Гц)')
    plt.ylabel('Амплитуда (дБ)')
    plt.grid(True)
    plt.xscale('log')

    for order in orders:
        for cutoff_frequency in cutoff_frequencies:
            frequencies, amplitude_db = butterworth_lpf_freq_response(
                order, cutoff_frequency, sampling_frequency
            )
            plt.plot(frequencies, amplitude_db, label=f'Порядок: {order}, Частота среза: {cutoff_frequency} Гц')

    plt.legend()
    plt.ylim(-80, 5)  # Задаем пределы по оси Y для лучшей видимости
    plt.show()

    # Комментарии к результатам сравнения АЧХ:
    # * Чем выше порядок фильтра, тем круче спад АЧХ в полосе задерживания. Это означает, что фильтр лучше подавляет частоты, превышающие частоту среза.
    # * Чем выше частота среза, тем больше частот пропускает фильтр.  Частоты ниже частоты среза ослабляются меньше.

    # --- 7. Демонстрация работы фильтра на тестовых сигналах ---

    # Выбираем один набор параметров фильтра для демонстрации
    selected_order = 4
    selected_cutoff_frequency = 2000

    # Применяем фильтр к тестовым сигналам
    filtered_sum_of_harmonics = apply_filter(
        sum_of_harmonics, selected_order, selected_cutoff_frequency, sampling_frequency
    )
    filtered_harmonic_with_noise = apply_filter(
        harmonic_with_noise, selected_order, selected_cutoff_frequency, sampling_frequency
    )

    # --- 8. Визуализация результатов фильтрации ---

    # Спектральный анализ (FFT) для сравнения частотного содержания сигналов
    def plot_spectrum(signal, sampling_frequency, title):
      """Вычисляет и отображает спектр сигнала."""
      yf = np.fft.fft(signal)
      T = 1.0 / sampling_frequency
      xf = np.fft.fftfreq(len(signal), T)[:len(signal)//2]
      plt.plot(xf, 2.0/len(signal) * np.abs(yf[0:len(signal)//2]))
      plt.title(title)
      plt.xlabel("Частота (Гц)")
      plt.ylabel("Амплитуда")
      plt.grid(True)

    plt.figure(figsize=(15, 8))

    plt.subplot(2, 2, 1)
    plt.plot(time, sum_of_harmonics)
    plt.title('Сумма гармоник (исходный сигнал)')
    plt.xlabel('Время (с)')
    plt.ylabel('Амплитуда')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plot_spectrum(sum_of_harmonics, sampling_frequency, 'Спектр суммы гармоник (исходный)')


    plt.subplot(2, 2, 3)
    plt.plot(time, filtered_sum_of_harmonics)
    plt.title(f'Сумма гармоник (отфильтрованный, порядок {selected_order}, частота среза {selected_cutoff_frequency} Гц)')
    plt.xlabel('Время (с)')
    plt.ylabel('Амплитуда')
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plot_spectrum(filtered_sum_of_harmonics, sampling_frequency, f'Спектр суммы гармоник (отфильтрованный, порядок {selected_order}, частота среза {selected_cutoff_frequency} Гц)')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15, 8))

    plt.subplot(2, 2, 1)
    plt.plot(time, harmonic_with_noise)
    plt.title('Гармоника с шумом (исходный сигнал)')
    plt.xlabel('Время (с)')
    plt.ylabel('Амплитуда')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plot_spectrum(harmonic_with_noise, sampling_frequency, 'Спектр гармоники с шумом (исходный)')

    plt.subplot(2, 2, 3)
    plt.plot(time, filtered_harmonic_with_noise)
    plt.title(f'Гармоника с шумом (отфильтрованный, порядок {selected_order}, частота среза {selected_cutoff_frequency} Гц)')
    plt.xlabel('Время (с)')
    plt.ylabel('Амплитуда')
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plot_spectrum(filtered_harmonic_with_noise, sampling_frequency, f'Спектр гармоники с шумом (отфильтрованный, порядок {selected_order}, частота среза {selected_cutoff_frequency} Гц)')
    plt.tight_layout()
    plt.show()
