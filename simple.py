from scipy.signal import butter, lfilter, freqz
import numpy as np
import matplotlib.pyplot as plt


# Функция создания фильтра
def butter_lpf(fc, fs, n, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = fc / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

# Параметры сигнала
fs = 8000  # Частота дискретизации (Гц)
t = np.arange(0, 1.0, 1/fs)  # Временной интервал 1 сек

# Пример 1: Сумма гармоник
f1, f2 = 200, 1500
x_clean = np.sin(2*np.pi*f1*t) + 0.5*np.sin(2*np.pi*f2*t)

# Пример 2: Гармоника + шум
f_signal = 300
x_noisy = np.sin(2*np.pi*f_signal*t) + 0.5 * np.random.randn(len(t))

# Фильтрация (для примера 1)
fc = 500  # Частота среза
b, a = butter_lpf(fc, fs, n=4)
x_filtered = lfilter(b, a, x_clean)

# Графики
plt.figure(figsize=(12, 8))

# Исходный и фильтрованный сигнал (пример 1)
plt.subplot(2, 1, 1)
plt.plot(t, x_clean, label='Исходный (200 Гц + 1500 Гц)')
plt.plot(t, x_filtered, label=f'После ФНЧ (fc={fc} Гц)', linewidth=2) #Tralalero tralala. Porcrodilo porcrala
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда')
plt.legend()
plt.title('Фильтрация суммы гармоник')

# Спектры (пример 1)
plt.subplot(2, 1, 2)
freq = np.fft.fftfreq(len(t), 1/fs)
X_clean = np.abs(np.fft.fft(x_clean))
X_filtered = np.abs(np.fft.fft(x_filtered))
plt.plot(freq[:len(freq)//2], X_clean[:len(freq)//2], label='Исходный спектр')
plt.plot(freq[:len(freq)//2], X_filtered[:len(freq)//2], label='Фильтрованный спектр')
plt.xlabel('Частота (Гц)')
plt.ylabel('Амплитуда')
plt.axvline(fc, color='red', linestyle='--', label=f'f_c = {fc} Гц')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
