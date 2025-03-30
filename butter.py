import numpy as np
import matplotlib.pyplot as plt

def butterworth_lpf(f, fc, n):
    return 1 / np.sqrt(1 + (f / fc) ** (2 * n))

# Диапазон частот
f = np.logspace(1, 5, 1000)  # от 10 Гц до 100 кГц

# Сравнение при разных порядках (n)
fc = 1000  # Фиксированная частота среза
orders = [2, 4, 6]  # Разные порядки

plt.figure(figsize=(12, 6))
for n in orders:
    H = butterworth_lpf(f, fc, n)
    H_db = 20 * np.log10(H)
    plt.semilogx(f, H_db, label=f'n={n}')

plt.title('АЧХ ФНЧ Баттерворта при разных порядках (fc=1000 Гц)')
plt.xlabel('Частота (Гц)')
plt.ylabel('АЧХ (дБ)')
plt.grid(which='both', linestyle='--')
plt.axvline(fc, color='red', linestyle='--', label='f_c = 1000 Гц')
plt.axhline(-3, color='green', linestyle='--', label='-3 дБ')
plt.legend()
plt.show()

# Сравнение при разных частотах среза (fc)
n = 4  # Фиксированный порядок
cutoffs = [500, 1000, 3000]  # Разные частоты среза

plt.figure(figsize=(12, 6))
for fc in cutoffs:
    H = butterworth_lpf(f, fc, n)
    H_db = 20 * np.log10(H)
    plt.semilogx(f, H_db, label=f'fc={fc} Гц')

plt.title('АЧХ ФНЧ Баттерворта при разных частотах среза (n=4)')
plt.xlabel('Частота (Гц)')
plt.ylabel('АЧХ (дБ)')
plt.grid(which='both', linestyle='--')
plt.axhline(-3, color='green', linestyle='--', label='-3 дБ')
plt.legend()
plt.show()