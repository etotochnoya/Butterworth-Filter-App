hello its our processing digital signal app and we ve made a project about Butterworth filters and some any mathematics about it.


Разъяснения к коду и теоретическим расчетам:

Теоретические расчеты: 
Передаточная функция: H(s) = 1 / (1 + (s/ωc)^(2n))^(1/2) , где s = jω.
АЧХ (амплитуда): |H(jω)| = 1 / (1 + (ω/ωc)^(2n))^(1/2).
АЧХ в дБ: H(f)dB = 20 * log10(|H(jω)|) = -10 * log10(1 + (f/fc)^(2n)).
Эти формулы определяют, как амплитуда сигнала изменяется в зависимости от частоты после прохождения через фильтр Баттерворта. Чем больше отношение частоты сигнала к частоте среза, тем больше затухание сигнала. Порядок фильтра определяет скорость этого затухания.

butterworth_lpf_freq_response Функция:

Принимает параметры фильтра: порядок (order), частоту среза (cutoff_frequency) и частоту дискретизации (sampling_frequency).
Вычисляет АЧХ, используя функции signal.butter (для создания фильтра) и signal.freqz (для расчета АЧХ) из библиотеки SciPy. signal.freqz эффективно вычисляет передаточную функцию в дискретном времени для набора частот.
Возвращает частоты и соответствующие значения АЧХ в децибелах.
generate_test_signals Функция:

Создает два типа тестовых сигналов:
Сумма гармоник: Складывает несколько синусоидальных сигналов с разными частотами. Это позволяет проверить, как фильтр ослабляет различные частотные компоненты.
Гармоника с шумом: Добавляет случайный шум к синусоидальному сигналу. Это позволяет оценить способность фильтра подавлять шум.
Возвращает временной ряд и сгенерированные сигналы.
apply_filter Функция:

Применяет фильтр Баттерворта к входному сигналу, используя функцию signal.filtfilt из SciPy.
Важно: signal.filtfilt используется вместо signal.lfilter. filtfilt применяет фильтр в прямом и обратном направлениях, что обеспечивает нулевую фазовую задержку. Это важно для сохранения формы сигнала при фильтрации.
Основная часть программы (if __name__ == "__main__":)

Задает параметры сигнала (частоту дискретизации, длительность, частоты гармоник, амплитуду шума).
Генерирует тестовые сигналы.
Задает различные параметры фильтра (порядок, частоту среза) для сравнения.
Рассчитывает и строит графики АЧХ для различных параметров фильтра. Графики позволяют визуально сравнить, как меняется АЧХ при изменении порядка и частоты среза.
Применяет фильтр к тестовым сигналам с выбранными параметрами.
Строит графики исходных и отфильтрованных сигналов во временной области.
Вычисляет и отображает спектры сигналов (с помощью FFT), что позволяет проанализировать изменения в частотном содержании сигналов после фильтрации.
Сравнение АЧХ при различных параметрах:

Графики АЧХ демонстрируют влияние порядка фильтра и частоты среза.
Порядок фильтра: Чем выше порядок, тем круче спад АЧХ в полосе задерживания. Это означает, что фильтр более эффективно подавляет частоты выше частоты среза.
Частота среза: Определяет частоту, выше которой сигнал начинает ослабляться. Более высокая частота среза пропускает больше высоких частот.
Демонстрация работы фильтра на тестовых сигналах:

Графики показывают, как фильтр изменяет форму сигналов во временной области.
Анализ спектра (FFT) показывает, какие частотные компоненты сигнала были ослаблены фильтром.
Как использовать код:

Установите необходимые библиотеки:
pip install numpy matplotlib scipy

bash
Запустите Python-скрипт.
Анализируйте графики:
Сравните АЧХ для разных параметров фильтра.
Посмотрите, как фильтр влияет на форму тестовых сигналов во временной области.
Изучите спектры сигналов, чтобы понять, какие частотные компоненты были ослаблены.
Поэкспериментируйте с параметрами: Изменяйте порядок фильтра, частоту среза, частоты гармоник, уровень шума, чтобы увидеть, как это влияет на результаты.
Ключевые моменты, на которые стоит обратить внимание при анализе результатов:

Крутизна спада АЧХ: Определяется порядком фильтра. Более высокий порядок обеспечивает более резкий переход между полосой пропускания и полосой задерживания.
Подавление нежелательных частот: Фильтр эффективно подавляет частоты выше частоты среза, но подавление не идеально. Всегда будет некоторое остаточное количество этих частот.
Фазовая задержка: Использование signal.filtfilt минимизирует фазовую задержку, что важно для сохранения формы сигнала.
Выбор параметров фильтра: Зависит от конкретной задачи. Нужно найти компромисс между желаемым уровнем подавления нежелательных частот и допустимыми искажениями полезного сигнала.
Этот код предоставляет основу для изучения свойств фильтра нижних частот Баттерворта. Вы можете расширить его, добавив другие типы фильтров (например, фильтры верхних частот, полосовые фильтры), более сложные тестовые сигналы, или дополнительные метрики для оценки производительности фильтра.