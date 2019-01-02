"""                            Лабораторна работа номер 7
                          з курсу Чисельні методи, варіант 6
    Завдання: Побудувати кубічний сплайн з глобальним заданням нахилів і
    розвязуванням СЛАР розмірністю n-2 віносно других похідних Si''(xi)
    для функції f(x), яку задано таблично.
      Для оцінювання похибки значень функція  f(x) задана.
      f(x) = x^2 + sin(x)
    Виконав студент 2 курсу: Зуєв Михайло Олександрович
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.multiarray import ndarray


def my_function(x):
    return x * x + math.sin(x)


def spline(x_data, y_data, n):
    """
        Строит полином вида Si(x) = ai + bi(x - xi) + ci(x - xi)^2 + di(x - xi)^3,
    для таблично заданной функции.
    :param x_data: Таблица аргументов функции
    :param y_data: Таблица значений функции
    :param n: количество точек
    :return: массивы коэффициентов ai, bi, ci, di
    """
    # Массивы коэффициентов
    a: ndarray = np.array([y_data[i] for i in range(n - 1)], dtype=np.float64)
    b: ndarray = np.zeros(n - 1, dtype=np.float64)
    c: ndarray = np.zeros(n - 1, dtype=np.float64)
    d: ndarray = np.zeros(n - 1, dtype=np.float64)

    # Значенния h: h[i] = x[i + 1] - x[i]
    h = np.array([x_data[i + 1] - x_data[i] for i in range(n - 1)], dtype=np.float64)

    #                      y[i + 2] - y[i + 1]    y[i + 1] - y[i]
    # Значения g: g[i] = ( ------------------- -  --------------- )
    #                           h[i + 1]                h[i]
    g = np.array([(((y_data[i + 2] - y_data[i + 1]) / h[i + 1]) - ((y_data[i + 1] - y_data[i]) / h[i])) \
                  * 3 for i in range(n - 2)], dtype=np.float64)

    # Формируем СЛАР t*c = g
    # ┍ -----------------------------------------------------------------┑  ┍--------┑
    # | 2(h1 + h2)       h2                                              | |   c2   |
    # |     h2       2(h2 + h3)       h3                                 | |   c3   |
    # |                             ......                               | |   ..   |
    # |                             h[n-2]       2(h[n - 2) + h[n - 1])  | | c[n-1] |
    # ┕ -----------------------------------------------------------------┙  ┕--------┙
    t = np.zeros((n - 2, n - 2), dtype=np.float64)

    t[0, 0] = (h[0] + h[1]) * 2
    t[0, 1] = h[1]
    for i in range(1, n - 3):
        t[i, i - 1] = h[i]
        t[i, i] = (h[i] + h[i + 1]) * 2
        t[i, i + 1] = h[i + 1]
    t[n - 3, n - 4] = h[n - 3]
    t[n - 3, n - 3] = (h[n - 3] + h[n - 2]) * 2

    # Решаем СЛАР и находим сi, i >= 2 (c1 = 0)
    c[1:] = np.linalg.solve(t, g)

    # Вычисляем d1, ..., d[n - 1] и b1, ..., b[n - 1]
    for i in range(n - 2):
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])
        b[i] = ((y_data[i + 1] - y_data[i]) / h[i]) - (((c[i + 1] + 2 * c[i]) * h[i]) / 3.0)
    d[n - 2] = (-c[n - 2]) / (3 * h[n - 2])
    b[n - 2] = ((y_data[n - 1] - y_data[n - 2]) / h[n - 2]) - (2.0 / 3.0) * c[n - 2] * h[n - 2]

    return a, b, c, d


def find_index(x_data, x):
    '''
       Находит индекс i такой что x є [x_data[i]; x_data[i+1]]
    :param x_data: таблица аргументов функции
    :param x: арумент для которого ищется индекс
    :return: нужный индекс
    '''
    # Определяем i
    global i
    x_size = len(x_data)
    flag = True
    for j in range(x_size - 2):
        if x_data[j] <= x <= x_data[j + 1]:
            i = j
            flag = False
            break
    if flag:
        i = 0 if x < x_data[0] else x_size - 2
    return i


def calculate_spline(a, b, c, d, x_data, x):
    """
       Считает значение полинома вида Si(x) = ai + bi(x - xi) + ci(x - xi)^2 + di(x - xi)^3
    в точке x
    :param a: массив коэффициентов a полинома
    :param b: массив коэффициентов b полинома
    :param c: массив коэффициентов c полинома
    :param d: массив коэффициентов d полинома
    :param x_data: таблица аргументов функции
    :param x: аргумент полинома
    :return: значение полинома Si(x)
    """
    # Определяем i
    i = find_index(x_data, x)

    # Считаем значение Si(x)
    tmp = x - x_data[i]
    tmp_2 = tmp ** 2
    tmp_3 = tmp_2 * tmp
    return a[i] + b[i] * tmp + c[i] * tmp_2 + d[i] * tmp_3


def calculate_derivative(x, n, x_data, d=[], c=[], b=[], a=[], pos='right'):
    """
       Вычисляет значение производной полинома вида Si(x) = ai + bi(x - xi) + ci(x - xi)^2 + di(x - xi)^3
    :param x: аргумент полинома
    :param n: степень производной
    :param x_data: агрументы функции
    :param pos: 'left' - что бы посчитать производную слева
    :param a: массив коэффициентов a полинома
    :param b: массив коэффициентов b полинома
    :param c: массив коэффициентов c полинома
    :param d: массив коэффициентов d полинома
    :return: значение производной n-й степени Si(x)
    """
    if n == 0:
        return calculate_spline(a, b, c, d, x_data, x)

    i = find_index(x_data, x)
    if pos == 'left':
        i = i - 1 if i > 0 else 0
    if n == 1:
        tmp = x - x_data[i]
        return b[i] + 2 * c[i] * tmp + 3 * d[i] * tmp * tmp
    elif n == 2:
        tmp = x - x_data[i]
        return 2 * c[i] + 6 * d[i] * tmp
    elif n == 3:
        return 6 * d[i]
    return 0.0


# Границы функции
left = 1
right = 3

# Количество точек
m = 6

# Шаг
H = (right - left) / (m - 1)

# Таблица значений функций
xData = np.zeros(m)
yData = np.zeros(m)

# Создание таблицы
for i in range(m):
    xData[i] = left + i * H
    yData[i] = my_function(xData[i])

# Получаем коэффициенты кубического сплайна
a, b, c, d = spline(xData, yData, m)

# Рисуем график и выводим таблицу
h1 = (right - left) / 20
H2 = 2 * H
start = left - H2
end = right + H2
xlist = []
ylist1 = []
ylist2 = []
print("----------------------------------------------------------------------")
print("|      |          |          |                | f(xj) - Si(xj)       |")
print("|  xj  |   f(xj)  |  Si(xj)  | f(xj) - Si(xj) | -------------- * 100 |")
print("|      |          |          |                |      Si(xj)          |")
print("----------------------------------------------------------------------")
while start <= end:
    f = my_function(start)
    s = calculate_spline(a, b, c, d, xData, start)
    xlist.append(start)
    ylist1.append(f)
    ylist2.append(s)
    # Выводим таблицу
    print("|{0:5.2f} | {1:8.3f} | {2:8.3f} | {3:14.9f} | {4:21.16f}|".format(start, f, s, s - f, (s - f) * 100 / f))
    start += h1
print("----------------------------------------------------------------------")

plt.plot(xlist, ylist1, 'k', label='f(x)')
plt.plot(xlist, ylist2, '--', label='S(x)')

plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'$y = f(x), y = S(x)$')
plt.legend()
plt.show()

# Дополнительное задание
eps = 1e-10
print("----------------------------------------------------------------------------")
print("|      |     |             |             | S^(a)(xj+0) - S^(a)(xj-0)       |")
print("|  xj  |  a  | S^(a)(xj-0) | S^(a)(xj+0) | ------------------------- * 100 |")
print("|      |     |             |             |        S^(a)(xj-0)              |")
print("----------------------------------------------------------------------------")
for j in range(1, m - 1):
    x = xData[j]
    for t in range(4):
        l = calculate_derivative(x + eps, t, xData, d, c, b, a, pos='left')
        r = calculate_derivative(x + eps, t, xData, d, c, b, a, pos='right')
        if l != 0:
            tmp = (r - l) * 100 / l
            print("|{0:5.2f} |{1:4d} |{2:12.7f} |{3:12.7f} |{4:33.27f}|".format(x, t, l, r, tmp))
        else:
            print("|{0:5.2f} |{1:4d} |{2:12.7f} |{3:12.7f} |    --------------------------   |".format(x, t, l, r))
print("----------------------------------------------------------------------------")
