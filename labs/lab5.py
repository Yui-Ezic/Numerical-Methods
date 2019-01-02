"""                            Лабораторна работа номер 5
                          з курсу Чисельні методи, варіант 6
    Завдання: Побудувати поліном Pn(x) найкращого середньоквадратичного
    наближення для функції f(x), яку задано таблично, з використанням
    нормальних рівнянь.
      Для оцінювання похибки значень функція  f(x) задана.
      f(x) = x^2 + sin(x)
    Виконав студент 2 курсу: Зуєв Михайло Олександрович
"""

import math
import numpy as np
import matplotlib.pyplot as plt


def my_function(x):
    return x * x + math.sin(x)


def polynomial(n, Xt, Yt):
    """
       Строит полином Pn(x) n-й степени лучшего среднеквадратического
    приближенния к функции f(x) заданной таблично.
    :param n: Степень полинома
    :param Xt: список x значений
    :param Yt: список y значений функции f(x)
    :return: список коэфициэнтов (a0, a1, ... , an) полинома Pn(x)
    """
    matrixSize = n + 1
    A = np.zeros((matrixSize, matrixSize))
    B = np.zeros(matrixSize)

    # Получаем СЛАР в матричном виде
    for t in range(matrixSize):
        x_sum = 0
        y_sum = 0
        for i in range(len(Xt)):
            tmp = Xt[i] ** t
            # Ищем суму Xi^n
            x_sum += tmp
            # Ищем суму Yi * Xi^n
            y_sum += tmp * Yt[i]

        # Заполняем матрицу B
        B[t] = y_sum

        # Ставим сумму в нужные места в матрице A
        i = 0
        j = t
        while i <= t and j >= 0:
            A[i, j] = x_sum
            i += 1
            j -= 1

    for t in range(n + 1, 2 * n + 1):
        # Ищем суму Xi
        x_sum = 0
        for x in Xt:
            x_sum += x ** t

        # Ставим сумму в нужные места в матрице
        tmp = t - n
        i = tmp
        j = n
        while i <= n and j >= tmp:
            A[i, j] = x_sum
            i += 1
            j -= 1

    return np.linalg.solve(A, B)


def calculate_polynomial(Pn, x):
    """
       Считает значение полинома Pn(x) в точке х
    :param Pn: список коэфициенттов полинома
    :param x: точка в которой нужно посчитать значение
    :return: значение полинома Pn(x) в точке х
    """
    result = 0
    t = len(Pn)
    for i in range(t):
        result += Pn[i] * (x ** i)
    return result


# Границы функции
a = 1
b = 3

# Шаг
H = (b - a) / 10

# Количество точек
m = 11
arrSize = m

# Таблица значений функций
Xt = np.zeros(arrSize)
Yt = np.zeros(arrSize)

# Создание таблицы
for i in range(arrSize):
    Xt[i] = a + i*H
    Yt[i] = my_function(Xt[i])

# Максимальный степень полинома (включитльно)
n = 4

# для графика основной функции
h1 = (b - a) / 20
H2 = 2*H
start = a - H2
end = b + H2
xlist = []
ylist1 = []
while start <= end:
    f = my_function(start)
    xlist.append(start)
    ylist1.append(f)
    start += h1
plt.subplot(2, 1, 1)
plt.plot(xlist, ylist1, 'k', label='f(x)')

for i in range(n + 1):
    # Получаем коэфициенты полинома i-й степени
    Pn = polynomial(i, Xt, Yt)
    # Выводим таблицу
    print("For n = {0}".format(i))
    print("----------------------------------------------------------------------")
    print("|      |          |          |                | f(xj) - Pn(xj)       |")
    print("|  xj  |   f(xj)  |  Pn(xj)  | f(xj) - Pn(xj) | -------------- * 100 |")
    print("|      |          |          |                |      Pn(xj)          |")
    print("----------------------------------------------------------------------")
    start = a - H2
    end = b + H2
    ylist2 = []
    while start <= end:
        f = my_function(start)
        p = calculate_polynomial(Pn, start)
        ylist2.append(p)
        print("|{0:5.2f} | {1:8.3f} | {2:8.3f} | {3:14.9f} | {4:21.16f}|".format(start, f, p, p - f, (p-f) * 100 / f))
        start += h1
    plt.plot(xlist, ylist2, '--', label='P{0}(x)'.format(i))
    print("----------------------------------------------------------------------\n")

plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'$y = f(x), y = Pn(x)$')
plt.legend(loc='best', ncol=2)

# Additional task
plt.subplot(2, 1, 2)
plt.plot(xlist, ylist1, label='f(x)')
Pn = polynomial(3, Xt, Yt)
ylist = [calculate_polynomial(Pn, x) for x in xlist]
plt.plot(xlist, ylist, '--', label='P3(x)')
P1 = [Pn[1], Pn[2], Pn[3]]
ylist = [calculate_polynomial(P1, x) for x in xlist]
plt.plot(xlist, ylist, ':', label='P3\'(x)')
ylist = [2*x + math.cos(x) for x in xlist]
plt.plot(xlist, ylist, label='f\'(x)')

plt.legend()
plt.show()
