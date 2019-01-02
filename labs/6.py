"""                            Лабораторна работа номер 6
                          з курсу Чисельні методи, варіант 6
    Завдання: Побудувати поліном Pn(x) найкращого середньоквадратичного
    наближення для функції f(x), яку задано таблично, з використанням
    ортогональних поліномів.
      Для оцінювання похибки значень функція  f(x) задана.
      f(x) = x^2 + sin(x)
    Виконав студент 2 курсу: Зуєв Михайло Олександрович
"""

import numpy as np
import matplotlib.pyplot as plt
import math


def polynomial(xData, yData, n):
    """
       Вычисляет коэфициенты полинома n-й степени, полученного с помощью
    ортогональных полиномов в методе наименших квадратов.
    :param xData: Таблица аргументов функции
    :param yData: Таблица значений функции
    :param n: степень полинома
    :return: список коэфициэнтов c полинома c0 + c1*x * c2*x^2 + ... + cn*x^n
    """
    a = np.zeros((n + 1, n + 1), dtype=np.float64)
    b = np.zeros((n + 1), dtype=np.float64)
    s = np.zeros((2 * n + 1), dtype=np.float64)

    for i in range(len(xData)):
        temp = yData[i]
        for j in range(n + 1):
            b[j] = b[j] + temp
            temp *= xData[i]

        temp = 1.0
        for j in range(2 * n + 1):
            s[j] = s[j] + temp
            temp *= xData[i]

    for i in range(n + 1):
        for j in range(n + 1):
            a[i, j] = s[i+j]
    return np.linalg.solve(a, b)


def my_function(x):
    return x * x + math.sin(x)


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
    Pn = polynomial(Xt, Yt, i)
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
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'$y = f(x), y = Pn(x), y = f\'(x), y = P\'(x)$')
plt.plot(xlist, ylist1, label='f(x)')
Pn = polynomial(Xt, Yt, 3)
ylist = [calculate_polynomial(Pn, x) for x in xlist]
plt.plot(xlist, ylist, '--', label='P3(x)')
P1 = [Pn[1], Pn[2], Pn[3]]
ylist = [calculate_polynomial(P1, x) for x in xlist]
plt.plot(xlist, ylist, ':', label='P3\'(x)')
ylist = [2*x + math.cos(x) for x in xlist]
plt.plot(xlist, ylist, label='f\'(x)')

plt.legend()
plt.show()