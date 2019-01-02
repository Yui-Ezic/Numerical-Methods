"""                            Лабораторна работа номер 9
                          з курсу Чисельні методи, варіант 6
    Завдання: Знайти всі корені рівняння НАТР за допомогою
    метода простих ітерацій.
      | y - sin(x) = 0
      |
      | x + 100y = 0
    Виконав студент 2 курсу: Зуєв Михайло Олександрович
"""
import matplotlib.pyplot as plt
import numpy as np
import math


def solve(x_0=None, print_table=False):
    """
       Находит корень НАТР на промежутке x0, y0 є [left; right]
    :param print_table: True чтобы печатать таблицу
    :param x_0: начальное приближение
    :return: корень x0, y0 НАТР, количество итераций
    """
    if x_0 is None:
        x_0 = [0, 0]
    # точность с которой ищем корни
    eps = 1e-3
    # счетчик
    k = 0
    # предыдущее приближение ([x0, y0])
    prev = np.array(x_0, dtype=np.float64)
    # текущее приближение ([xk, yk])
    cur = np.array(x_0, dtype=np.float64)
    # усливие выхода
    flag = True
    # печатаем таблицу
    if print_table:
        print("----------------------------------------------------")
        print("| k  |    xk    |    yk    |  F1(x,y)  |  F2(x,y)  |")
        print("----------------------------------------------------")
        y1 = cur[1] - math.sin(cur[0])
        y2 = cur[1] - 50 * cur[0]
        print("|{0:3} | {1:8.3} | {2:8.3} |{3:10.3} |{4:10.3} |".format(k, cur[0], cur[1], y1, y2))
    while flag:
        k += 1
        # считаем следующее приближение
        cur[0] = prev[1] / 50.0
        cur[1] = math.sin(prev[0])
        # проверяем условие выхода
        flag = math.fabs(max(cur - prev)) > eps
        # печатаем таблицу
        if print_table:
            y1 = cur[1] - math.sin(cur[0])
            y2 = cur[1] - 50 * cur[0]
            print("|{0:3} | {1:8.3} | {2:8.3} |{3:10.3} |{4:10.3} |".format(k, cur[0], cur[1], y1, y2))
        prev[0] = cur[0]
        prev[1] = cur[1]
    if print_table:
        print("----------------------------------------------------")
    return cur, k


root, k = solve([1, 1], True)
f1 = root[1] - math.sin(root[0])
f2 = root[1] - 50 * root[0]
print("x = {0:5.3f}\ny = {1:5.3f}\nF1(x, y) = {2:5.3f}\nF2(x, y) = {2:5.3f}".format(root[0], root[1], f1, f2))

# Рисуем график
plt.subplot(2, 1, 1)
xData = np.arange(-0.5, 0.5, 0.01)
yData = [math.asin(x) for x in xData]
plt.plot(xData, yData, label='f1:x=sin(y)')
yData = [50 * x for x in xData]
plt.plot(xData, yData, label='f2:y=50x')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'$y = f1(x), y = f2(x)$')
plt.legend()

# Дополнительное задание
plt.subplot(2, 1, 2)
plt.title(r'$N = N(x0), eps = 0.001$')
plt.xlabel(r'$x0$')
plt.ylabel(r'$N$')
x1 =[2, -2, 5, -8, -4]
x2 = [1, 1, -5, 0, 3]
for i in range(5):
    root, n = solve([x1[i], x2[i]])
    plt.bar([i], [n], label='{0}:{1}'.format(x1[i], x2[i]))

plt.legend()
plt.show()


