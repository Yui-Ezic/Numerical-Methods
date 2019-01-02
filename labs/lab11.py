"""              Лабораторна работа номер 11
             з курсу Чисельні методи, варіант 6
    Завдання: Наближенно обчислити значення похідної.
    f(x) = sin(x) + cos(2x), на відрізку [1;3.5] для 10 значень.
    Виконав студент 2 курсу: Зуєв Михайло Олександрович
"""
import math
import matplotlib.pyplot as plt


def function(arg):
    """
       Считает значение функции f(x) = sin(x) + cos(2x)
    :param arg: аргумент функции
    :return: значение функции f(x) в точке x = arg
    """
    return math.sin(arg) + math.cos(2*arg)


def function_derivative2(arg):
    """
       Считает значение второй производной функции f(x) = sin(x) + cos(2x)
    f'(x) = cos(x) - 2sin(2x)
    f''(x) = -sin(x) - 4*cos(2x)
    :param arg: аргумент функции
    :return: значение функции f'(x) в точке x = arg
    """
    return -math.sin(arg) - 4 * math.cos(2*arg)


def approximate_derivative2(y0, y1, y2, dx):
    """
        Считает приближенное значение второй производной по двум значением и шагу
    :param y0: первое значение функции
    :param y1: второе значение функции (y0 + dx)
    :param y2: третье значение функции (y0 + 2*dx)
    :param dx: шаг агрумента функции
    :return: значение второй производной
    """
    return (y2 - 2*y1 + y0) / (dx * dx)


# Границы функции
a = 1
b = 3.5

# Количество точек
n = 10

# Шаг
h: float = (b - a) / n
dx = 1e-3

# Для графиков функций
xData = []

# Значения функции
f_yData = []

# Значения точной производной
d1_yData = []

# Значения приближенной производной
d2_yData = []

# Считаем производные и строим таблицу
x = a
print("---------------------------------------------------------------------------------")
print("|      |          |           |           | d_ex - d_ap       |                 |")
print("|  xi  |   f(xi)  |  d_aprox  |  d_exact  | ----------- * 100 | additional task |")
print("|      |          |           |           |     d_ex          |                 |")
print("---------------------------------------------------------------------------------")
while x < b:
    # Значение функции
    f = function(x)
    # Точное значение производной
    d1 = function_derivative2(x)
    # Приближенное значение производной
    d2 = approximate_derivative2(f, function(x + dx), function(x + 2 * dx), dx)
    # Приближенное значение производной уменьшеным в 4 раза шагом
    d3 = approximate_derivative2(f, function(x + dx / 4), function(x + 2 * dx / 4), dx / 4)
    print("|{0:5.2f} | {1:8.3f} | {2:9.3f} | {3:9.3f} |  {4:13.2f} %  |  {5:11.2f} %  |".\
          format(x, f, d2, d1, abs((d1 - d2) * 100 / d1), abs((d1 - d3) * 100 / d1)))
    # Добавляем данные для графиков
    xData.append(x)
    f_yData.append(f)
    d1_yData.append(d1)
    d2_yData.append(d2)
    x += h
print("---------------------------------------------------------------------------------")

# Рисуем графики
plt.subplot(2, 1, 1)
plt.plot(xData, f_yData, label='f(x)')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r"$f(x) = sin(x) + cos(2x)$")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(xData, d1_yData, label='exact f\'\'(x)')
plt.plot(xData, d2_yData, '--', label='approximate f\'\'(x)')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r"$f''(x)$")
plt.legend()

plt.show()

