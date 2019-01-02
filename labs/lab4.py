"""                            Лабораторна работа номер 4
                          з курсу Чисельні методи, варіант 6
    Завдання: Дана функція f(x). Треба спочатку побудувати таблицю її значень
    і потім провести інтерполяцію. Побудувати таблицю її значень в (n+1)-й точцi
    xi=a+iH, H=(b-a)/n, i=0,1,…,n. В усiх варiантах, якщо не вказано iншого, взяти n=5.
    Додаткове завдання: Взявши чотири будь-які проміжні точки zj(zj≠xi), одна з яких
    лежить за межами відрізка [a,b], побудувати таблицю значень zj, f(zj), φ(zj),
    f(zj)-φ(zj), [f(zj)-φ(zj)]/ f(zj)·100 для n = 5, n = 20.
    Виконав студент 2 курсу: Зуєв Михайло Олександрович
"""
import numpy as np
import matplotlib.pyplot as plt
import math


def fi(x, xt, yt, n):
    """
      Рахує значення апроксимуючої функції за допомогою
    інтерполяційної формули Лагранжа.
    :param x: аргумент функції
    :param xt: таблиця значень x точної функції
    :param yt: таблиця значень y точної функції
    :param n: степінь поліному
    :return: значення апроксимуючої функції в точці X
    """
    func = 0
    tmp = []

    for i in range(n + 1):
        tmp.append(x - xt[i])

    for i in range(n + 1):
        numerator = 1
        denominator = 1
        for j in range(i):
            numerator *= tmp[j]
            denominator *= (xt[i] - xt[j])
        for j in range(i + 1, n + 1):
            denominator *= (xt[i] - xt[j])
            numerator *= tmp[j]
        func += yt[i] * (numerator / denominator)
    return func


# Границы функции
a = 1
b = 3

# Количество точек
n = 5
arrSize = n + 1

# Таблица значений функций
Xt = np.zeros(arrSize)
Yt = np.zeros(arrSize)

# Создание таблицы
H = (b - a) / n
for i in range(arrSize):
    Xt[i] = a + i*H
    Yt[i] = Xt[i] * Xt[i] + math.sin(Xt[i])

# Рисуем график
xlist = []
ylist1 = []
ylist2 = []
for i in range(a*10, b*10):
    x = i / 10
    xlist.append(x)
    ylist1.append(fi(x, Xt, Yt, n))
    ylist2.append(x * x + math.sin(x))

plt.plot(xlist, ylist1, xlist, ylist2)
plt.show()

# Строим таблицу
h = (b - a) / (4 * n)
x = a - H
while x < (b + H):
    fx = x * x + math.sin(x)
    xfi = fi(x, Xt, Yt, n)
    print("%3.2f | %6.3f | %6.3f | %6.3f | %6.3f |" % (x, fx, xfi, fx - xfi, ((fx - xfi) * 100) / fx))
    x += h

z1 = 1.85
z2 = 2.23
z3 = 0.78
z4 = 4.5
print("Additional task for n:")
f = z1*z1 + math.sin(z1)
xfi = fi(z1, Xt, Yt, n)
print("%3.2f | %6.3f | %6.3f | %6.3f | %6.3f |" % (z1, f, xfi, f - xfi, ((f - xfi) * 100) / f))
f = z2*z2 + math.sin(z2)
xfi = fi(z2, Xt, Yt, n)
print("%3.2f | %6.3f | %6.3f | %6.3f | %6.3f |" % (z2, f, xfi, f - xfi, ((f - xfi) * 100) / f))
f = z3*z3 + math.sin(z3)
xfi = fi(z3, Xt, Yt, n)
print("%3.2f | %6.3f | %6.3f | %6.3f | %6.3f |" % (z3, f, xfi, f - xfi, ((f - xfi) * 100) / f))
f = z4*z4 + math.sin(z4)
xfi = fi(z4, Xt, Yt, n)
print("%3.2f | %6.3f | %6.3f | %6.3f | %6.3f |" % (z4, f, xfi, f - xfi, ((f - xfi) * 100) / f))


# Создание таблицы для 4n
n4 = 4*n
arrSize4 = n4 + 1

# Таблица значений функций
Xt4 = np.zeros(arrSize4)
Yt4 = np.zeros(arrSize4)
H = (b - a) / n4
for i in range(arrSize4):
    Xt4[i] = a + i*H
    Yt4[i] = Xt4[i] * Xt4[i] + math.sin(Xt4[i])
print("Additional task for 4n:")
f = z1*z1 + math.sin(z1)
xfi = fi(z1, Xt4, Yt4, n4)
print("%3.2f | %6.3f | %6.3f | %6.3f | %6.3f |" % (z1, f, xfi, f - xfi, ((f - xfi) * 100) / f))
f = z2*z2 + math.sin(z2)
xfi = fi(z2, Xt4, Yt4, n4)
print("%3.2f | %6.3f | %6.3f | %6.3f | %6.3f |" % (z2, f, xfi, f - xfi, ((f - xfi) * 100) / f))
f = z3*z3 + math.sin(z3)
xfi = fi(z3, Xt4, Yt4, n4)
print("%3.2f | %6.3f | %6.3f | %6.3f | %6.3f |" % (z3, f, xfi, f - xfi, ((f - xfi) * 100) / f))
f = z4*z4 + math.sin(z4)
xfi = fi(z4, Xt4, Yt4, n4)
print("%3.2f | %6.3f | %6.3f | %6.3f | %6.3f |" % (z4, f, xfi, f - xfi, ((f - xfi) * 100) / f))
