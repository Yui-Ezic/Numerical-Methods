import numpy as np
import math
import matplotlib.pyplot as plt


def my_function(x):
    return x * x + math.sin(x)


# Границы функции
a1 = 1
b1 = 3

# Шаг
H = (b1 - a1) / 10

# Количество точек
m = 11
arrSize = m

# Таблица значений функций
Xt = np.zeros(arrSize)
Yt = np.zeros(arrSize)

# Создание таблицы
for i in range(arrSize):
    Xt[i] = a1 + i*H
    Yt[i] = my_function(Xt[i])

n = 4
a = np.zeros(n + 1)
aFlag = np.zeros(n + 1)

for i in range(m):
    a[0] += Yt[i]
a[0] /= m
aFlag[0] = 1

b = np.zeros(n + 1)
bFlag = np.zeros(n + 1)
bFlag[0] = 1


def g(j, x):
    if j == -1:
        return 0
    if j == 0:
        return 1
    if aFlag[j] == 0:
        c = 0
        d = 0
        for i in range(m):
            f = g(j - 1, Xt[i])
            f_2 = f * f
            c += Xt[i] * f_2
            d += f_2
        a[j] = c / d
        aFlag[j] = 1
        c = 0
        d = 0
        for i in range(m):
            f = g(j, Xt[i])
            f_2 = f * f
            c += f * Yt[i]
            d += f_2
        a[j] = c / d
    if bFlag[j - 1] == 0:
        c = 0
        d = 0
        for i in range(m):
            f = g(j - 1, Xt[i])
            f1 = g(j - 2, Xt[i])
            c += Xt[i] * f * f1
            d += f1 * f1
        b[j - 1] = c / d
        bFlag[j - 1] = 1
    return (x - a[j]) * g(j-1, x) - b[j - 1] * g(j - 2, x)


def calculate_polynomial(x, n):
    result = 0
    for i in range(n + 1):
        result += a[i] * g(i, x)
    return result


# для графика основной функции
h1 = (b1 - a1) / 20
H2 = 2*H
start = a1 - H2
end = b1 + H2
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
    #Выводим таблицу
    print("For n = {0}".format(i))
    print("----------------------------------------------------------------------")
    print("|      |          |          |                | f(xj) - Pn(xj)       |")
    print("|  xj  |   f(xj)  |  Pn(xj)  | f(xj) - Pn(xj) | -------------- * 100 |")
    print("|      |          |          |                |      Pn(xj)          |")
    print("----------------------------------------------------------------------")
    start = a1 - H2
    ylist2 = []
    while start <= end:
        f = my_function(start)
        p = calculate_polynomial(start, i)
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
ylist = [calculate_polynomial(x, 3) for x in xlist]
plt.plot(xlist, ylist, '--', label='P3(x)')
ylist = [2 * x * x - 12 * x + 11.288 for x in xlist]
plt.plot(xlist, ylist, ':', label='P3\'(x)')
ylist = [2*x + math.cos(x) for x in xlist]
plt.plot(xlist, ylist, label='f\'(x)')

plt.legend()
plt.show()

