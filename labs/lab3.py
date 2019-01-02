"""                            Лабораторна работа номер 3
                          з курсу Чисельні методи, варіант 6
    Завдання: Записати систему рiвнянь, яку треба розв'язати, у матричнiй формi.
    Скласти програму для розв'язування системи iтерацiйним методом при  = 0,001.
    У програмi використати метод простої ітерації.
    Виконав студент 2 курсу: Зуєв Михайло Олександрович

"""
import numpy as np
import matplotlib.pyplot as plt

eps = 0.001  # точность
N = 4  # порядок матрицы
A = np.array([[100.0, 6, 8, 0.0],
              [8.0, 1000.0, -1.0, 7.0],
              [-3.0, 8.0, 87.0, 5.0],
              [12.0, 6.0, 1, 67.0]])
B = np.array([-5.0, 65.0, 3.0, 0.0])
X1 = np.zeros(N)
X2 = np.zeros(N)
flag = True
k = 0

H = np.zeros((N, N))
V = np.zeros(N)

for i in range(N):
    for j in range(N):
        if i != j:
            H[i, j] = A[i, j] / A[i, i]
    V[i] = B[i] / A[i, i]

while flag and k < 50:
    X2 = np.dot(H, X1) + V
    g = 0
    i = 0
    while g < eps and i < N:
        g = abs(X2[i] - X1[i])
        s = abs(X2[i])
        if s > 1:
            g = g/s
        i += 1
    if i == N:
        flag = False
        print("Solution is found")
    k += 1
    X1 = X2

if k == 50:
    print("ERROR")
else:
    print(X2)

# рисуем график
eps = 10**(-7)
X1 = np.zeros(N)
X2 = np.zeros(N)
flag = True
k = 0
XTOCH = np.linalg.solve(A, B)
xlist = []
ylist = []
while flag and k < 50:
    X2 = np.dot(H, X1) + V
    g = 0
    i = 0
    while g < eps and i < N:
        g = abs(X2[i] - X1[i])
        s = abs(X2[i])
        if s > 1:
            g = g/s
        i += 1
    if i == N:
        flag = False
    k += 1
    xindex = X2.tolist().index(max(X2))
    xlist.append(k)
    ylist.append((XTOCH[xindex] - X2[xindex]) / XTOCH[xindex])
    X1 = X2
plt.plot(xlist, ylist)
plt.xlabel(r'$n$')
plt.ylabel(r'$(X^n - X^*)/X^*$')
plt.show()

# Вектор отклонений
D = B - np.dot(A, X2)
print("Vector of deviations: ")
print(D)

print(X2)
print(XTOCH)
print(X2 - XTOCH)

