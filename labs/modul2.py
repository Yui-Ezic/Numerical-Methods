"""                            Модульна контрольна робота
                          з курсу Чисельні методи, варіант 6
    Завдання: Знайти обернену матрицю
    Виконав студент 2 курсу: Зуєв Михайло Олександрович
"""
import numpy as np


def lu_solve(A, B):
    """
      Решает СЛАР методом LU-факторизации
    :param A: Входная матрица
    :param B: Матрица неизветсных
    :return: матрицу неизвестных
    """
    A1 = np.array([y for y in A])
    B1 = np.array([y for y in B])
    B2 = np.array([y for y in B])
    n = 4
    X = np.zeros(n)
    L = np.eye(n)
    Y = np.array([0.0, 0.0, 0.0, 0.0])
    for i in range(4):
        for j in range(i + 1, 4, 1):
            R = A1[j][i] / A1[i][i]
            A1[j] = A1[j] - A1[i] * R
            B1[j] = B1[j] - B1[i] * R
            L[j][i] = R

    for i in range(4):
        sum = 0.0
        for j in range(0, i, 1):
            sum += L[i][j] * Y[j]
        Y[i] = (B2[i] - sum) / L[i][i]

    for i in range(3, -1, -1):
        sum = 0.0
        for j in range(3, i, -1):
            sum += A1[i][j] * X[j]
        X[i] = (Y[i] - sum) / A1[i][i]
    return X


n = 4
A = np.array([[8.0, 1000.0, -1.0, 7.0],
              [-3.0, 8.0, 87.0, 5.0],
              [100.0, 6, 8, 0.0],
              [12.0, 6.0, 1, 67.0]])
B = np.eye(n)
Z = np.zeros([n, n])
for i in range(n):
    Z[:, i] = lu_solve(A, B[:, i])

print("A ^ -1: ")
print(Z)
print()
print("A ^ (-1) * A")
print(np.dot(Z, A))
print()
print("A * A ^ -1")
print(np.dot(A, Z))

# Additional task
sumA = np.zeros(n)
sumZ = np.zeros(n)
for i in range(n):
    for j in range(n):
        sumA[i] += A[i, j]
        sumZ[i] += Z[i, j]
maxSumA = max(sumA)
maxSumZ = max(sumZ)
cond = maxSumA * maxSumZ

print()
print("cond(A) = {0}".format(cond))




