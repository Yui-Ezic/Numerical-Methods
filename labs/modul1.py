"""                            Модульна контрольна робота
                          з курсу Чисельні методи, варіант 6
    Завдання: Знайти визначник матриці
    Виконав студент 2 курсу: Зуєв Михайло Олександрович
"""
import numpy as np


def minor(i, j, A):
    """ Ищет минор матрицы 4х4"""
    m_shape = A.shape[0] - 1
    m = np.eye(m_shape)
    m[:i, :j] = A[:i, :j]
    m[:i, j:] = A[:i, j + 1:]
    m[i:, :j] = A[i + 1:, :j]
    m[i:, j:] = A[i + 1:, j + 1:]
    return np.linalg.det(m)


def det(A):
    """
       Шукає визначник за алгебраїчним доповненням
    :param A: Матриця
    :return: Визначник матриці
    """
    det = 0
    # размер матрицы
    n = int(A.size ** 0.5)
    for j in range(n):
        # Алгебраическое дополнение
        alg = minor(1, j, A) * (-1)**(1+j)
        det += A[1, j]*alg
    return det


def lu_det(A):
    """
       Обчислює визначник матриці за допомогою LU-факторизації.
    :param A: Матриця
    :return: Визначник матриці
    """
    # размер матрицы
    n = int(A.size ** 0.5)
    # LU факторизация (матрицу L не ищем за ненадобностью)
    for i in range(n):
        for j in range(i + 1, n, 1):
            r = A[j][i] / A[i][i]
            A[j] = A[j] - A[i] * r
    # Вычесляем определитель
    det = 1
    for i in range(n):
        det *= A[i, i]
    return det


# Входная матрица
A = np.array([[8.0, 1000.0, -1.0, 7.0],
              [-3.0, 8.0, 87.0, 5.0],
              [100.0, 6, 8, 0.0],
              [12.0, 6.0, 1, 67.0]])

# Ищем определитель методом алгебраических дополнений
det1 = det(A)
# Ищем определитель методом LU-факторизации
det2 = lu_det(A)

print("Определитель методом аглебраических дополнений: {0}".format(det1))
print("Определитель методом LU-факторизации: {0}".format(det2))
print()

print("Additional task")
# Входная матрица
Ad = np.array([[1000.0, 8.0, -1.0, 7.0],
              [8.0, -1.0, 87.0, 5.0],
              [6.0, 100, 8, 0.0],
              [6.0, 12.0, 1, 67.0]])

# Ищем определитель методом алгебраических дополнений
det1d = det(Ad)
# Ищем определитель методом LU-факторизации
det2d = lu_det(Ad)

print("Определитель методом аглебраических дополнений: {0}".format(det1d))
print("Определитель методом LU-факторизации: {0}".format(det2d))


