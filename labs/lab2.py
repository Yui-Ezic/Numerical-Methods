"""                            Лабораторна работа номер 2
                          з курсу Чисельні методи, варіант 6
    Завдання: записати систему рiвнянь у матричнiй формi i скласти програму її розв'язування методом
    Гаусса  з  порядковою  реалiзацiєю i вибором головного елемента
    Виконав студент 2 курсу: Зуєв Михайло Олександрович

"""
import numpy as np

A = np.array([[8.0, 1000.0, -1.0, 7.0],
              [-3.0, 8.0, 87.0, 5.0],
              [100.0, 6, 8, 0.0],
              [12.0, 6.0, 1, 67.0]])
Ac = np.array([[8.0, 1000.0, -1.0, 7.0],
              [-3.0, 8.0, 87.0, 5.0],
              [100.0, 6, 8, 0.0],
              [12.0, 6.0, 1, 67.0]])
B = np.array([65.0, 3.0, -5.0, 0.0])
Bc = np.array([65.0, 3.0, -5.0, 0.0])
XN = np.array(["x1", "x2", "x3", "x4"])
X = np.zeros(4)

for i in range(0, 4):
    # Ищем максимальный элемент в строке
    maxElement = A[i, 0]
    maxIndex = 0
    for j in range(1, 4):
        if abs(A[i, j]) > abs(maxElement):
            maxElement = A[i, j]
            maxIndex = j
    # Размещаем его на диагонали
    if maxIndex != i:
        A[:, [i, maxIndex]] = A[:, [maxIndex, i]]
        XN[[i, maxIndex]] = XN[[maxIndex, i]]
    # Прямой ход
    for k in range(i + 1, 4):
        R = A[k, i] / A[i, i]
        A[k] = A[k] - A[i] * R
        B[k] = B[k] - B[i]*R

# Обратный ход
for k in range(3, -1, -1):
    for i in range(3, k, -1):
        B[k] -= A[k, i]*X[i]
    X[k] = B[k] / A[k, k]

# Сортировка массив X
for i in range(0, 4):
    for j in range(0, 3 - i):
        if XN[j] > XN[j+1]:
            XN[[j, j + 1]] = XN[[j+1, j]]
            X[[j, j+1]] = X[[j+1, j]]

# Вывод результатов
for i in range(0, 4):
    print(XN[i] + " = " + str(X[i]))

# Вектор отклонений
V = Bc - np.dot(Ac, X)
print("Vector of deviations: ")
print(V)