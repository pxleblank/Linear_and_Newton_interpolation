import matplotlib.pyplot as plt
import numpy as np
import math
import random


def fi(x):
    global a, b
    y = abs(x)
    return y


# def lin_interpol(x0: list, x: list, y: list) -> list:
#     n = len(x) - 1
#     f_x_stars = [0] * len(x0)
#
#     for j in range(len(x0)):
#         for i in range(n):
#             if x[i] <= x0[j] <= x[i + 1]:
#                 f_x_stars[j] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]) * (x0[j] - x[i]) + y[i]
#                 break
#
#     return f_x_stars

def lin_interpol(x0: float, x: list, y: list) -> float:
    n = len(x) - 1

    for i in range(n):
        if x[i] <= x0 <= x[i + 1]:
            return (y[i + 1] - y[i]) / (x[i + 1] - x[i]) * (x0 - x[i]) + y[i]


def divided_diff(x, y):
    """Функция для вычисления таблицы разделённых разностей"""
    n = len(x)
    div_diff = [[0.] * n for _ in range(n)]

    # Первая колонка — это значения y
    for i in range(n):
        div_diff[i][0] = y[i]

    # Вычисляем разделённые разности
    for j in range(1, n):
        for i in range(n - j):
            div_diff[i][j] = (div_diff[i + 1][j - 1] - div_diff[i][j - 1]) / (x[i + j] - x[i])

    return div_diff[0]  # Возвращаем только верхнюю строку (разделённые разности)


def newton_pol(x0, x, div_diff):
    """Функция для вычисления полинома Ньютона"""
    n = len(div_diff)
    result = div_diff[0]
    product = 1

    for i in range(1, n):
        product *= (x0 - x[i - 1])
        result += float(div_diff[i]) * product

    return result


# def newton_pol(x0, x, y):
#     # Рекурсивная функция для вычисления разделённых разностей
#     def divided_diff_recursive(x, y, i, j):
#         if j == 0:
#             return y[i]
#         return (divided_diff_recursive(x, y, i + 1, j - 1) - divided_diff_recursive(x, y, i, j - 1)) / (x[i + j] - x[i])
#
#     # Рекурсивная функция для вычисления интерполяции
#     def newton_interpol_recursive(x0, x, i):
#         if i == 0:
#             return divided_diff_recursive(x, y, 0, 0)
#         product = 1
#         for j in range(i):
#             product *= (x0 - x[j])
#         return divided_diff_recursive(x, y, 0, i) * product + newton_interpol_recursive(x0, x, i - 1)
#
#     n = len(x)
#     return newton_interpol_recursive(x0, x, n - 1)


N = 20
a = -1.
b = 1.
x = [0] * (N + 1)
y = [0] * (N + 1)
x[0] = a
y[0] = fi(a)
x[N] = b
y[N] = fi(b)

h = (b - a) / N

# for i in range(1, N):
#     x[i] = a + h * (i + 0.0 * np.sin(5 * i))
#     y[i] = fi(x[i])


x_mid = (a + b) / 2
r = (b - a) / 2
for i in range(N + 1):
    alpha = math.pi * i / N  # угол для каждого узла на полукруге
    x[i] = x_mid + r * math.cos(alpha)  # координата узла по оси x
    y[i] = fi(x[i])

# Сортируем узлы по возрастанию x
sorted_pairs = sorted(zip(x, y))
x_sorted, y_sorted = zip(*sorted_pairs)
x = list(x_sorted)
y = list(y_sorted)

div_diff = divided_diff(x, y)

x_star = float(input('x* = '))

# функция fi
L = 500
step = (b - a) / L
x0 = [a + step * i for i in range(L+1)]
y0 = [fi(xi) for xi in x0]

y_linear = [lin_interpol(xi, x, y) for xi in x0]
y_star_linear = lin_interpol(x_star, x, y)

y_newton = [newton_pol(x0_i, x, div_diff) for x0_i in x0]
y_star_newton = newton_pol(x_star, x, div_diff)

plt.figure(figsize=(10, 6))
# построение fi
plt.plot(x0, y0, label='f0(x)', color='blue', linewidth=2)

# построение f(x*) линейной
plt.plot(x0, y_linear, label='Линейная интерполяция', linestyle='--', color='red')
plt.scatter([x_star], [y_star_linear], color='red', marker='x', s=100, label=f'f(x*) (линейная)')

# построение f(x*) Ньютона
plt.plot(x0, y_newton, label=f'Полином Ньютона (степень {N})', linestyle='--', color='green')
plt.scatter([x_star], [y_star_newton], color='green', marker='x', s=100, label=f'f(x*) (Ньютон)')

plt.scatter(x, y, color='black', zorder=5, label='Узлы интерполяции')

plt.xlim(a - (b-a)/20, b + (b-a)/20)
plt.ylim(-1, 2)

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Интерполяция')
plt.legend()
plt.grid(True)
plt.show()

print(f'Значение f(x*) при линейной интерполяции: {y_star_linear}')
print(f'Значение f(x*) при интерполяции полиномом Ньютона: {y_star_newton}')
