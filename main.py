import matplotlib.pyplot as plt
import numpy as np
import random


def fi(x_):
    return [np.sin(x) for x in x_]


def lin_interpol(x0: list, x: list, y: list) -> list:
    n = len(x) - 1
    f_x_stars = [0] * len(x0)

    for j in range(len(x0)):
        for i in range(n):
            if x[i] <= x0[j] <= x[i + 1]:
                f_x_stars[j] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]) * (x0[j] - x[i]) + y[i]
                break

    return f_x_stars


def newton_pol(x0, x, y):
    def divided_diff(x, y):
        n = len(y)
        div_diff = [[0] * n for _ in range(n)]
        for i in range(n):
            div_diff[i][0] = y[i]
        for j in range(1, n):
            for i in range(n - j):
                div_diff[i][j] = (div_diff[i + 1][j - 1] - div_diff[i][j - 1]) / (x[i + j] - x[i])

        return div_diff[0]

    def newton_interpol(x0, x, div_diff):
        n = len(div_diff)
        result = div_diff[0]
        product = 1
        for i in range(1, n):
            product *= (x0 - x[i - 1])
            result += div_diff[i] * product
        return result

    div_diff = divided_diff(x, y)
    return newton_interpol(x0, x, div_diff)


x = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
# x = sorted({random.randint(a=-10, b=10) for x in range(10)})
y = fi(x)
N = len(x) - 1
x_star = float(input('x* = '))

x0 = np.linspace(x[0], x[-1], 500)
y0 = fi(x0)

y_linear = lin_interpol(x0, x, y)
y_star_linear = lin_interpol([x_star], x, y)[0]

y_newton = [newton_pol(x0_i, x, y) for x0_i in x0]
y_star_newton = newton_pol(x_star, x, y)

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

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Интерполяция')
plt.legend()
plt.grid(True)
plt.show()

print(f'Значение f(x*) при линейной интерполяции: {y_star_linear}')
print(f'Значение f(x*) при интерполяции полиномом Ньютона: {y_star_newton}')
