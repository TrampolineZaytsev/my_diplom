import numpy as np

def get_approx_polinom(points, deg):

    x = points[:, 0]
    y = points[:, 1]
   
    # получаем коэфициенты полинома
    coefficients = np.polyfit(x, y, deg=deg)
    poly = np.poly1d(coefficients)

    # вычисляем точки полинома
    x_poly = x
    y_poly = poly(x_poly)

    # оцениваем качество интерполяции
    mse = np.mean((y_poly - y)**2)

    return coefficients, mse