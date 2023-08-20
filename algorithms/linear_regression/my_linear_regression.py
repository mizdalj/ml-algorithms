import numpy as np


def simple_linear_regression(X, y):
    n = len(X)

    # Calculate the sums
    sum_x = sum(X)
    sum_y = sum(y)
    sum_x_squared = sum([x ** 2 for x in X])
    sum_xy = sum([X[i] * y[i] for i in range(n)])

    # Calculate slope (m) and y-intercept (c)
    m = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)
    c = (sum_y - m * sum_x) / n

    return m, c


def predict(X, m, c):
    return [m * x + c for x in X]


def mean_squared_error(y_true, y_pred):
    return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)
