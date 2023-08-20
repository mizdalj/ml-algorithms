import numpy as np


# Custom Linear Regression Functions
def my_train_test_split(X, y, test_size=0.5, random_seed=None):
    # Set the random seed for reproducibility
    np.random.seed(random_seed)

    # Generate shuffled indices
    shuffled_indices = np.random.permutation(len(X))

    # Calculate the test set size
    test_set_size = int(len(X) * test_size)

    # Split the indices for the train and test set
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]

    # Use indices to get train and test subsets
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

    return X_train, X_test, y_train, y_test


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
