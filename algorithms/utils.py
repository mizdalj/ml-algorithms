import numpy as np


def my_train_test_split(X, y, test_size=0.2, random_seed=None):
    # Set the random seed for reproducibility
    np.random.seed(random_seed)

    # Generate shuffled indices
    shuffled_indices = np.random.permutation(len(X))

    # Calculate the test set size
    test_set_size = int(len(X) * test_size)

    # Split the indices for the train and test set
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]

    # Check if X, y are numpy arrays or pandas DataFrame/Series
    if isinstance(X, (np.ndarray)):
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
    else:  # for pandas DataFrame/Series
        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

    return X_train, X_test, y_train, y_test
