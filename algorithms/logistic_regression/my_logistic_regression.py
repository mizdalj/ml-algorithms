import numpy as np


class MyLogisticRegression:
    def __init__(self, learning_rate=0.001, n_iters=1000, reg_lambda=0.01):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.reg_lambda = reg_lambda  # Regularization strength

    def fit(self, X, y, lambda_value=0.01, gamma=0.9, batch_size=32, decay_step=100, decay_rate=0.5):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0
        v_w = 0
        v_b = 0

        for epoch in range(self.n_iters):
            indices = np.arange(num_samples)
            np.random.shuffle(indices)

            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_X = X[indices[start_idx:end_idx]]
                batch_y = y[indices[start_idx:end_idx]]

                model = np.dot(batch_X, self.weights) + self.bias
                predictions = self.sigmoid(model)
                dw = (1 / len(batch_X)) * np.dot(batch_X.T, (predictions - batch_y)) + 2 * lambda_value * self.weights
                db = (1 / len(batch_X)) * np.sum(predictions - batch_y)

                v_w = gamma * v_w + self.lr * dw
                v_b = gamma * v_b + self.lr * db

                self.weights -= v_w
                self.bias -= v_b

            if epoch % decay_step == 0:
                self.lr *= decay_rate

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict_prob(self, X):
        model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(model)

    def predict(self, X, threshold=0.5):
        probs = self.predict_prob(X)
        class_predictions = [1 if i > threshold else 0 for i in probs]
        return class_predictions

    def loss(self, y_true, y_pred_prob):
        epsilon = 1e-15  # To prevent log(0)
        y_pred_prob = np.clip(y_pred_prob, epsilon, 1 - epsilon)
        loss = -1 / len(y_true) * np.sum(y_true * np.log(y_pred_prob) + (1 - y_true) * np.log(1 - y_pred_prob))
        return loss
