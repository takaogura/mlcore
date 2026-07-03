from __future__ import annotations

import numpy as np


class LinearRegression:
    def __init__(self, lr: float = 0.01, n_epochs: int = 1000):
        self.lr = lr
        self.n_epochs = n_epochs
        self.weights: np.ndarray | None = None
        self.bias: float | None = None
        self.loss_history: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> LinearRegression:
        X = np.asarray(X) #if X is a python list
        y = np.asarray(y) #if y is a python list
        if X.ndim == 1:
            X = X.reshape(-1,1)
        if X.ndim !=2:
            raise ValueError(f"Expected 2D array for X, got {X.ndim}D array instead.")
        n_samples, n_features = X.shape

        y = y.reshape(-1)

        if y.shape[0] != n_samples:
            raise ValueError(f"Number of samples in X ({n_samples}) and y ({y.shape[0]}) do not match.")

        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.n_epochs):
            y_pred = X @ self.weights + self.bias
            error = y_pred - y
            loss = np.mean(error**2)
            self.loss_history.append(loss)

            dw = (2 / n_samples) * (X.T @ error) #(n_features, n_samples) @ (n_samples,) -> (n_features,)
            db = (2 / n_samples) * np.sum(error)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None or self.bias is None:
            raise RuntimeError("Model has not been fitted yet. Please call 'fit()' first.")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array for X, got {X.ndim}D array instead.")

        return X @ self.weights + self.bias

    def score_mse(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y).reshape(-1)
        y_pred = self.predict(X)
        if y_pred.shape[0] != y.shape[0]:
            raise ValueError(f"Number of samples in X ({y_pred.shape[0]}) and y ({y.shape[0]}) do not match.")
        return np.mean((y_pred - y) ** 2)
