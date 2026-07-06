from __future__ import annotations

import numpy as np

class LogisticRegression:
    def __init__(self, lr: float = 0.01, n_epochs: int = 500):
        self.lr = lr
        self.n_epochs = n_epochs
        self.weights: np.ndarray | None = None
        self.bias: float | None = None
        self.loss_history: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogisticRegression:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array for X, got {X.ndim}D array instead.")
        n_samples, n_features = X.shape
        if y.shape[0] != n_samples:
            raise ValueError(f"Number of samples in X ({n_samples}) and y ({y.shape[0]}) do not match.")
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("y must be a binary array containing only 0s and 1s.")

        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.n_epochs):
            logits = X @ self.weights + self.bias
            y_pred = self._sigmoid(logits)
            loss = self._binary_cross_entropy(y, y_pred)
            self.loss_history.append(loss)

            error = y_pred - y
            dw = (1 / n_samples) * (X.T @ error)
            db = (1 / n_samples) * np.sum(error)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None or self.bias is None:
            raise RuntimeError("Model has not been fitted yet.")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array for X, got {X.ndim}D array instead.")
        if X.shape[1] != self.weights.shape[0]:
            raise ValueError(f"Number of features in X ({X.shape[1]}) does not match number of weights ({self.weights.shape[0]}).")
        logits = X @ self.weights + self.bias
        return self._sigmoid(logits)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("Threshold must be between 0 and 1.")
        y_proba = self.predict_proba(X)
        return (y_proba >= threshold).astype(int)

    def score_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y).reshape(-1)
        y_pred = self.predict(X)
        return float(np.mean(y == y_pred))

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def _binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return float(loss)