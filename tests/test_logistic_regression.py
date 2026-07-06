import numpy as np
import pytest

from mlcore.basics.logistic_regression import LogisticRegression


def test_logistic_regression_fits_simple_data():
    X = np.array(
        [
            [-2.0],
            [-1.0],
            [0.0],
            [1.0],
            [2.0],
        ]
    )
    y = np.array([0, 0, 0, 1, 1])

    model = LogisticRegression(lr=0.1, n_epochs=2000)
    model.fit(X, y)

    y_pred = model.predict(X)

    assert np.mean(y_pred == y) >= 0.8


def test_predict_proba_shape():
    X = np.array([[-1.0], [0.0], [1.0]])
    y = np.array([0, 0, 1])

    model = LogisticRegression(lr=0.1, n_epochs=1000)
    model.fit(X, y)

    proba = model.predict_proba(X)

    assert proba.shape == y.shape


def test_predict_returns_binary_labels():
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])

    model = LogisticRegression(lr=0.1, n_epochs=1000)
    model.fit(X, y)

    y_pred = model.predict(X)

    assert set(np.unique(y_pred)).issubset({0, 1})


def test_loss_decreases():
    rng = np.random.default_rng(42)

    X = rng.normal(size=(200, 2))
    true_weights = np.array([2.0, -3.0])
    logits = X @ true_weights + 0.5
    y = (logits > 0).astype(int)

    model = LogisticRegression(lr=0.1, n_epochs=300)
    model.fit(X, y)

    assert model.loss_history[0] > model.loss_history[-1]


def test_logistic_regression_fits_multiple_features():
    rng = np.random.default_rng(42)

    X = rng.normal(size=(300, 2))
    true_weights = np.array([2.0, -3.0])
    logits = X @ true_weights + 0.5
    y = (logits > 0).astype(int)

    model = LogisticRegression(lr=0.1, n_epochs=1000)
    model.fit(X, y)

    accuracy = model.score_accuracy(X, y)

    assert accuracy > 0.9


def test_predict_before_fit_raises_error():
    model = LogisticRegression()

    with pytest.raises(RuntimeError):
        model.predict(np.array([[1.0]]))


def test_fit_rejects_non_binary_labels():
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([0, 1, 2])

    model = LogisticRegression()

    with pytest.raises(ValueError):
        model.fit(X, y)

def test_predict_rejects_wrong_number_of_features():
    X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    y = np.array([0, 1, 1])

    model = LogisticRegression(lr=0.1, n_epochs=100)
    model.fit(X, y)

    X_wrong = np.array([[1.0, 2.0, 3.0]])

    with pytest.raises(ValueError):
        model.predict(X_wrong)