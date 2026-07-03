import numpy as np
import pytest

from mlcore.basics.linear_regression import LinearRegression


def test_linear_regression_fits_simple_data():
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = 2.0 * X.reshape(-1) + 1.0

    model = LinearRegression(lr=0.01, n_epochs=3000)
    model.fit(X, y)

    assert model.weights is not None
    assert model.bias is not None

    assert model.weights[0] == pytest.approx(2.0, abs=0.1)
    assert model.bias == pytest.approx(1.0, abs=0.2)


def test_predict_shape():
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([3.0, 5.0, 7.0])

    model = LinearRegression(lr=0.01, n_epochs=1000)
    model.fit(X, y)

    pred = model.predict(X)

    assert pred.shape == y.shape


def test_loss_decreases():
    rng = np.random.default_rng(42)

    X = rng.normal(size=(100, 1))
    y = 3.0 * X.reshape(-1) - 2.0

    model = LinearRegression(lr=0.05, n_epochs=200)
    model.fit(X, y)

    assert model.loss_history[0] > model.loss_history[-1]


def test_predict_before_fit_raises_error():
    model = LinearRegression()

    with pytest.raises(RuntimeError):
        model.predict(np.array([[1.0]]))