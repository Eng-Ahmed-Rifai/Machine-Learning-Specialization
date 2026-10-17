"""Vectorized Linear & Logistic Regression from scratch in NumPy.

Implements cost functions, analytical gradients, and gradient descent optimization
with optional L2 weight regularization.
"""

from typing import Tuple
import numpy as np


def compute_linear_cost(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lambda_: float = 0.0) -> float:
    """Compute mean squared error with L2 regularization."""
    m = X.shape[0]
    predictions = np.dot(X, w) + b
    cost = np.sum((predictions - y) ** 2) / (2 * m)
    reg_cost = (lambda_ / (2 * m)) * np.sum(w ** 2)
    return float(cost + reg_cost)


def compute_linear_gradient(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lambda_: float = 0.0) -> Tuple[np.ndarray, float]:
    """Compute gradients for linear regression."""
    m = X.shape[0]
    errors = (np.dot(X, w) + b) - y
    dj_dw = (np.dot(X.T, errors) / m) + (lambda_ / m) * w
    dj_db = float(np.sum(errors) / m)
    return dj_dw, dj_db


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Stable sigmoid activation function."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def compute_logistic_cost(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lambda_: float = 0.0) -> float:
    """Compute binary cross-entropy log loss with L2 regularization."""
    m = X.shape[0]
    z = np.dot(X, w) + b
    f_wb = sigmoid(z)
    epsilon = 1e-15
    f_wb = np.clip(f_wb, epsilon, 1 - epsilon)
    loss = - (y * np.log(f_wb) + (1 - y) * np.log(1 - f_wb))
    cost = np.sum(loss) / m
    reg_cost = (lambda_ / (2 * m)) * np.sum(w ** 2)
    return float(cost + reg_cost)


def compute_logistic_gradient(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lambda_: float = 0.0) -> Tuple[np.ndarray, float]:
    """Compute analytical gradients for logistic regression."""
    m = X.shape[0]
    f_wb = sigmoid(np.dot(X, w) + b)
    errors = f_wb - y
    dj_dw = (np.dot(X.T, errors) / m) + (lambda_ / m) * w
    dj_db = float(np.sum(errors) / m)
    return dj_dw, dj_db
