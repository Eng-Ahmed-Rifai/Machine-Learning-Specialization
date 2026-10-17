"""Collaborative Filtering Cost Function and Gradients in NumPy.

Implements matrix factorization loss and gradients for personalized recommender systems.
"""

from typing import Tuple
import numpy as np


def cofi_cost_func(
    X: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    Y: np.ndarray,
    R: np.ndarray,
    lambda_: float = 0.0
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute collaborative filtering cost and analytical gradients.

    Args:
        X: Item feature matrix (num_items, num_features).
        W: User parameter matrix (num_users, num_features).
        b: User bias vector (1, num_users).
        Y: Target ratings matrix (num_items, num_users).
        R: Binary indicator matrix where R[i, j] == 1 if user j rated item i.
        lambda_: Regularization coefficient.
    """
    diff = (np.dot(X, W.T) + b - Y) * R
    cost = 0.5 * np.sum(diff ** 2)
    reg_cost = 0.5 * lambda_ * (np.sum(W ** 2) + np.sum(X ** 2))
    total_cost = float(cost + reg_cost)

    X_grad = np.dot(diff, W) + lambda_ * X
    W_grad = np.dot(diff.T, X) + lambda_ * W
    b_grad = np.sum(diff, axis=0)

    return total_cost, X_grad, W_grad, b_grad
