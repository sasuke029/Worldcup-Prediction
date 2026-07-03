"""
From-scratch binary logistic regression.

Implements the model from first principles using only NumPy:
- sigmoid activation
- binary cross-entropy loss (negative log-likelihood)
- analytically derived gradients
- batch gradient descent with optional L2 regularisation

See reports/model_report.html and the project write-up for the full
mathematical derivation.
"""

from __future__ import annotations
import numpy as np


class LogisticRegressionScratch:
    """Binary logistic regression classifier implemented from scratch.
    
    Uses batch gradient descent with optional L2 regularization.
    All computations are done with NumPy only (no sklearn).
    """
    
    def __init__(self, learning_rate: float = 0.1, n_iterations: int = 10_000,
                 tol: float = 1e-6, l2_lambda: float = 0.0, verbose: bool = False) -> None:
        """Initialize the logistic regression model.
        
        Args:
            learning_rate: Step size for gradient descent.
            n_iterations: Maximum number of training iterations.
            tol: Convergence tolerance on loss change.
            l2_lambda: L2 regularization coefficient.
            verbose: Whether to print convergence messages.
        """
        self.lr: float = learning_rate
        self.n_iterations: int = n_iterations
        self.tol: float = tol
        self.l2_lambda: float = l2_lambda
        self.verbose: bool = verbose
        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.loss_history: list[float] = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Sigmoid activation function with numerical stability."""
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _loss(self, y: np.ndarray, p: np.ndarray) -> float:
        """Binary cross-entropy loss with L2 regularization."""
        eps = 1e-12
        p = np.clip(p, eps, 1 - eps)
        ce = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        reg: float = (self.l2_lambda / (2 * len(y))) * np.sum(self.w ** 2)
        return ce + reg

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogisticRegressionScratch:
        """Train the model using batch gradient descent.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Binary target vector of shape (n_samples,).
            
        Returns:
            Self for chaining.
        """
        n_samples, n_features = X.shape
        self.w: np.ndarray = np.zeros(n_features)
        self.b = 0.0
        self.loss_history = []

        prev_loss: float = np.inf
        for i in range(self.n_iterations):
            z = X @ self.w + self.b
            p = self._sigmoid(z)

            error = p - y
            grad_w = (X.T @ error) / n_samples + (self.l2_lambda / n_samples) * self.w
            grad_b = np.mean(error)

            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

            loss = self._loss(y, p)
            self.loss_history.append(loss)
            if abs(prev_loss - loss) < self.tol:
                if self.verbose:
                    print(f"Converged at iteration {i}, loss={loss:.6f}")
                break
            prev_loss = loss
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of class 1.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features).
            
        Returns:
            Predicted probabilities of shape (n_samples,).
        """
        return self._sigmoid(X @ self.w + self.b)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class labels.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features).
            threshold: Classification threshold (default 0.5).
            
        Returns:
            Binary predictions of shape (n_samples,).
        """
        return (self.predict_proba(X) >= threshold).astype(int)
