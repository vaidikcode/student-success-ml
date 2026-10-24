"""
NumPy foundations — everything here is built with plain NumPy, no scikit-learn.

    * array creation, indexing, slicing, reshaping
    * broadcasting & vectorization (and how much faster it is than loops)
    * matrix ops: dot, transpose, inverse  -> linear regression via the Normal Equation
    * statistics: mean, std, variance, correlation
    * random shuffling for a train/test split
    * normalization & standardization
    * a tiny neural network (MLP) with hand-written backpropagation
"""
import time

import numpy as np


# ----------------------------------------------------------------------------
# 1. Array basics
# ----------------------------------------------------------------------------
def array_basics_demo() -> None:
    print("\n=== NumPy: array basics ===")
    a = np.arange(12)                       # creation
    m = a.reshape(3, 4)                     # reshaping
    print("arange(12).reshape(3, 4):\n", m)
    print("row 1:", m[1], "| column 2:", m[:, 2], "| bottom-right 2x2:\n", m[1:, 2:])
    print("even numbers via boolean mask:", a[a % 2 == 0])
    print("zeros/ones/linspace:", np.zeros(3), np.ones(3), np.linspace(0, 1, 5))
    print("flatten back:", m.ravel(), "| transposed shape:", m.T.shape)


# ----------------------------------------------------------------------------
# 2. Broadcasting & vectorization
# ----------------------------------------------------------------------------
def vectorization_demo(n: int = 1_000_000) -> None:
    print("\n=== NumPy: loops vs vectorization ===")
    x = np.random.default_rng(0).random(n)

    t0 = time.perf_counter()
    out_loop = [v * 2 + 1 for v in x]
    t_loop = time.perf_counter() - t0

    t0 = time.perf_counter()
    out_vec = x * 2 + 1                      # one line, runs in C
    t_vec = time.perf_counter() - t0

    assert np.allclose(out_loop, out_vec)
    print(f"python loop: {t_loop*1000:.1f} ms | vectorized: {t_vec*1000:.2f} ms | speed-up: {t_loop/t_vec:.0f}x")

    # Broadcasting: a (3,4) matrix minus a (4,) row vector -> the row is "stretched" over every row
    m = np.arange(12, dtype=float).reshape(3, 4)
    print("matrix - column means (broadcasting):\n", m - m.mean(axis=0))


# ----------------------------------------------------------------------------
# 3. Statistics
# ----------------------------------------------------------------------------
def describe(X: np.ndarray, names: list[str]) -> None:
    print("\n=== NumPy: statistics ===")
    print(f"{'feature':<22}{'mean':>9}{'std':>9}{'var':>10}{'min':>8}{'max':>8}")
    for i, name in enumerate(names):
        col = X[:, i]
        print(f"{name:<22}{col.mean():>9.2f}{col.std():>9.2f}{col.var():>10.2f}{col.min():>8.1f}{col.max():>8.1f}")


def correlation_with_target(X: np.ndarray, y: np.ndarray, names: list[str]) -> dict[str, float]:
    corr = {name: float(np.corrcoef(X[:, i], y)[0, 1]) for i, name in enumerate(names)}
    print("\nPearson correlation with final_score (np.corrcoef):")
    for name, r in sorted(corr.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {name:<25}{r:+.3f}  {'#' * int(abs(r) * 40)}")
    return corr


# ----------------------------------------------------------------------------
# 4. Train/test split, normalization, standardization
# ----------------------------------------------------------------------------
def train_test_split_np(X, y, test_size=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))            # shuffle indices
    cut = int(len(X) * (1 - test_size))
    return X[idx[:cut]], X[idx[cut:]], y[idx[:cut]], y[idx[cut:]]


def standardize(train, test):
    """z = (x - mean) / std, using ONLY training statistics (no data leakage)."""
    mu, sigma = train.mean(axis=0), train.std(axis=0)
    sigma[sigma == 0] = 1
    return (train - mu) / sigma, (test - mu) / sigma


def min_max_normalize(train, test):
    """x' = (x - min) / (max - min) -> squashes every feature into [0, 1]."""
    lo, hi = train.min(axis=0), train.max(axis=0)
    rng = np.where(hi - lo == 0, 1, hi - lo)
    return (train - lo) / rng, (test - lo) / rng


# ----------------------------------------------------------------------------
# 5. Linear regression with the Normal Equation: w = (XᵀX)⁻¹ Xᵀy
# ----------------------------------------------------------------------------
def normal_equation(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    Xb = np.c_[np.ones(len(X)), X]                       # add bias column
    return np.linalg.inv(Xb.T @ Xb) @ Xb.T @ y            # transpose, dot product, inverse


def predict_linear(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.c_[np.ones(len(X)), X] @ w


# ----------------------------------------------------------------------------
# 6. A neural network from scratch
# ----------------------------------------------------------------------------
class NumpyMLP:
    """
    input -> [Dense + ReLU] -> [Dense + ReLU] -> [Dense + Sigmoid] -> P(pass)

    Trained with mini-batch gradient descent and hand-written backpropagation.
    Loss: binary cross-entropy (+ L2 regularization, same idea as sklearn's `alpha`).
    """

    def __init__(self, layer_sizes=(16, 8), lr=0.05, alpha=1e-4, epochs=200, batch_size=32, seed=0):
        self.layer_sizes, self.lr, self.alpha = layer_sizes, lr, alpha
        self.epochs, self.batch_size = epochs, batch_size
        self.rng = np.random.default_rng(seed)
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    # --- activations ---
    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def _init(self, n_in):
        sizes = [n_in, *self.layer_sizes, 1]
        # He initialization: keeps activations from exploding/vanishing with ReLU
        self.W = [self.rng.normal(0, np.sqrt(2 / a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]

    def _forward(self, X):
        acts, zs = [X], []
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = acts[-1] @ W + b
            zs.append(z)
            acts.append(self.sigmoid(z) if i == len(self.W) - 1 else self.relu(z))
        return acts, zs

    def _loss(self, p, y):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        bce = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        return bce + self.alpha / 2 * sum((W ** 2).sum() for W in self.W) / len(y)

    def _backward(self, acts, zs, y):
        m = len(y)
        delta = acts[-1] - y.reshape(-1, 1)             # dL/dz for sigmoid + BCE simplifies nicely
        for i in reversed(range(len(self.W))):
            dW = acts[i].T @ delta / m + self.alpha * self.W[i] / m
            db = delta.mean(axis=0)
            if i > 0:
                delta = (delta @ self.W[i].T) * (zs[i - 1] > 0)   # chain rule through ReLU
            self.W[i] -= self.lr * dW                   # gradient descent step
            self.b[i] -= self.lr * db

    def fit(self, X, y, X_val=None, y_val=None):
        self._init(X.shape[1])
        for _ in range(self.epochs):
            idx = self.rng.permutation(len(X))          # shuffle every epoch
            for start in range(0, len(X), self.batch_size):
                batch = idx[start:start + self.batch_size]
                acts, zs = self._forward(X[batch])
                self._backward(acts, zs, y[batch])
            self._log(X, y, "train")
            if X_val is not None:
                self._log(X_val, y_val, "val")
        return self

    def _log(self, X, y, split):
        p = self.predict_proba(X)
        self.history[f"{split}_loss"].append(self._loss(p, y))
        self.history[f"{split}_acc"].append(((p >= 0.5) == y).mean())

    def predict_proba(self, X):
        return self._forward(X)[0][-1].ravel()

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
