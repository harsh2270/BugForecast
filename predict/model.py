"""Logistic hazard model in pure numpy (L2-regularized, full-batch GD)."""
from __future__ import annotations

import json
import numpy as np

from .features import FEATURE_NAMES


class HazardModel:
    def __init__(self, lr: float = 0.1, epochs: int = 400, l2: float = 1e-3):
        self.lr, self.epochs, self.l2 = lr, epochs, l2
        self.w: np.ndarray | None = None
        self.b = 0.0
        self.mu: np.ndarray | None = None
        self.sd: np.ndarray | None = None

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HazardModel":
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-9
        Z = (X - self.mu) / self.sd
        n, d = Z.shape
        self.w = np.zeros(d)
        self.b = 0.0
        pos_w = max(1.0, (len(y) - y.sum()) / max(1.0, y.sum()))  # class imbalance
        for _ in range(self.epochs):
            p = self._sigmoid(Z @ self.w + self.b)
            sample_w = np.where(y == 1, pos_w, 1.0)
            grad = (Z * ((p - y) * sample_w)[:, None]).sum(axis=0) / n + self.l2 * self.w
            gb = ((p - y) * sample_w).sum() / n
            self.w -= self.lr * grad
            self.b -= self.lr * gb
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.w is not None and self.mu is not None and self.sd is not None
        Z = (np.asarray(X, dtype=float) - self.mu) / self.sd
        return self._sigmoid(Z @ self.w + self.b)

    def explain(self, x) -> list[tuple[str, float]]:
        """Per-feature contribution to the logit for a single sample."""
        assert self.w is not None and self.mu is not None and self.sd is not None
        x = np.asarray(x, dtype=float)
        z = (x - self.mu) / self.sd
        contribs = z * self.w
        order = np.argsort(-np.abs(contribs))
        return [(FEATURE_NAMES[i], float(contribs[i])) for i in order]

    def save(self, path: str) -> None:
        data = {
            "w": self.w.tolist(), "b": self.b,
            "mu": self.mu.tolist(), "sd": self.sd.tolist(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "HazardModel":
        m = cls()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        m.w = np.asarray(data["w"])
        m.b = data["b"]
        m.mu = np.asarray(data["mu"])
        m.sd = np.asarray(data["sd"])
        return m
