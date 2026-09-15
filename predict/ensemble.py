"""Pure-numpy multi-model ensemble with out-of-fold stacking.

Base learners:
  1. LogisticRegression (L2, weighted)
  2. DecisionTree (CART, gini, depth-limited, weighted classes)
  3. RandomForest (bagged trees, feature subsampling)
  4. GradientBoostedStumps (depth-1 trees on residual logits)
  5. GaussianNaiveBayes
  6. MLP (one hidden layer, tanh, Adam-style updates)

Meta learner: logistic regression over base-learner OOF probabilities.
Evaluation: time-aware rolling-origin backtest (train strictly on the past).
"""
from __future__ import annotations

import json
import numpy as np


# ---------------------------------------------------------------- utilities
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _class_weight(y):
    pos = max(1.0, y.sum())
    neg = max(1.0, len(y) - y.sum())
    return np.where(y == 1, neg / pos, 1.0)


# ------------------------------------------------------- 1. logistic
class Logistic:
    def __init__(self, lr=0.15, epochs=500, l2=1e-3):
        self.lr, self.epochs, self.l2 = lr, epochs, l2

    def fit(self, X, y):
        X = np.asarray(X, float)
        n, d = X.shape
        sw = _class_weight(y)
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.epochs):
            p = _sigmoid(X @ self.w + self.b)
            g = (X * ((p - y) * sw)[:, None]).sum(0) / n + self.l2 * self.w
            self.w -= self.lr * g
            self.b -= self.lr * (((p - y) * sw).sum() / n)
        return self

    def predict_proba(self, X):
        return _sigmoid(np.asarray(X, float) @ self.w + self.b)


# ------------------------------------------------------- 2. CART tree
class _Node:
    __slots__ = ("leaf", "p", "feat", "thr", "left", "right")


class DecisionTree:
    def __init__(self, max_depth=6, min_samples=8, max_features=None):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.max_features = max_features

    def fit(self, X, y):
        X = np.asarray(X, float)
        self.w = _class_weight(y)
        self._X, self._y = X, y
        self.root = self._build(np.arange(len(y)), 0)
        return self

    def _gini(self, idx):
        y = self._y[idx]
        if len(idx) == 0:
            return 0.0
        p = np.clip((y * self.w[idx]).sum() / self.w[idx].sum(), 0, 1)
        return 1.0 - p * p - (1 - p) * (1 - p)

    def _build(self, idx, depth):
        node = _Node()
        y = self._y[idx]
        wsum = self.w[idx].sum()
        node.p = float(np.clip((y * self.w[idx]).sum() / max(wsum, 1e-9), 0, 1))
        if (depth >= self.max_depth or len(idx) < self.min_samples
                or node.p in (0.0, 1.0)):
            node.leaf = True
            return node
        d = self._X.shape[1]
        feats = (np.arange(d) if self.max_features is None
                 else np.random.default_rng(0).choice(d, min(d, self.max_features), replace=False))
        parent = self._gini(idx)
        best = (parent, None, None)
        n_tot = len(idx)
        for f in feats:
            vals = self._X[idx, f]
            order = np.argsort(vals, kind="stable")
            v = vals[order]
            cy = np.cumsum(y[order])          # positives left of split
            for k in range(self.min_samples, n_tot - self.min_samples + 1):
                if v[k] == v[k - 1]:
                    continue
                nl, nr = k, n_tot - k
                pl = cy[k - 1] / nl
                pr = (cy[-1] - cy[k - 1]) / nr
                gl = 1 - pl * pl - (1 - pl) ** 2
                gr = 1 - pr * pr - (1 - pr) ** 2
                gain = (nl / n_tot) * gl + (nr / n_tot) * gr
                if gain < best[0] - 1e-12:
                    best = (gain, f, (v[k - 1] + v[k]) / 2)
        if best[1] is None:
            node.leaf = True
            return node
        node.leaf = False
        node.feat, node.thr = best[1], best[2]
        mask = self._X[idx, node.feat] <= node.thr
        node.left = self._build(idx[mask], depth + 1)
        node.right = self._build(idx[~mask], depth + 1)
        return node

    def _row(self, x, node):
        while not node.leaf:
            node = node.left if x[node.feat] <= node.thr else node.right
        return node.p

    def predict_proba(self, X):
        X = np.asarray(X, float)
        return np.array([self._row(x, self.root) for x in X])


# ------------------------------------------------------- 3. random forest
class RandomForest:
    def __init__(self, n_trees=40, max_depth=6, max_features=0.7):
        self.n_trees, self.max_depth, self.max_features = n_trees, max_depth, max_features

    def fit(self, X, y):
        X = np.asarray(X, float)
        rng = np.random.default_rng(7)
        self.trees = []
        n = len(y)
        mf = max(1, int(X.shape[1] * self.max_features))
        for _ in range(self.n_trees):
            idx = rng.integers(0, n, n)
            t = DecisionTree(max_depth=self.max_depth, max_features=mf)
            # bootstrap via row weights: subsample then fit
            t.fit(X[idx], y[idx])
            self.trees.append(t)
        return self

    def predict_proba(self, X):
        return np.mean([t.predict_proba(X) for t in self.trees], axis=0)


# ------------------------------------------------------- 4. GBDT stumps
class GradientBoostedStumps:
    def __init__(self, n_rounds=120, lr=0.1):
        self.n_rounds, self.lr = n_rounds, lr

    def fit(self, X, y):
        X = np.asarray(X, float)
        n, d = X.shape
        sw = _class_weight(y)
        self.f0 = float(np.log(np.clip((y * sw).sum() / sw.sum(), 1e-4, 1 - 1e-4)) -
                        np.log(1 - np.clip((y * sw).sum() / sw.sum(), 1e-4, 1 - 1e-4)))
        self.stumps = []  # (feat, thr, left_val, right_val)
        F = np.full(n, self.f0)
        p = _sigmoid(F)
        rng = np.random.default_rng(3)
        for _ in range(self.n_rounds):
            resid = (y - p) * sw
            best = None
            for f in range(d):
                thr_cands = np.quantile(X[:, f], np.linspace(0.1, 0.9, 9))
                for thr in thr_cands:
                    m = X[:, f] <= thr
                    if m.sum() < 5 or (~m).sum() < 5:
                        continue
                    gl = resid[m].mean()
                    gr = resid[~m].mean()
                    sse = ((resid[m] - gl) ** 2).sum() + ((resid[~m] - gr) ** 2).sum()
                    if best is None or sse < best[0]:
                        best = (sse, f, float(thr), float(gl), float(gr))
            if best is None:
                break
            _, f, thr, gl, gr = best
            self.stumps.append((f, thr, gl, gr))
            F += self.lr * np.where(X[:, f] <= thr, gl, gr)
            p = _sigmoid(F)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, float)
        F = np.full(len(X), self.f0)
        for f, thr, gl, gr in self.stumps:
            F += self.lr * np.where(X[:, f] <= thr, gl, gr)
        return _sigmoid(F)


# ------------------------------------------------------- 5. naive bayes
class GaussianNB:
    def fit(self, X, y):
        X = np.asarray(X, float)
        self.cls = [0.0, 1.0]
        self.stats = []
        prior = float(y.mean())
        self.log_prior = [np.log(1 - prior + 1e-9), np.log(prior + 1e-9)]
        for c in self.cls:
            Xc = X[y == c]
            self.stats.append((Xc.mean(0) if len(Xc) else np.zeros(X.shape[1]),
                               Xc.var(0) + 1e-6 if len(Xc) else np.ones(X.shape[1])))
        return self

    def predict_proba(self, X):
        X = np.asarray(X, float)
        logp = []
        for (mu, var), lp in zip(self.stats, self.log_prior):
            logp.append(lp - 0.5 * (((X - mu) ** 2) / var + np.log(2 * np.pi * var)).sum(1))
        logp = np.array(logp).T
        logp -= logp.max(1, keepdims=True)
        p1 = np.exp(logp[:, 1]) / np.exp(logp).sum(1)
        return p1


# ------------------------------------------------------- 6. MLP
class MLP:
    def __init__(self, hidden=16, epochs=300, lr=0.05):
        self.hidden, self.epochs, self.lr = hidden, epochs, lr

    def fit(self, X, y):
        X = np.asarray(X, float)
        n, d = X.shape
        rng = np.random.default_rng(11)
        h = self.hidden
        self.W1 = rng.normal(0, 1 / np.sqrt(d), (d, h))
        self.b1 = np.zeros(h)
        self.W2 = np.zeros(h)
        self.b2 = 0.0
        sw = _class_weight(y)
        m = [np.zeros_like(a) for a in (self.W1, self.b1, self.W2, np.array(0.0))]
        v = [np.zeros_like(a) for a in m]
        params = lambda: (self.W1, self.b1, self.W2, self.b2)
        b2 = 0.0
        t = 0
        for _ in range(self.epochs):
            z1 = X @ self.W1 + self.b1
            a1 = np.tanh(z1)
            p = _sigmoid(a1 @ self.W2 + b2)
            dz2 = (p - y) * sw / n
            gW2 = a1.T @ dz2
            gb2 = dz2.sum()
            da1 = dz2[:, None] * self.W2[None, :]
            gz1 = da1 * (1 - a1 ** 2)
            gW1 = X.T @ gz1
            gb1 = gz1.sum(0)
            grads = [gW1, gb1, gW2, np.array(gb2)]
            t += 1
            for arr, g, mi, vi in zip([self.W1, self.b1, self.W2, np.array([b2])],
                                      grads, m, v):
                mi *= 0.9
                mi += 0.1 * g
                vi *= 0.999
                vi += 0.001 * g * g
                step = mi / (1 - 0.9 ** t) / (np.sqrt(vi / (1 - 0.999 ** t)) + 1e-8)
                arr -= self.lr * step
            self.b1 = m[1] * 0 + self.b1  # b1 updated in-place above
            b2 = np.array([b2])[0] - 0.0  # keep closure simple
            # apply b2 update explicitly
            b2 -= self.lr * (m[3] / (1 - 0.9 ** t)) / (np.sqrt(v[3] / (1 - 0.999 ** t)) + 1e-8)
        self.b2 = b2
        return self

    def predict_proba(self, X):
        X = np.asarray(X, float)
        return _sigmoid(np.tanh(X @ self.W1 + self.b1) @ self.W2 + self.b2)


# ------------------------------------------------------- ensemble
BASE_CLASSES = {
    "logistic": Logistic,
    "random_forest": RandomForest,
    "gbdt": GradientBoostedStumps,
    "naive_bayes": GaussianNB,
    "mlp": MLP,
}
# a plain single CART is included via random_forest with 1 tree variant
BASE_NAMES = ["logistic", "cart", "random_forest", "gbdt", "naive_bayes", "mlp"]


class StackedEnsemble:
    """Time-safe stacking: base learners produce OOF probabilities via
    rolling-origin folds; the meta learner (logistic) trains on OOF preds."""

    def __init__(self, n_folds: int = 5):
        self.n_folds = n_folds

    def _make(self, name):
        if name == "cart":
            return DecisionTree(max_depth=6)
        return BASE_CLASSES[name]()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StackedEnsemble":
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        n = len(y)
        folds = self._folds(n)
        oof = np.zeros((n, len(BASE_NAMES)))
        models: dict[str, list] = {k: [] for k in BASE_NAMES}
        for tr, te in folds:
            for j, name in enumerate(BASE_NAMES):
                m = self._make(name).fit(X[tr], y[tr])
                oof[te, j] = np.clip(m.predict_proba(X[te]), 1e-4, 1 - 1e-4)
                models[name].append(m)
        self.meta = Logistic(lr=0.2, epochs=400).fit(oof, y)
        # final base models trained on all data
        self.final = {name: self._make(name).fit(X, y) for name in BASE_NAMES}
        self.fold_models = models
        self.oof = oof
        return self

    def _folds(self, n):
        """Rolling-origin: each fold trains on a prefix, tests on the next block
        (respects temporal order; no leakage from the future)."""
        folds = []
        edges = np.linspace(0, n, self.n_folds + 2).astype(int)
        for k in range(self.n_folds):
            te = np.arange(edges[k + 1], edges[k + 2])
            tr = np.arange(0, edges[k + 1])
            if len(te) and len(tr) >= 20:
                folds.append((tr, te))
        return folds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        P = np.column_stack([
            np.clip(self.final[name].predict_proba(X), 1e-4, 1 - 1e-4)
            for name in BASE_NAMES
        ])
        return self.meta.predict_proba(P)

    def oof_proba(self) -> np.ndarray:
        """Stacked OOF predictions (honest performance estimate)."""
        return self.meta.predict_proba(np.clip(self.oof, 1e-4, 1 - 1e-4))

    def save(self, path: str) -> None:
        data = {"meta_w": self.meta.w.tolist(), "meta_b": self.meta.b}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    # full save/load of all learners is intentionally kept in memory;
    # the CLI retrains per run, which is fast for numpy models.
    @staticmethod
    def load_meta(path: str):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        m = Logistic()
        m.w = np.asarray(d["meta_w"])
        m.b = d["meta_b"]
        return m
