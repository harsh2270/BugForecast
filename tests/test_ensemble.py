import numpy as np

from predict.features import FEATURE_NAMES, Commit
from predict.ensemble import (StackedEnsemble, BASE_NAMES, Logistic, DecisionTree,
                              RandomForest, GradientBoostedStumps, GaussianNB, MLP)
from predict.analytics import auc, prf, brier, recall_at_effort, file_heatmap, author_table
from predict.report import generate


def _synth(n=500, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(FEATURE_NAMES)))
    logits = 2.0 * X[:, 0] - 1.5 * X[:, 5] + 0.8 * np.abs(X[:, 2])
    y = (logits + rng.normal(scale=0.3, size=n) > 0).astype(float)
    return X, y


def test_each_base_learner_learns():
    X, y = _synth()
    for cls in (Logistic, DecisionTree, RandomForest,
                GradientBoostedStumps, GaussianNB, MLP):
        m = cls().fit(X, y)
        p = m.predict_proba(X)
        acc = ((p > 0.5).astype(float) == y).mean()
        assert acc > 0.8, f"{cls.__name__} acc={acc:.3f}"


def test_stacked_ensemble_oof_beats_chance():
    X, y = _synth()
    ens = StackedEnsemble(n_folds=4).fit(X, y)
    p = ens.oof_proba()
    assert len(p) == len(y)
    assert auc(y, p) > 0.85  # temporal backtest is harder than random CV


def test_ensemble_output_probabilities_valid():
    X, y = _synth(300, 5)
    ens = StackedEnsemble(n_folds=3).fit(X, y)
    p = ens.predict_proba(X[:20])
    assert ((p >= 0) & (p <= 1)).all()


def test_commit_is_fix_heuristic():
    assert Commit("h", "a", 0, "fix: null pointer crash").is_fix
    assert not Commit("h", "a", 0, "merge branch 'dev'").is_fix


def test_analytics():
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0] * 5, float)
    rng = np.random.default_rng(3)
    p = y * 0.8 + 0.1 + rng.uniform(0, 0.05, len(y))  # distinct scores, no ties
    assert auc(y, p) > 0.95
    prec, rec, f1 = prf(y, p, 0.5)
    assert prec > 0.9 and rec > 0.9
    assert brier(y, p) < 0.1
    assert recall_at_effort(y, p, 0.2) > 0.35  # cap is 0.4: 20% effort, 50% positives


def test_heatmap_and_authors():
    commits = [Commit("h1", "alice", 100, "a", files=["f.py"]),
               Commit("h2", "bob", 200, "b", files=["f.py", "g.py"])]
    labels = {"h1": 1, "h2": 0}
    files = file_heatmap(commits, labels)
    assert files[0]["file"] == "f.py" and files[0]["defects"] == 1
    authors = author_table(commits, labels)
    assert {a["author"] for a in authors} == {"alice", "bob"}


def test_dashboard_renders():
    report = {
        "repo": "demo/repo",
        "kpis": [("0.85", "AUC"), ("70%", "caught")],
        "models": [{"name": "stacked", "auc": 0.85, "f1": 0.6, "brier": 0.15},
                   {"name": "gbdt", "auc": 0.80, "f1": 0.5, "brier": 0.18}],
        "risky": [{"p": 0.9, "hash": "abc123", "subject": "big refactor",
                    "ts": 1700000000, "drivers": [("la", 5.2)]}],
        "files": [{"file": "src/a.py", "commits": 10, "defects": 3, "risk": 0.3}],
        "authors": [{"author": "alice", "commits": 10, "defects": 2, "risk": 0.2}],
        "trend": [{"t": 1, "risk": 0.1, "n": 5}, {"t": 2, "risk": 0.3, "n": 5}],
        "ensemble_gain": {"stacked": 0.85, "best_single": 0.80,
                           "recall20": 0.7, "baseline20": 0.3},
        "risk_dist": [0.9, 0.6, 0.3, 0.1],
        "top_features": [("la", 1.2), ("file_hot", -0.4)],
    }
    html = generate(report)
    assert "BugForecast" in html and "stacked" in html and "<svg" in html
    assert "tabpane" in html and "cfilt" in html  # tabbed UI + filters
