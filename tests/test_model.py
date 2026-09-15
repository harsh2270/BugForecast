import numpy as np

from predict.features import FEATURE_NAMES, Commit, build_dataset
from predict.model import HazardModel


def test_model_learns_separable_data():
    rng = np.random.default_rng(0)
    n = 400
    X = rng.normal(size=(n, len(FEATURE_NAMES)))
    logits = 2.0 * X[:, 0] - 1.5 * X[:, 5]
    y = (logits + rng.normal(scale=0.3, size=n) > 0).astype(float)
    m = HazardModel(epochs=200).fit(X, y)
    p = m.predict_proba(X)
    acc = ((p > 0.5).astype(float) == y).mean()
    assert acc > 0.9, acc
    assert m.w[0] > 0 and m.w[5] < 0


def test_explain_ranks_by_contribution():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, len(FEATURE_NAMES)))
    y = (X[:, 2] > 0).astype(float)
    m = HazardModel(epochs=150).fit(X, y)
    expl = m.explain(X[0])
    assert expl[0][0] == "nfiles"


def test_commit_is_fix_heuristic():
    c = Commit("h", "a", 0, "fix: null pointer crash")
    assert c.is_fix
    c2 = Commit("h", "a", 0, "merge branch 'dev'")
    assert not c2.is_fix


def test_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(2)
    X = rng.normal(size=(100, len(FEATURE_NAMES)))
    y = (X[:, 0] > 0).astype(float)
    m = HazardModel(epochs=50).fit(X, y)
    p = tmp_path / "m.json"
    m.save(str(p))
    m2 = HazardModel.load(str(p))
    assert np.allclose(m.predict_proba(X), m2.predict_proba(X))
