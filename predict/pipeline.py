"""Shared analysis pipeline used by both the CLI and the web server.

Single source of truth: mine -> label -> train ensemble -> backtest ->
build report dict -> render HTML. This avoids the server and CLI drifting
apart (they previously had near-identical ~50-line copies).
"""
from __future__ import annotations

import numpy as np

from .ensemble import StackedEnsemble, BASE_NAMES
from .features import build_dataset, FEATURE_NAMES
from .analytics import (auc, prf, brier, recall_at_effort, file_heatmap,
                        author_table, weekly_risk_curve)
from .report import generate

MIN_COMMITS = 60
MIN_DEFECTS = 5


def _eval(y, p):
    prec, rec, f1 = prf(y, p)
    return {"auc": auc(y, p), "f1": f1, "precision": prec, "recall": rec,
            "brier": brier(y, p)}


def analyze_repo(repo: str, max_count: int = 5000, progress=None) -> dict:
    """Run the full analysis and return {"report": dict, "html": str}.

    `progress` is an optional callback(stage: str, pct: int).
    Raises ValueError with a user-friendly message for unsuitable repos.
    """
    if progress:
        progress("Mining git history (features + defect labels)...", 20)
    X, y, meta, commits = build_dataset(repo, max_count)
    if len(X) < MIN_COMMITS or y.sum() < MIN_DEFECTS:
        raise ValueError(
            f"Repo not suitable for analysis: {len(X)} usable commits, "
            f"{int(y.sum())} defect-linked (need at least {MIN_COMMITS} commits "
            f"and {MIN_DEFECTS} bug-fix style commits).")

    if progress:
        progress(f"Training 6-model ensemble on {len(X)} commits...", 45)
    ens = StackedEnsemble().fit(X, y)
    p_stack = ens.oof_proba()

    if progress:
        progress("Backtesting and building dashboard...", 80)
    models = [{"name": "stacked ensemble", **_eval(y, p_stack)}]
    for j, name in enumerate(BASE_NAMES):
        models.append({"name": name, **_eval(y, ens.oof[:, j])})
    models.sort(key=lambda m: -m["auc"])

    labels_by_hash = {m["hash"]: int(l) for m, l in zip(meta, y)}
    risky_idx = np.argsort(-p_stack)[:25]
    # logistic base learner's feature weights (length = n_features) — used for
    # model-derived per-commit contributions. ens.meta.w is over base MODELS,
    # not features, and must not be presented as feature importance.
    lw = ens.final["logistic"].w
    risky = []
    for i in risky_idx:
        xs = X[i]
        # model-derived contributions (weight x feature value); sorting
        # by raw |x| would be scale-dependent and misleading.
        contribs = sorted(zip(FEATURE_NAMES, (lw * xs).tolist()),
                          key=lambda t: -abs(t[1]))[:3]
        risky.append({"p": float(p_stack[i]), "hash": meta[i]["hash"],
                      "subject": meta[i]["subject"], "ts": meta[i]["ts"],
                      "drivers": [(n, float(v)) for n, v in contribs]})

    best_single = max(m["auc"] for m in models if m["name"] != "stacked ensemble")
    hist, _ = np.histogram(p_stack, bins=24, range=(0, 1))
    risk_dist = (hist / max(1, hist.max())).tolist()
    fw = sorted(zip(FEATURE_NAMES, lw.tolist()), key=lambda t: -abs(t[1]))[:8]

    report = {
        "repo": repo,
        "kpis": [
            (f"{auc(y, p_stack):.3f}", "ensemble AUC (backtest)"),
            (f"{recall_at_effort(y, p_stack, 0.2)*100:.0f}%",
             "defects caught @ 20% review effort"),
            (f"{int(y.sum())}", "defect-linked commits"),
            (f"{len(X)}", "commits analyzed"),
            (f"{len(set(f for c in commits for f in c.files))}", "files tracked"),
        ],
        "models": models,
        "risky": risky,
        "files": file_heatmap(commits, labels_by_hash),
        "authors": author_table(commits, labels_by_hash),
        "trend": weekly_risk_curve(meta, p_stack),
        "ensemble_gain": {
            "stacked": auc(y, p_stack), "best_single": best_single,
            "recall20": recall_at_effort(y, p_stack, 0.2),
            "baseline20": float(y.mean()),
        },
        "risk_dist": risk_dist,
        "top_features": [(n, float(v)) for n, v in fw],
    }
    return {"report": report, "html": generate(report)}


# Backward-compatible alias for the old name
build_and_report = analyze_repo
