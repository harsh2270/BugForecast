"""Evaluation metrics and per-file risk analytics."""
from __future__ import annotations

import math
import numpy as np


def confusion(y, p, thr=0.5):
    yb = (y == 1).astype(int)
    pb = (p >= thr).astype(int)
    tp = int(((yb == 1) & (pb == 1)).sum())
    fp = int(((yb == 0) & (pb == 1)).sum())
    fn = int(((yb == 1) & (pb == 0)).sum())
    tn = int(((yb == 0) & (pb == 0)).sum())
    return tp, fp, fn, tn


def prf(y, p, thr=0.5):
    tp, fp, fn, _ = confusion(y, p, thr)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return prec, rec, f1


def auc(y, p):
    order = np.argsort(p)
    ranks = np.empty(len(p), float)
    ranks[order] = np.arange(1, len(p) + 1)
    # average ranks for ties
    sp = np.asarray(p, float)
    for v in np.unique(sp):
        m = sp == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    n1 = int((y == 1).sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def brier(y, p):
    return float(((np.asarray(p) - np.asarray(y)) ** 2).mean())


def recall_at_effort(y, p, fraction=0.2):
    """If a reviewer inspects only the riskiest `fraction` of commits,
    what fraction of all defects do they catch? The headline metric."""
    k = max(1, int(len(y) * fraction))
    idx = np.argsort(-np.asarray(p))[:k]
    return float((np.asarray(y)[idx] == 1).sum() / max(1, (np.asarray(y) == 1).sum()))


def file_heatmap(commits, labels_by_hash, top=15):
    """Per-file defect density: how often changes to a file precede a fix."""
    stats: dict[str, dict] = {}
    for c in commits:
        y = labels_by_hash.get(c.hash, 0)
        for f in set(c.files):
            s = stats.setdefault(f, {"commits": 0, "defective": 0})
            s["commits"] += 1
            s["defective"] += int(y)
    rows = []
    for f, s in stats.items():
        rows.append({
            "file": f,
            "commits": s["commits"],
            "defects": s["defective"],
            "risk": s["defective"] / s["commits"],
        })
    rows.sort(key=lambda r: (-r["risk"] * math.log1p(r["commits"]), -r["commits"]))
    return rows[:top]


def author_table(commits, labels_by_hash, top=10):
    stats: dict[str, dict] = {}
    for c in commits:
        s = stats.setdefault(c.author, {"commits": 0, "defective": 0})
        s["commits"] += 1
        s["defective"] += int(labels_by_hash.get(c.hash, 0))
    rows = [{"author": a, "commits": s["commits"], "defects": s["defective"],
             "risk": s["defective"] / s["commits"]} for a, s in stats.items()]
    rows.sort(key=lambda r: -r["commits"])
    return rows[:top]


def weekly_risk_curve(meta, probs, bins=26):
    """Average predicted risk over time — shows whether risk is trending up."""
    if not meta:
        return []
    ts = [m["ts"] for m in meta]
    t0, t1 = min(ts), max(ts)
    width = max(1, (t1 - t0) // bins)
    buckets: dict[int, list] = {}
    for m, p in zip(meta, probs):
        buckets.setdefault((m["ts"] - t0) // width, []).append(p)
    out = []
    for b in sorted(buckets):
        out.append({
            "t": t0 + (b + 0.5) * width,
            "risk": float(np.mean(buckets[b])),
            "n": len(buckets[b]),
        })
    return out
