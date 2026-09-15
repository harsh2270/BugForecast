# BugForecast

**Predictive defect-risk engine for git repositories.** BugForecast mines a
repository's git history, learns which commit patterns historically preceded
bugs, and scores every commit — and the files it touched — with a probability
of being defect-linked, using a six-model machine-learning ensemble.

## The problem it solves

Code review effort is finite, but commits are not. Most review time is spent
on changes that will never cause problems, while the risky ones slip through.
BugForecast answers one practical question: *given limited review time, which
commits and files should a team look at first?*

## What it is NOT

- It is **not** a static analyzer — it does not read your code for bugs.
- It does **not** predict with certainty. It outputs probabilities from
  historical patterns; every metric is a statistical estimate, not a guarantee.
- It is **not** a general-purpose security tool.

## How it works

1. **Mining** — parses `git log --numstat` and computes 11 features per
   commit: added/deleted lines, files touched, repo-relative change size,
   add/delete balance, author experience, author-file familiarity, file
   hotness, file age, and time-of-day (encoded as sin/cos).
2. **Labeling (SZZ-style)** — a commit is labeled defective (1) if a later
   fix-style commit ("fix", "bug", "hotfix", etc. in the subject) touched any
   of the same files within 14 days. Fix commits themselves are excluded from
   the training rows so the model cannot cheat by recognizing fix subjects.
3. **Ensemble** — six base learners, all implemented in pure numpy:
   logistic regression (L2), CART decision tree, random forest,
   gradient-boosted stumps, Gaussian naive Bayes, and a one-hidden-layer MLP.
   A second-level logistic **meta-learner** stacks their out-of-fold
   predictions.
4. **Time-safe evaluation** — rolling-origin folds: each fold trains only on
   the past and tests on the next block. Every metric shown is honest
   out-of-sample; no future data leaks into training. This matters because a
   randomly shuffled split would let the model peek forward in time and
   inflate scores on history that repeats.
5. **Dashboard** — a single self-contained dark-theme HTML file: KPI cards,
   model comparison, riskiest commits with their top risk drivers, per-file
   heatmap, author risk table, risk distributions, and the meta-model's
   learned feature weights for that repository.

## The headline metric

**"Defects caught at 20% review effort"** — if reviewers inspect only the
riskiest 20% of commits, what fraction of the defect-linked commits do they
intercept? Chance-level performance is the base defect rate itself (e.g. if
15% of commits are defect-linked, a random reviewer catches 15%). Read the
number against that baseline, not against 100%.

## Installation

```bash
pip install -r requirements.txt   # numpy, flask
```

Requires Python 3.10+ and `git` on PATH.

## One-command startup

```bash
python -m predict web
```

The browser opens automatically. Paste a public GitHub repository URL (query
strings and `/tree/...` paths are fine), watch the live progress stages, and
the dashboard opens when done. Demo chips on the welcome screen launch
pre-picked repositories.

## CLI usage

```bash
python -m predict train  --repo path/to/repo   # backtest table
python -m predict score  --repo path/to/repo   # top-10 riskiest commits
python -m predict report --repo path/to/repo --out report.html
python scripts/make_demo_repo.py               # synthetic demo repository
```

A repository needs at least 60 usable commits and 5 fix-style commits to
train; unsuitable repos fail with a clear explanation instead of a garbage
model.

## Interpreting the metrics

| Metric | Meaning | Caution |
|---|---|---|
| AUC | probability a random defective commit ranks above a random clean one; 0.5 = chance | 0.6–0.7 is normal and still useful for prioritization |
| F1 / precision / recall | balance of false alarms vs misses at a fixed threshold | depends on class imbalance |
| Brier | mean squared error of probabilities; lower is better | measures calibration, not ranking |
| recall@20% | headline review-effort metric | compare against the base defect rate |

If the model performs poorly on a repository (AUC near 0.5), the dashboard
shows that honestly — some repositories simply don't have learnable defect
patterns in their history.

## Known limitations

- **SZZ labels are approximate.** Matching by file co-touching overestimates
  defects (not every later fix traces back to the flagged commit) and misses
  cross-file causes.
- **Fix-commit detection is a subject-line heuristic**, which under- or
  over-counts depending on the repo's commit-message conventions.
- Merge commits, binary files, and renames are handled but imperfectly:
  merges with empty numstat are skipped; binary deltas count as touches
  without line changes.
- Metrics come from a temporal backtest on one repository; they do not
  transfer to a guarantee about future commits.
- Small repositories, mono-message histories, or squashed-merge workflows
  produce weak labels and weak models.
- The web server binds to 127.0.0.1 and is intended for local use; it is not
  hardened for public deployment.

## Project layout

```
predict/features.py    git mining + SZZ labeling
predict/ensemble.py    6 numpy models + out-of-fold stacking
predict/analytics.py   AUC / F1 / Brier / recall@effort, heatmap, trends
predict/pipeline.py    shared analysis pipeline (CLI + web)
predict/report.py      HTML dashboard generator
predict/server.py      Flask app, job queue, URL validation
predict/__main__.py    CLI entry point
scripts/               demo repo generator, Word report generator
tests/                 pytest suite
```
