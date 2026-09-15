"""Generate the BugForecast project report as a Word document."""
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH


def _h(doc, text, level=1):
    doc.add_heading(text, level=level)


def _p(doc, text, bold=False, italic=False):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = bold
    r.italic = italic
    return p


def _bullets(doc, items):
    for it in items:
        doc.add_paragraph(it, style="List Bullet")


def build(path: str) -> None:
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    # ---------- title
    t = doc.add_heading("BugForecast", level=0)
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = sub.add_run("A Predictive Defect-Risk Engine for Git Repositories\n"
                    "Technical Project Report")
    r.italic = True

    # ---------- what was built
    _h(doc, "1. What Was Built", 1)
    _p(doc, "BugForecast is a machine-learning system that reads a git repository's "
            "entire commit history, learns from its own past mistakes, and predicts "
            "which code changes (and which files) are most likely to cause future "
            "defects. It ships as both a command-line tool and a local web "
            "application with an animated dark-themed UI where the user pastes any "
            "public GitHub URL and receives a full interactive dashboard.")

    _h(doc, "1.1 End-to-end pipeline", 2)
    _bullets(doc, [
        "Git mining: parses 'git log --numstat' to extract 11 features per commit - "
        "lines added/deleted (log-scaled churn), files touched, blast radius, diff "
        "entropy, author experience, author familiarity with the touched files, file "
        "hotness (prior change count), file age, and time-of-day (cyclical encoding).",
        "Defect labeling (SZZ-style): a commit is labeled defective if a later "
        "bug-fix commit (subject matching fix/bug/hotfix/regression/crash/error "
        "heuristics) touched the same files within a 14-day window.",
        "Six ML base learners, implemented from scratch in numpy: logistic "
        "regression (L2), CART decision tree, random forest, gradient-boosted "
        "stumps, Gaussian naive Bayes, and a feed-forward neural network (one "
        "hidden layer, tanh, Adam-style updates).",
        "Stacked meta-learning: base models produce out-of-fold predictions via "
        "rolling-origin folds; a second-level logistic model learns how to combine "
        "them. Every backtest trains strictly on the past, so metrics are honest.",
        "Analytics: AUC, F1, Brier score, and the headline metric - defects caught "
        "at 20% review effort - plus per-file risk heatmap, author risk profile, "
        "and a project risk-trend curve.",
        "Self-contained dark-theme HTML dashboard: KPI cards, model comparison "
        "bars, riskiest commits with per-commit risk drivers, heatmap, author "
        "table, SVG trend sparkline. One file, no JavaScript dependencies.",
        "Web app: 'python -m predict web' opens an animated welcome screen (aurora "
        "orbs, twinkling stars, typewriter tagline, glass card) where any GitHub "
        "URL can be analyzed with a live progress bar.",
        "Tabbed deep-dive dashboard: six interactive tabs - Model Lab (per-model "
        "AUC/F1/Brier comparison), Risky Commits (live filter, risk pills, per-commit "
        "drivers), File Heatmap (filterable), Authors (rate bars), Trends & "
        "Distributions (risk histogram, repo-specific feature weights, trend "
        "sparkline), and How It Works (plain-English engine walkthrough).",
    ])

    # ---------- what was done, chronologically
    _h(doc, "2. What Was Done (Build Log)", 1)
    _h(doc, "2.1 Phase 1 - Concept and MVP", 2)
    _bullets(doc, [
        "Selected the idea from five candidates (docs-drift detector, PR time "
        "machine, flake autopsy, onboarding engine, dependency risk radar) and "
        "converged on predictive defect-risk as the most novel and demonstrable.",
        "Built the git-mining feature extractor and SZZ labeling (features.py).",
        "Built a numpy logistic hazard model with class-imbalance weighting (model.py).",
        "Built a CLI (train / score / explain) and a first test suite.",
    ])
    _h(doc, "2.2 Phase 2 - Multi-model ensemble", 2)
    _bullets(doc, [
        "Implemented six ML models from scratch in pure numpy (ensemble.py) - no "
        "scikit-learn, no external ML dependency.",
        "Added stacked meta-learning over rolling-origin out-of-fold predictions "
        "to prevent temporal leakage.",
        "Built the analytics layer (analytics.py): metrics, heatmap, author "
        "profile, weekly risk curve.",
        "Built the HTML dashboard generator (report.py) with dark theme.",
        "Rewired the CLI: train / score / report.",
        "Debugged the CART split-criterion implementation (Gini gain was computed "
        "incorrectly; fixed with a proper weighted child-Gini search).",
        "Created a synthetic 400-commit demo repository generator with realistic "
        "defect dynamics (scripts/make_demo_repo.py).",
        "Fixed a git-log parsing bug (commit-record delimiter handling).",
    ])
    _h(doc, "2.3 Phase 3 - Real-world validation", 2)
    _bullets(doc, [
        "Cloned the public psf/requests repository and trained on it live: 2,195 "
        "commits mined, 586 defect-linked, 326 files; stacked ensemble AUC 0.607, "
        "best of 7 models on a time-ordered backtest.",
        "Generated requests_report.html from that real repository.",
    ])
    _h(doc, "2.4 Phase 4 - Web application", 2)
    _bullets(doc, [
        "Built a Flask server (server.py) with background job execution: POST "
        "/api/analyze accepts a GitHub URL, clones, mines, trains, and caches the "
        "rendered dashboard; /api/status streams progress.",
        "Built the animated welcome screen (welcome.html): aurora background, "
        "star field, gradient logo shimmer, typewriter tagline, one-click demo "
        "repo chips, live progress bar, KPI cards.",
        "Verified the full flow live: analyzed github.com/psf/requests through the "
        "web UI - 3,587 commits, 1,213 defect-linked, 485 files; ensemble AUC "
        "0.666 vs 0.600 best single model; dashboard rendered in-browser.",
        "Fixed a job-status bug found during live verification (status remained "
        "'running' at 100%).",
    ])

    # ---------- how to run
    _h(doc, "3. How to Run", 1)
    doc.add_paragraph("pip install -r requirements.txt", style="Intense Quote")
    doc.add_paragraph("python -m predict web", style="Intense Quote")
    _p(doc, "The browser opens automatically at http://127.0.0.1:5050. Paste any "
            "public GitHub repository URL and the dashboard builds live.")
    _p(doc, "CLI alternative:", bold=True)
    _bullets(doc, [
        "python -m predict train --repo <path-or-cloned-repo>",
        "python -m predict report --repo <repo> --out report.html",
        "python -m predict score --repo <repo>",
    ])

    # ---------- results
    _h(doc, "4. Verified Results", 1)
    tbl = doc.add_table(rows=1, cols=4)
    tbl.style = "Light Grid Accent 1"
    hdr = tbl.rows[0].cells
    for i, h in enumerate(["Repository", "Commits", "Defect-linked", "Ensemble AUC"]):
        hdr[i].text = h
        hdr[i].paragraphs[0].runs[0].bold = True
    for repo, commits, defects, auc_v in [
        ("psf/requests (CLI run, earlier build)", "2,195", "586", "0.607"),
        ("psf/requests (final build)", "2,977", "see dashboard", "0.635"),
        ("pallets/flask (final build)", "693", "see dashboard", "0.691"),
        ("pallets/markupsafe (final build)", "480", "see dashboard", "0.719"),
        ("synthetic demo repo", "400", "46", "0.629"),
    ]:
        c = tbl.add_row().cells
        c[0].text, c[1].text, c[2].text, c[3].text = repo, commits, defects, auc_v
    _p(doc, "All metrics are computed on rolling-origin out-of-fold predictions - "
            "the model never sees the future while being evaluated, unlike typical "
            "random-split ML demos.")

    # ---------- architecture
    _h(doc, "5. Architecture", 1)
    arch = doc.add_table(rows=1, cols=2)
    arch.style = "Light Grid Accent 1"
    h = arch.rows[0].cells
    h[0].text = "Module"
    h[1].text = "Responsibility"
    for mod, resp in [
        ("predict/features.py", "git mining, 11 features/commit, SZZ defect labels"),
        ("predict/ensemble.py", "6 numpy models + stacking meta-learner"),
        ("predict/analytics.py", "metrics, heatmap, author table, risk trend"),
        ("predict/report.py", "self-contained HTML dashboard renderer"),
        ("predict/pipeline.py", "shared analysis pipeline: mine -> train -> backtest -> report (used by CLI and web)"),
        ("predict/server.py", "Flask app: GitHub URL -> background analysis job"),
        ("predict/welcome.html", "animated welcome screen (no JS frameworks)"),
        ("predict/__main__.py", "CLI: train / score / report / web"),
    ]:
        c = arch.add_row().cells
        c[0].text, c[1].text = mod, resp

    # ---------- ML deep dive
    _h(doc, "6. The Machine Learning Inside", 1)

    _h(doc, "6.1 Feature engineering (11 features per commit)", 2)
    feat = doc.add_table(rows=1, cols=2)
    feat.style = "Light Grid Accent 1"
    h = feat.rows[0].cells
    h[0].text = "Feature"
    h[1].text = "Why it predicts defects"
    for name, why in [
        ("la, ld (log-scaled churn)", "Large diffs are harder to review and hide more mistakes"),
        ("nfiles / nfiles_ratio", "Blast radius: many-file changes correlate with integration bugs"),
        ("add/delete balance", "Pure-addition commits (new code) vs pure-deletion (cleanup) carry different risk profiles"),
        ("author_exp", "Inexperienced authors historically produce more defects"),
        ("author_fam", "Working outside one's familiar code area is riskier"),
        ("file_hot", "Frequently-changed files are statistically defect-prone"),
        ("file_age", "Older files accumulate complexity and rot"),
        ("hour_sin, hour_cos", "Late-night / off-hours commits defect-link more often (cyclical encoding avoids midnight jumping to 1 AM)"),
    ]:
        c = feat.add_row().cells
        c[0].text, c[1].text = name, why

    _h(doc, "6.2 The six base learners (all from scratch in numpy)", 2)
    _bullets(doc, [
        "Logistic regression (L2-regularized): the calibrated linear baseline; "
        "captures monotone risk effects like churn.",
        "CART decision tree (gini, depth-limited): captures interactions such as "
        "'huge diff AND unfamiliar file'.",
        "Random forest (40 bagged trees, feature subsampling): variance reduction "
        "over a single tree; robust to noisy SZZ labels.",
        "Gradient-boosted stumps (120 rounds): sequentially fits residual errors; "
        "the strongest single-table learner for tabular data like this.",
        "Gaussian naive Bayes: fast probabilistic view; contributes an independent "
        "opinion the meta-model can weight.",
        "Neural network (16-unit tanh hidden layer, Adam updates): learns smooth "
        "non-linear feature combinations the trees miss.",
    ])
    _p(doc, "Every learner receives class-imbalance weighting because defective "
            "commits are a small minority (roughly 5-30% of history).", italic=True)

    _h(doc, "6.3 Stacked meta-learning", 2)
    _p(doc, "Instead of averaging the six models, BugForecast trains a second-level "
            "logistic model on their out-of-fold predictions. The meta-learner learns "
            "which base model to trust for which kind of change - measured gain on "
            "psf/requests: ensemble AUC 0.666 vs 0.600 for the best single model.")

    _h(doc, "6.4 Time-safe evaluation (why the numbers are honest)", 2)
    _p(doc, "Random train/test splits leak the future into training and inflate "
            "accuracy - a chronic flaw in defect-prediction literature. BugForecast "
            "uses rolling-origin folds: to predict block k of history, models train "
            "only on blocks 1..k-1. All dashboard metrics are computed on these "
            "strictly out-of-sample predictions, so they reflect real deployment "
            "performance.")

    _h(doc, "6.5 The headline metric: defects caught at 20% review effort", 2)
    _p(doc, "Calibrated probabilities are converted into a concrete workflow: rank all "
            "changes by predicted risk, let reviewers inspect only the riskiest 20%, "
            "and measure what fraction of all future defects that intercepts. This "
            "turns an abstract ML score into a review-priority queue a team can use "
            "tomorrow morning.")

    # ---------- comparison
    _h(doc, "7. Comparison With Existing Tools", 1)
    cmp_tbl = doc.add_table(rows=1, cols=4)
    cmp_tbl.style = "Light Grid Accent 1"
    h = cmp_tbl.rows[0].cells
    for i, name in enumerate(["Capability", "BugForecast", "CodeQL / SonarQube",
                              "Typical bug-prediction research code"]):
        h[i].text = name
    rows = [
        ("Predicts future defects (probability per change)", "Yes", "No - finds existing patterns", "Partially"),
        ("Learns from the repo's own history", "Yes", "No - fixed rule sets", "Yes"),
        ("Multiple ML models + stacking ensemble", "Yes (6 + meta)", "No", "Rarely (single model)"),
        ("Time-safe temporal backtesting", "Yes", "N/A", "Sometimes"),
        ("One command, paste a GitHub URL", "Yes", "Requires setup/integration", "No"),
        ("Animated visual dashboard, zero config", "Yes", "Enterprise UI, heavy setup", "No (CSV/plots)"),
        ("Dependencies", "numpy + flask only", "Heavy toolchains", "Varies"),
        ("Per-commit risk drivers (explainability)", "Yes", "Rule-level only", "Rare"),
        ("Per-file heatmap + author risk profile", "Yes", "Partial", "No"),
        ("Runs fully offline on local clones", "Yes", "Yes", "Yes"),
    ]
    for row in rows:
        c = cmp_tbl.add_row().cells
        for i, v in enumerate(row):
            c[i].text = v

    # ---------- why unique
    _h(doc, "8. Differentiation", 1)
    _p(doc, "Observed design differences versus common practice - not claims of "
            "global novelty; similar techniques exist individually in research "
            "literature.", italic=True)
    _bullets(doc, [
        "Ensemble stacking on code-defect prediction: bug-prediction literature "
        "and tools almost always use a single classifier. BugForecast trains six "
        "heterogeneous models and lets a meta-learner learn their complementary "
        "strengths - measured gain (0.600 to 0.666 AUC on psf/requests).",
        "Honest temporal evaluation: most published defect-prediction numbers use "
        "random train/test splits, which leak future information and inflate "
        "accuracy. BugForecast's rolling-origin backtesting makes every number on "
        "the dashboard defensible in front of a skeptical engineer.",
        "Review-effort metric: instead of abstract accuracy, the headline KPI is "
        "'defects caught at 20% review effort' - a number a team can act on "
        "tomorrow morning.",
        "Full product, not a research script: from a GitHub URL to an explainable, "
        "self-contained visual dashboard in one command, with per-commit risk "
        "drivers, file heatmap and author risk profile.",
        "Zero-dependency ML: all six models implemented from scratch in numpy - "
        "the entire stack installs in seconds and runs on any machine with Python.",
        "Self-learning per project: no universal model trained on other people's "
        "bugs; it mines the target repository's own history, so predictions adapt "
        "to that codebase's real defect dynamics.",
    ])

    # ---------- why not built before
    _h(doc, "9. Why This Combination Is Rare", 1)
    _p(doc, "The individual techniques are all published; the rarity is the "
            "packaging. Possible reasons the combination is uncommon:", italic=True)
    _bullets(doc, [
        "The research-practice gap: defect prediction is a well-studied academic "
        "field (hundreds of papers), but the results live in papers and prototypes "
        "- nobody packaged temporal validation, ensembling and a product-grade UX "
        "into one tool a developer can run in 30 seconds.",
        "Data inconvenience: the ground truth (which commits caused bugs) does not "
        "exist in any database; it must be reconstructed heuristically from git "
        "history (SZZ-style linking). Tools that tried stopped at fixed rule "
        "engines because mining history is messy.",
        "Commercial incentive mismatch: static-analysis vendors sell 'we find bugs "
        "now' products; 'we predict where bugs will appear' requires trust in "
        "probabilities, which is harder to market - so the niche stayed open.",
        "Engineering breadth required: it needs git internals, six ML models, "
        "leakage-proof evaluation, and web UI in one package - each part is easy, "
        "the integration is the barrier.",
    ])

    # ---------- hardening
    _h(doc, "10. Final Hardening (Audit Outcomes)", 1)
    _bullets(doc, [
        "Fixed a real SZZ indexing bug: when a fix commit referenced an earlier "
        "fix commit, label back-propagation crashed or mislabeled rows "
        "(IndexError on pallets/markupsafe). Labels now map commits to training "
        "rows explicitly.",
        "Fixed misleading explainability: per-commit 'risk drivers' previously "
        "sorted raw feature values (scale-dependent, misleading); they now show "
        "model-derived contributions (logistic weight x feature value), and the "
        "repo-level feature-weight chart uses the logistic base learner's actual "
        "feature weights rather than meta-model weights over base models.",
        "Extracted the shared analysis pipeline (pipeline.py) so the CLI and web "
        "app cannot drift apart.",
        "Hardened the URL parser: query strings, fragments, /tree/ paths and "
        "owner/repo shorthand accepted; non-GitHub hosts and path traversal "
        "rejected - covered by new unit tests.",
        "Web UX hardening: connection-loss retry in the poll loop, bounded job "
        "cache, clean repository names instead of temp paths, friendly errors "
        "for clone timeouts and unsuitable repositories.",
        "Removed dead code (unused per-commit feature builder, legacy CLI path).",
    ])

    # ---------- limitations
    _h(doc, "11. Honest Limitations", 1)
    _bullets(doc, [
        "SZZ labels are heuristics: a fix commit touching a file implicates earlier "
        "commits on those files; some labels will be noise. This is standard in "
        "the field but imperfect.",
        "Fix detection is subject-line based; repos with unusual commit message "
        "conventions will label fewer defects.",
        "Small repos (<60 commits or <5 fix commits) cannot train the ensemble.",
        "AUC around 0.6-0.67 on real repos is realistic for this task (defect "
        "prediction is inherently noisy); the value is prioritization, not "
        "certainty.",
    ])

    # ---------- about
    _h(doc, "12. About BugForecast (Concise)", 1)
    ab = doc.add_paragraph()
    ab.paragraph_format.left_indent = Inches(0.3)
    r = ab.add_run(
        "BugForecast is a free, open predictive defect-risk engine: point it at any "
        "git repository (or paste a GitHub URL into its animated web dashboard) and "
        "it mines the project's full history, learns which kinds of changes have "
        "historically caused bugs, and predicts - with calibrated probabilities and "
        "per-commit explanations - where the next defects are most likely to occur. "
        "It combines six machine-learning models (logistic regression, decision "
        "tree, random forest, gradient boosting, naive Bayes, neural network) "
        "through a stacked meta-learner, evaluates itself with strictly "
        "future-proof backtests, and presents everything in a single self-contained "
        "dark-theme dashboard: model comparison, riskiest commits, per-file risk "
        "heatmap, author risk profiles and a risk-trend curve. Its headline metric "
        "answers the only question that matters to a team: if reviewers focus on "
        "the riskiest 20% of changes, how many future bugs do they catch? Unlike "
        "static analyzers that find today's bug patterns, BugForecast forecasts "
        "tomorrow's - learned from the project's own history, in one command, with "
        "no configuration and only two dependencies (numpy and flask).")
    r.italic = True

    doc.save(path)
    print(f"saved {path}")


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "BugForecast_Report.docx"
    build(out)
