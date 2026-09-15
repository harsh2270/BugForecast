"""BugForecast Streamlit frontend.

Reuses the existing predict pipeline (mine -> SZZ label -> 6-model ensemble
-> backtest -> report dict) without touching the core ML/analytics code.

Run:  streamlit run app.py
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile

import streamlit as st

from predict.pipeline import analyze_repo

st.set_page_config(page_title="BugForecast", page_icon=":bar_chart:",
                   layout="wide")

_GITHUB_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9-]{1,39})"
    r"/([A-Za-z0-9._-]{1,100})(?:[/?#].*)?$", re.I)


def parse_github_url(url: str) -> str | None:
    """Same contract as predict.server._parse_github_url."""
    url = url.strip()
    url = re.split(r"[?#]", url, maxsplit=1)[0].rstrip("/")
    url = re.sub(r"/(tree|blob|commits|releases|issues|pulls?)/.*$", "", url, flags=re.I)
    m = _GITHUB_RE.match(url)
    if m:
        name = m.group(2)
        if name.endswith(".git"):
            name = name[:-4]
        return f"https://github.com/{m.group(1)}/{name}.git" if name else None
    m = re.match(r"^([A-Za-z0-9-]{1,39})/([A-Za-z0-9._-]{1,100})$", url)
    if m:
        return f"https://github.com/{m.group(1)}/{m.group(2)}.git"
    return None


@st.cache_data(show_spinner=False)
def analyze_cached(repo_path: str, repo_label: str, _mtime: float) -> dict | None:
    try:
        return analyze_repo(repo_path)
    except ValueError as exc:
        st.error(str(exc))
        return None
    except Exception as exc:  # clone/analysis failure surfaced to the user
        st.error(f"Analysis failed: {exc}")
        return None


def clone_to_temp(repo_url: str) -> tuple[str | None, str | None]:
    workdir = tempfile.mkdtemp(prefix="bugforecast_st_")
    target = os.path.join(workdir, "repo")
    try:
        r = subprocess.run(["git", "clone", "--quiet", repo_url, target],
                           capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        shutil.rmtree(workdir, ignore_errors=True)
        return None, "Cloning timed out after 10 minutes. Try a smaller repository."
    if r.returncode != 0:
        shutil.rmtree(workdir, ignore_errors=True)
        err = r.stderr.strip() or "repository not found or inaccessible"
        return None, f"git clone failed: {err[:300]}"
    return target, None


# ------------------------------------------------------------------ UI
st.markdown(
    """
    <style>
      .stApp { background: #05070f; color: #e6edf3; }
      #MainMenu, header, footer, [data-testid="stSidebar"],
      [data-testid="stStatusWidget"], [data-testid="stToolbar"] { display: none; }
      .block-container { padding-top: 1.2rem; max-width: 1200px; }
      h1, h2, h3 { color: #e6edf3; }
      [data-testid="stMetricValue"], [data-testid="stMetricLabel"] p { color: #c9d1d9; }
      .stTabs [data-baseweb="tab"] { color: #8b949e; }
      .stTabs [aria-selected="true"] { color: #58a6ff !important;
        border-color: #58a6ff !important; }

      /* animated aurora background */
      .stApp::before, .stApp::after {
        content: ""; position: fixed; border-radius: 50%;
        filter: blur(90px); opacity: .30; z-index: 0; pointer-events: none;
      }
      .stApp::before {
        width: 55vw; height: 55vw; top: -20vw; left: -15vw;
        background: radial-gradient(circle, #1f6feb 0%, transparent 70%);
        animation: drift1 26s ease-in-out infinite alternate;
      }
      .stApp::after {
        width: 50vw; height: 50vw; bottom: -22vw; right: -12vw;
        background: radial-gradient(circle, #a371f7 0%, transparent 70%);
        animation: drift2 32s ease-in-out infinite alternate;
      }
      @keyframes drift1 { to { transform: translate(9vw, 7vh) scale(1.15); } }
      @keyframes drift2 { to { transform: translate(-8vw, -6vh) scale(1.2); } }

      .bf-hero { text-align: center; padding: 3.2rem 0 1.4rem; position: relative; z-index: 1; }
      .bf-logo {
        font-size: 3.4rem; font-weight: 800; letter-spacing: -1px;
        background: linear-gradient(92deg, #58a6ff, #a371f7, #58a6ff);
        background-size: 220% auto; -webkit-background-clip: text;
        background-clip: text; color: transparent;
        animation: shimmer 7s linear infinite;
      }
      @keyframes shimmer { to { background-position: 220% center; } }
      .bf-tag { color: #8b949e; font-size: 1.05rem; margin-top: .4rem; }
      .bf-card {
        background: rgba(22, 27, 34, .78); border: 1px solid #21262d;
        border-radius: 14px; padding: 1.5rem 1.7rem; max-width: 640px;
        margin: 1.6rem auto 0; box-shadow: 0 0 40px rgba(31,111,235,.12);
        position: relative; z-index: 1;
      }
      .stApp > div { position: relative; z-index: 1; }
      .stButton > button {
        border-radius: 10px; font-weight: 600; border: 1px solid #21262d;
      }
      [data-testid="stForm"] { border: none; }
      [data-baseweb="input"] > div { background: #0d1117 !important;
        border-color: #30363d !important; color: #e6edf3 !important; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="bf-hero">
  <div class="bf-logo">BugForecast</div>
  <div class="bf-tag">Learns from a repository's own git history to flag which
  commits and files are most likely to be defect-prone. Predictions are
  probabilistic estimates, not certainties.</div>
</div>
""", unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="bf-card">', unsafe_allow_html=True)
    mode = st.selectbox("Source", ["GitHub URL", "Local repository path"],
                        label_visibility="collapsed")
    url = path = None
    if mode == "GitHub URL":
        url = st.text_input("GitHub URL", label_visibility="collapsed",
                            placeholder="https://github.com/owner/repo")
    else:
        path = st.text_input("Local path to a git repository",
                             label_visibility="collapsed",
                             placeholder=r"C:\path\to\repo")
    c_run, c_demo = st.columns([3, 2])
    run_btn = c_run.button("Analyze", type="primary", use_container_width=True)
    demo = c_demo.button("Try demo: psf/requests", use_container_width=True)
    if demo and not url:
        url = "https://github.com/psf/requests.git"
    st.caption("Needs >= 60 usable commits and >= 5 bug-fix style commits to "
               "train. Runs locally; no data leaves your machine.")
    st.markdown('</div>', unsafe_allow_html=True)

if not run_btn:
    st.stop()

repo_path: str | None = None
workdir: str | None = None
repo_label: str
if url:
    clean = parse_github_url(url)
    if not clean:
        st.error("Enter a valid GitHub repository URL, e.g. "
                 "https://github.com/psf/requests")
        st.stop()
    repo_label = clean.rsplit("/", 1)[-1].removesuffix(".git")
    with st.status(f"Cloning {repo_label}...", expanded=True) as s:
        repo_path, clone_err = clone_to_temp(clean)
        if clone_err:
            s.update(label="Clone failed", state="error")
            st.error(clone_err)
            st.stop()
        s.update(label="Cloned", state="complete")
elif path:
    repo_path = os.path.abspath(path.strip())
    if not os.path.isdir(os.path.join(repo_path, ".git")):
        st.error("That path does not look like a git repository "
                 "(no .git directory found).")
        st.stop()
    repo_label = os.path.basename(repo_path)
else:
    st.warning("Enter a GitHub URL or a local repository path first.")
    st.stop()

mtime = 0.0
try:
    mtime = os.path.getmtime(os.path.join(repo_path, ".git"))  # type: ignore[arg-type]
except OSError:
    pass

with st.spinner("Mining history, training the 6-model ensemble, backtesting..."):
    result = analyze_cached(repo_path, repo_label, mtime)

if workdir:
    shutil.rmtree(workdir, ignore_errors=True)
if result is None:
    st.stop()

rep = result["report"]

# KPIs
st.subheader(f"Analysis: {repo_label}")
cols = st.columns(len(rep["kpis"]))
for c, (val, label) in zip(cols, rep["kpis"]):
    c.metric(label, val)
st.caption("All metrics come from an out-of-sample, time-ordered backtest — "
           "not from the training fit.")

tabs = st.tabs(["Model Lab", "Risky Commits", "File Heatmap", "Authors",
                "Trends & Distributions", "How It Works"])

with tabs[0]:
    st.markdown("Per-model out-of-sample performance. AUC 0.5 = coin flip; "
                "Brier = mean squared error of probabilities (lower is better).")
    best = max(m["auc"] for m in rep["models"])
    for m in rep["models"]:
        c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
        c1.markdown(f"**{m['name']}**")
        c2.metric("AUC", f"{m['auc']:.3f}")
        c3.metric("F1", f"{m['f1']:.3f}")
        c4.metric("Brier", f"{m['brier']:.3f}")
        c1.progress(m["auc"] / max(best, 0.01) if m["auc"] > 0 else 0.0)
    gain = rep["ensemble_gain"]
    st.success(f"Stacking gain: stacked AUC {gain['stacked']:.3f} vs best "
               f"single model {gain['best_single']:.3f}. Random-commit defect "
               f"baseline at 20% review effort: {gain['baseline20']*100:.0f}% "
               f"vs model {gain['recall20']*100:.0f}%.")

with tabs[1]:
    st.markdown("Highest predicted-risk commits. Drivers are the logistic "
                "learner's per-feature contributions (weight x value) — "
                "model-derived, not observed facts.")
    for r in rep["risky"]:
        with st.expander(f"`{r['hash'][:8]}` — {r['subject']}  ·  "
                         f"risk {r['p']:.0%}"):
            st.progress(min(r["p"], 1.0))
            for name, val in r["drivers"]:
                st.markdown(f"- `{name}`: contribution {val:+.2f}")

with tabs[2]:
    q = st.text_input("Filter files", "")
    rows = [(f["file"], f["risk"]) for f in rep["files"]
            if q.lower() in f["file"].lower()]
    if not rows:
        st.info("No files match that filter.")
    for fname, risk in rows[:40]:
        c1, c2 = st.columns([4, 1])
        c1.markdown(f"`{fname}`")
        c2.metric("risk", f"{risk:.0%}")
        st.progress(min(max(risk, 0.0), 1.0))

with tabs[3]:
    st.markdown("Commit counts and defect links per author (descriptive "
                "statistics, not blame).")
    st.table({"author": [a["author"] for a in rep["authors"][:20]],
              "commits": [a["commits"] for a in rep["authors"][:20]],
              "defect-linked": [a["defects"] for a in rep["authors"][:20]],
              "defect rate": [f"{a['rate']:.0%}" for a in rep["authors"][:20]]})

with tabs[4]:
    left, right = st.columns(2)
    with left:
        st.markdown("**Risk distribution across all commits**")
        st.bar_chart({"commits": rep["risk_dist"]})
    with right:
        st.markdown("**What drives risk in this repo** (logistic weights)")
        st.bar_chart({n: abs(v) for n, v in rep["top_features"]})
    st.line_chart([w["risk"] for w in rep["trend"]],
                  use_container_width=True)

with tabs[5]:
    st.markdown(
        """
1. **Mine** — every commit becomes a feature vector: churn, files touched,
   author familiarity with the files, file hotness, time of day, and more.
2. **Label** — SZZ-style: a commit is labeled defect-linked if a later
   bug-fix commit touched the same files within 14 days. This is a
   heuristic label, not ground truth.
3. **Train** — six models (logistic, decision tree, random forest, boosted
   stumps, naive Bayes, MLP) are combined by a stacked meta-learner.
4. **Backtest** — evaluation uses rolling-origin, time-ordered
   out-of-fold predictions, so the model never scores commits from its
   own future.
5. **Score** — the trained ensemble ranks commits and files by predicted
   defect probability. Treat scores as review-priority hints, not proof.
        """)
