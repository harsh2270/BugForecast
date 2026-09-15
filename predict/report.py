"""Self-contained HTML dashboard generator for BugForecast reports."""
from __future__ import annotations

import html
import time


CSS = """
:root{--bg:#0b0f19;--card:#131a2b;--line:#1f2a44;--fg:#e6ebf5;--dim:#8b96ad;
--red:#f87171;--amber:#fbbf24;--green:#34d399;--blue:#60a5fa;--violet:#a78bfa}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--fg);font:15px/1.55 'Segoe UI',system-ui,sans-serif;padding:32px}
h1{font-size:26px;letter-spacing:.5px}h2{font-size:17px;color:var(--dim);text-transform:uppercase;letter-spacing:2px;margin:34px 0 12px}
.sub{color:var(--dim);margin-top:4px}
.grid{display:grid;gap:14px}
.kpis{grid-template-columns:repeat(auto-fit,minmax(170px,1fr))}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px}
.kpi .v{font-size:30px;font-weight:700}
.kpi .l{color:var(--dim);font-size:12px;text-transform:uppercase;letter-spacing:1.5px;margin-top:4px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px}
table{width:100%;border-collapse:collapse;font-size:14px}
th{color:var(--dim);text-align:left;font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:1px;padding:8px 10px;border-bottom:1px solid var(--line)}
td{padding:8px 10px;border-bottom:1px solid var(--line)}
.bar{height:8px;border-radius:4px;background:var(--line);overflow:hidden}
.bar>i{display:block;height:100%}
.r-high{color:var(--red)}.r-med{color:var(--amber)}.r-low{color:var(--green)}
.tag{display:inline-block;padding:2px 9px;border-radius:20px;font-size:12px;background:var(--line)}
.bar>.r{background:var(--red)}.bar>.m{background:var(--amber)}.bar>.g{background:var(--green)}
.modelbar{display:flex;align-items:center;gap:10px;margin:7px 0}
.modelbar .n{width:150px;color:var(--dim);font-size:13px}
.modelbar .bar{flex:1}
.modelbar .v{width:110px;text-align:right;font-variant-numeric:tabular-nums}
.foot{color:var(--dim);font-size:12px;margin-top:36px;border-top:1px solid var(--line);padding-top:14px}
/* ---- tabs ---- */
.tabs{display:flex;gap:6px;margin:26px 0 0;flex-wrap:wrap}
.tabs button{padding:11px 20px;font-size:13px;font-weight:600;color:var(--dim);background:var(--card);
border:1px solid var(--line);border-radius:12px 12px 0 0;cursor:pointer;transition:.2s;border-bottom:none}
.tabs button:hover{color:var(--fg);transform:translateY(-2px)}
.tabs button.on{color:#06121f;background:linear-gradient(90deg,var(--blue),var(--violet))}
.tabpane{display:none;border:1px solid var(--line);border-radius:0 14px 14px 14px;background:var(--card);padding:24px}
.tabpane.on{display:block;animation:fadein .45s ease}
@keyframes fadein{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}
.filter{width:100%;max-width:380px;padding:10px 14px;color:var(--fg);background:#0d1424;
border:1px solid var(--line);border-radius:10px;outline:none;margin-bottom:14px;font-size:14px}
.filter:focus{border-color:var(--blue)}
.pill{display:inline-block;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:700;letter-spacing:.5px}
.pill.hi{background:#f8717122;color:var(--red)}.pill.md{background:#fbbf2422;color:var(--amber)}.pill.lo{background:#34d39922;color:var(--green)}
.kv{display:flex;gap:10px;flex-wrap:wrap;margin:6px 0 16px}
.kv div{background:#0d1424;border:1px solid var(--line);border-radius:10px;padding:10px 14px;font-size:13px}
.kv b{color:var(--blue);font-variant-numeric:tabular-nums}
.hist{display:flex;align-items:flex-end;gap:3px;height:120px;margin:10px 0}
.hist i{flex:1;background:linear-gradient(180deg,var(--violet),var(--blue));border-radius:3px 3px 0 0;min-height:2px}
.legend{color:var(--dim);font-size:12px;margin-top:4px}
"""


def _risk_color(p):
    return "r-high" if p >= 0.6 else ("r-med" if p >= 0.3 else "r-low")


def _bar_color(p):
    return "r" if p >= 0.6 else ("m" if p >= 0.3 else "g")


def generate(report: dict) -> str:
    e = html.escape
    kpis = report["kpis"]
    parts = [f"""<!doctype html><html><head><meta charset="utf-8">
<title>BugForecast — {e(report['repo'])}</title><style>{CSS}</style></head><body>
<h1>BugForecast — Predictive Defect Risk</h1>
<div class="sub">repo: {e(report['repo'])} · generated {time.strftime('%Y-%m-%d %H:%M')}</div>"""]

    # KPI row
    parts.append("<div class='grid kpis'>")
    for v, l in kpis:
        parts.append(f"<div class='kpi'><div class='v'>{e(str(v))}</div><div class='l'>{e(l)}</div></div>")
    parts.append("</div>")

    # ---------------- tab bar + panes ----------------
    parts.append("<div class='tabs'>")
    for i, label in enumerate(["Model Lab", "Risky Commits", "File Heatmap",
                               "Authors", "Trends & Distributions", "How It Works"]):
        parts.append(f"<button onclick=\"tab({i})\" class='{'' if i else 'on'}'>{label}</button>")
    parts.append("</div>")

    # ---- TAB 1: model lab ----
    parts.append("<div class='tabpane on'>")
    parts.append("<h2>Model Comparison — Out-of-Fold Backtest</h2>")
    for m in report["models"]:
        parts.append(
            f"<div class='modelbar'><div class='n'>{e(m['name'])}</div>"
            f"<div class='bar'><i class='{_bar_color(m['auc'])}' style='width:{m['auc']*100:.0f}%'></i></div>"
            f"<div class='v'>AUC {m['auc']:.3f} · F1 {m['f1']:.3f} · Brier {m['brier']:.3f}</div></div>")
    ens = report["ensemble_gain"]
    parts.append(f"""<div style='margin-top:20px;color:var(--dim);font-size:14px'>
The stacked ensemble reaches <b style='color:var(--fg)'>AUC {ens['stacked']:.3f}</b> vs the best single model at
<b style='color:var(--fg)'>AUC {ens['best_single']:.3f}</b> on strictly out-of-sample, time-ordered backtests.
At a 20% review budget it catches <b style='color:var(--fg)'>{ens['recall20']*100:.0f}%</b> of all defects
(random baseline: {ens['baseline20']*100:.0f}%). Six heterogeneous learners are combined by a
second-level meta-model that learns which model to trust for which kind of change.
</div><div class='legend'>AUC = ranking quality (0.5 = coin flip, 1.0 = perfect) · F1 = precision/recall balance ·
Brier = calibration of probabilities (lower is better)</div>""")
    parts.append("</div>")

    # ---- TAB 2: risky commits ----
    parts.append("<div class='tabpane'>")
    parts.append("<h2>Highest-Risk Commits (never seen in training)</h2>")
    parts.append("<input class='filter' id='cf' placeholder='filter by message, hash or risk level...' oninput='cfilt()'>")
    parts.append("<table id='ct'><tr><th>Risk</th><th>Commit</th><th>Top risk drivers</th></tr>")
    for r in report["risky"][:25]:
        lvl = "hi" if r["p"] >= 0.6 else ("md" if r["p"] >= 0.3 else "lo")
        drivers = ", ".join(f"{e(n)} ({v:+.2f})" for n, v in r["drivers"])
        parts.append(
            f"<tr><td><span class='pill {lvl}'>{r['p']*100:.0f}%</span></td>"
            f"<td>{e(r['hash'][:8])} — {e(r['subject'][:70])}<div class='sub'>{time.strftime('%Y-%m-%d', time.gmtime(r['ts']))}</div></td>"
            f"<td style='color:var(--dim);font-size:13px'>{drivers}</td></tr>")
    parts.append("""</table><div class='legend'>Risk drivers are the commit's feature values that pushed the
probability up or down (e.g. la = lines added, author_fam = author familiarity with the touched files).</div></div>""")

    # ---- TAB 3: file heatmap ----
    parts.append("<div class='tabpane'>")
    parts.append("<h2>File Risk Heatmap</h2>")
    parts.append("<input class='filter' id='ff' placeholder='filter files...' oninput='ffilt()'>")
    for f in report["files"]:
        w = int(f["risk"] * 100)
        parts.append(
            f"<div class='modelbar' data-f=\"{e(f['file'].lower())}\"><div class='n' title=\"{e(f['file'])}\">{e(f['file'][-52:])}</div>"
            f"<div class='bar'><i class='{_bar_color(f['risk'])}' style='width:{max(w,2)}%'></i></div>"
            f"<div class='v'>{f['defects']} defects / {f['commits']} commits ({f['risk']*100:.0f}%)</div></div>")
    parts.append("<div class='legend'>Files are ranked by defect density weighted by activity — hot files with frequent fixes rise to the top.</div></div>")

    # ---- TAB 4: authors ----
    parts.append("<div class='tabpane'>")
    parts.append("<h2>Author Risk Profile</h2><table>"
                 "<tr><th>Author</th><th>Commits</th><th>Linked defects</th><th>Defect rate</th><th>Profile</th></tr>")
    for a in report["authors"]:
        bar_w = int(min(a["risk"], 1.0) * 100)
        parts.append(f"<tr><td>{e(a['author'])}</td><td>{a['commits']}</td><td>{a['defects']}</td>"
                     f"<td><span class='{_risk_color(a['risk'])}'>{a['risk']*100:.0f}%</span></td>"
                     f"<td style='width:30%'><div class='bar'><i class='{_bar_color(a['risk'])}' style='width:{bar_w}%'></i></div></td></tr>")
    parts.append("</table><div class='legend'>Defect rate is not blame — it reflects which modules and change types each author works on. Use it to target mentoring and review pairing.</div></div>")

    # ---- TAB 5: trends & distributions ----
    parts.append("<div class='tabpane'>")
    trend = report["trend"]
    if len(trend) >= 2:
        w, h = 760, 130
        mx = max(t["risk"] for t in trend) or 1
        pts = " ".join(
            f"{i*w/(len(trend)-1):.1f},{h-8-(t['risk']/mx)*(h-20):.1f}"
            for i, t in enumerate(trend))
        parts.append(f"""<h2>Risk Trend Over Project History</h2>
<svg width='100%' viewBox='0 0 {w} {h}' preserveAspectRatio='none' style='background:#0d1424;border-radius:10px'>
<polyline fill='none' stroke='#60a5fa' stroke-width='2.5' points='{pts}'/>
</svg><div class='sub'>average predicted defect probability per time bucket — rising line = project entering a defect-prone phase</div>""")
    parts.append("<h2>Risk Distribution (all analyzed commits)</h2>")
    parts.append("<div class='hist'>" + "".join(
        f"<i style='height:{max(int(v*120),2)}px'></i>" for v in report["risk_dist"]) + "</div>")
    parts.append("<div class='legend'>left = low-risk commits, right = high-risk commits. A healthy repo has a tall left "
                 "spike; a long right tail means many changes carry meaningful defect risk.</div>")
    parts.append("<h2>What Drives Risk In This Repo</h2><div class='kv'>")
    for n, v in report["top_features"]:
        parts.append(f"<div>{e(n)} <b>{v:+.2f}</b></div>")
    parts.append("</div><div class='legend'>Meta-model weights for this repository — positive values push risk up. "
                 "e.g. la (lines added), ld (lines deleted), nfiles (files touched), author_fam (author familiarity), file_hot (file hotness).</div>")
    parts.append("</div>")

    # ---- TAB 6: how it works ----
    parts.append("<div class='tabpane'>")
    parts.append("""<h2>Inside the Engine</h2>
<div style='color:var(--dim);font-size:14px;line-height:1.8'>
<b style='color:var(--fg)'>1 · Mining.</b> Every commit in the history is converted into an 11-dimensional
feature vector: churn (lines added/deleted), blast radius (files touched relative to repo size), diff entropy,
author experience and familiarity with the touched files, file hotness and age, and time-of-day.
<br><b style='color:var(--fg)'>2 · Labeling (SZZ-style).</b> A commit is labeled defective if a later bug-fix
commit touched the same files within 14 days — reconstructing ground truth from history alone.
<br><b style='color:var(--fg)'>3 · Ensemble.</b> Six learners — logistic regression, CART, random forest,
gradient-boosted stumps, naive Bayes and a neural net — are trained with class-imbalance weighting.
<br><b style='color:var(--fg)'>4 · Stacking.</b> Rolling-origin folds produce honest out-of-fold predictions;
a second-level meta-model learns to combine them. Training always uses only the past.
<br><b style='color:var(--fg)'>5 · Scoring & review targeting.</b> The stacked model outputs calibrated
probabilities; the dashboard surfaces where to spend limited review time for maximum defect interception.
</div>""")
    parts.append("</div>")

    parts.append("""<script>
function tab(i){document.querySelectorAll('.tabs button').forEach((b,j)=>b.classList.toggle('on',i===j));
document.querySelectorAll('.tabpane').forEach((p,j)=>p.classList.toggle('on',i===j));}
function cfilt(){const q=document.getElementById('cf').value.toLowerCase();
document.querySelectorAll('#ct tr').forEach((r,i)=>{if(i===0)return;
r.style.display=r.textContent.toLowerCase().includes(q)?'':'none';});}
function ffilt(){const q=document.getElementById('ff').value.toLowerCase();
document.querySelectorAll('[data-f]').forEach(r=>{r.style.display=r.dataset.f.includes(q)?'':'none';});}
</script>""")

    parts.append("<div class='foot'>BugForecast — predictive defect-risk engine. Labels via SZZ-style "
                 "fix-linkage; metrics computed on time-safe out-of-fold predictions. Free & open.</div></body></html>")
    return "".join(parts)
