"""BugForecast web app: one command, animated welcome screen, paste a GitHub URL,
get the full risk dashboard in the browser.

Run:  python -m predict web        (auto-opens the browser)

Security notes: URLs are strictly validated to github.com/<owner>/<repo>;
git is invoked with list arguments (no shell); clones happen in isolated
temp dirs that are always cleaned up; the server binds to 127.0.0.1 by
default and is intended for local use, not public deployment.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import threading
import uuid
import webbrowser

from flask import Flask, request, jsonify, Response

from .pipeline import analyze_repo

app = Flask(__name__)

# job_id -> {status, stage, progress, error?, result?}. A cap prevents
# unbounded memory growth in a long-lived local server.
JOBS: dict[str, dict] = {}
MAX_JOBS = 30
CLONE_TIMEOUT_S = 600

_GITHUB_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9-]{1,39})/([A-Za-z0-9._-]{1,100})$",
    re.I,
)


def _parse_github_url(url: str) -> str | None:
    """Accept github.com URLs (with optional query/fragment/tree paths) or
    'owner/repo' shorthand. Returns a clean clone URL or None."""
    url = url.strip()
    # drop query strings, fragments, and UI paths after the repo segment
    url = re.split(r"[?#]", url, maxsplit=1)[0].rstrip("/")
    url = re.sub(r"/(tree|blob|commits|releases|issues|pulls?)/.*$", "", url, flags=re.I)
    m = _GITHUB_RE.match(url)
    if m:
        owner, name = m.group(1), m.group(2)
        if name.endswith(".git"):
            name = name[:-4]
        if not name:
            return None
        return f"https://github.com/{owner}/{name}.git"
    m = re.match(r"^([A-Za-z0-9-]{1,39})/([A-Za-z0-9._-]{1,100})$", url)
    if m:
        return f"https://github.com/{m.group(1)}/{m.group(2)}.git"
    return None


def _run_analysis(job_id: str, repo_url: str) -> None:
    job = JOBS[job_id]
    workdir = tempfile.mkdtemp(prefix="bugforecast_")
    try:
        job.update(stage="Cloning repository from GitHub...", progress=5)
        target = os.path.join(workdir, "repo")
        try:
            r = subprocess.run(
                ["git", "clone", "--quiet", repo_url, target],
                capture_output=True, text=True, timeout=CLONE_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            job.update(status="error",
                       error=f"Cloning timed out after {CLONE_TIMEOUT_S // 60} minutes. "
                             f"Try a smaller repository.")
            return
        if r.returncode != 0:
            err = r.stderr.strip() or "repository not found or inaccessible"
            job.update(status="error", error=f"git clone failed: {err[:300]}")
            return

        def progress(stage: str, pct: int) -> None:
            job.update(stage=stage, progress=pct)

        result = analyze_repo(target, progress=progress)
        # show a clean repo name (from the clone URL), not a temp path
        clean = repo_url.rsplit("/", 1)[-1].removesuffix(".git")
        result["report"]["repo"] = clean
        job.update(stage="Done", progress=100, status="done", result=result)
    except ValueError as exc:
        job.update(status="error", error=str(exc))
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI
        job.update(status="error", error=f"Analysis failed: {exc}"[:400])
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        # evict oldest finished jobs beyond the cap
        if len(JOBS) > MAX_JOBS:
            done = [k for k, v in JOBS.items() if v.get("status") == "done"]
            for jid in done[:-MAX_JOBS]:
                JOBS.pop(jid, None)


# ------------------------------------------------------------------ routes
@app.get("/")
def index() -> Response:
    with open(os.path.join(os.path.dirname(__file__), "welcome.html"),
              encoding="utf-8") as f:
        return Response(f.read(), mimetype="text/html")


@app.post("/api/analyze")
def analyze():
    try:
        data = request.get_json(force=True)
    except Exception:  # malformed JSON body
        return jsonify({"error": "Send JSON: {\"url\": \"https://github.com/owner/repo\"}"}), 400
    repo_url = _parse_github_url((data or {}).get("url", ""))
    if not repo_url:
        return jsonify({"error": "Enter a valid GitHub repository URL, e.g. "
                                "https://github.com/psf/requests"}), 400
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {"status": "running", "stage": "Starting...", "progress": 0}
    threading.Thread(target=_run_analysis, args=(job_id, repo_url), daemon=True).start()
    return jsonify({"job": job_id})


@app.get("/api/status/<job_id>")
def status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"status": "error",
                        "error": "Unknown or expired job. Please analyze again."}), 404
    out = dict(job)
    if job.get("result"):
        out["summary"] = job["result"]["report"]["kpis"]
        out["repo_name"] = job["result"]["report"]["repo"]
    return jsonify(out)


@app.get("/report/<job_id>")
def report(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return Response("Unknown or expired job. Please analyze again.",
                        status=404, mimetype="text/plain")
    if job.get("status") == "running":
        return Response("Analysis still in progress - refresh in a moment.",
                        status=503, mimetype="text/plain")
    if not job.get("result"):
        return Response(job.get("error", "Report unavailable."),
                        status=500, mimetype="text/plain")
    return Response(job["result"]["html"], mimetype="text/html")


def run(host: str = "127.0.0.1", port: int = 5050) -> None:
    url = f"http://{host}:{port}"
    print(f"BugForecast dashboard -> {url}")
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    run()
