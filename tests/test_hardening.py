"""Tests for the GitHub URL parser and SZZ labeling invariants."""
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from predict.server import _parse_github_url
from predict.features import Commit
import predict.features as F


def test_parser_accepts_query_strings_and_fragments():
    assert _parse_github_url(
        "https://github.com/vllm-project/vllm?utm_source=chatgpt.com"
    ) == "https://github.com/vllm-project/vllm.git"
    assert _parse_github_url("https://github.com/psf/requests#readme") == \
        "https://github.com/psf/requests.git"


def test_parser_accepts_tree_paths_and_shorthand():
    assert _parse_github_url("https://github.com/psf/requests/tree/main") == \
        "https://github.com/psf/requests.git"
    assert _parse_github_url("owner/repo") == "https://github.com/owner/repo.git"


def test_parser_rejects_non_github_and_traversal():
    assert _parse_github_url("https://evil.com/owner/repo") is None
    assert _parse_github_url("https://github.com/../etc") is None
    assert _parse_github_url("https://github.com/owner/repo/extra/deep") is None
    assert _parse_github_url("") is None


def test_fix_commits_excluded_from_training_rows():
    """Fix commits must not appear as training rows, or the model can cheat
    by recognizing fix-style subjects."""
    commits = [
        Commit("a" * 40, "dev", 1000, "add feature"),
        Commit("b" * 40, "dev", 2000, "fix bug"),
    ]
    commits[0].files = ["src/x.py"]
    commits[1].files = ["src/x.py"]
    orig = F.log_commits
    F.log_commits = lambda repo, max_count=5000: commits
    try:
        X, y, meta, _ = F.build_dataset("fake")
    finally:
        F.log_commits = orig
    assert len(X) == 1            # only the non-fix commit
    assert meta[0]["hash"] == "a" * 40
    assert y[0] == 1.0            # labeled by the later fix


def test_tiny_repo_raises_valueerror(tmp_path):
    r = tmp_path / "tiny"
    r.mkdir()
    for cmd in ("git init", "git config user.email t@t", "git config user.name t"):
        subprocess.run(cmd.split(), cwd=r, capture_output=True)
    (r / "f.txt").write_text("hello")
    subprocess.run(["git", "add", "."], cwd=r, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=r, capture_output=True)
    from predict.pipeline import analyze_repo
    try:
        analyze_repo(str(r))
        raised = False
    except ValueError as e:
        raised = True
        assert "not suitable" in str(e)
    assert raised, "tiny repo must fail with a friendly ValueError"
