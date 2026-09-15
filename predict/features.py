"""Extract per-commit features and defect labels from a git repository.

Approach: walk `git log` once. Each non-fix commit that touches files gets a
feature vector. A commit is a "fix" if its subject matches bug-fix heuristics;
for every file a fix touches, earlier commits touching that same file within
WINDOW days are labeled defective (classic SZZ-style weak label). Fix commits
themselves are excluded from training rows so the model cannot cheat by
recognizing fix-style subjects.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field

FIX_HINTS = ("fix", "bug", "hotfix", "regression", "patch", "crash", "error")

FEATURE_NAMES = [
    "la",           # lines added (log-scaled)
    "ld",           # lines deleted (log-scaled)
    "nfiles",       # files touched
    "nfiles_ratio", # files touched / repo file count so far (blast radius proxy)
    "add_ratio",    # additions / total churn: new code vs cleanup
    "author_exp",   # author's commit count before this commit (log-scaled)
    "author_fam",   # author's prior commits touching these files
    "file_hot",     # mean prior change-count of touched files
    "file_age",     # mean days since each file's first commit
    "hour_sin",
    "hour_cos",
]


def _git(repo: str, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", repo, *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {out.stderr.strip()}")
    return out.stdout


@dataclass
class Commit:
    hash: str
    author: str
    ts: int          # unix seconds
    subject: str
    files: list[str] = field(default_factory=list)
    la: int = 0
    ld: int = 0

    @property
    def is_fix(self) -> bool:
        s = self.subject.lower()
        return any(h in s for h in FIX_HINTS) and not s.startswith("merge")


def log_commits(repo: str, max_count: int = 5000) -> list[Commit]:
    """Parse `git log --numstat` into Commit records, oldest first."""
    fmt = "%x01%H%x02%an%x02%at%x02%s%x1e"
    raw = _git(repo, "log", f"--max-count={max_count}",
               f"--pretty=format:{fmt}", "--numstat", "-M")
    commits: list[Commit] = []
    cur: Commit | None = None
    for line in raw.splitlines():
        if "\x01" in line:
            fields = line.split("\x02", 3)
            h = fields[0].lstrip("\x01")
            an, at_s, subj = fields[1], fields[2], fields[3].rstrip("\x1e")
            cur = Commit(h, an, int(at_s), subj.strip())
            commits.append(cur)
        elif cur is not None and line.strip():
            parts = line.split("\t")
            if len(parts) >= 3:
                la, ld, path = parts[0], parts[1], parts[2]
                if la != "-":
                    cur.la += int(la)
                    cur.ld += int(ld)
                cur.files.append(path.split(" => ")[-1])
    commits.reverse()
    return commits


def build_dataset(repo: str, max_count: int = 5000):
    """Return (X, y, meta) where each row is one non-fix commit.

    Label = 1 if a later fix commit touched any file this commit touched
    within WINDOW days (SZZ-style blame link).
    """
    commits = log_commits(repo, max_count)
    commits = [c for c in commits if c.files]

    file_first_seen: dict[str, int] = {}
    file_touch_count: dict[str, int] = {}
    author_count: dict[str, int] = {}
    author_file: dict[tuple[str, str], int] = {}

    fixes: list[tuple[int, set[str]]] = []  # (index, files) of fix commits

    rows, labels, meta = [], [], []
    row_of: dict[int, int] = {}  # commit index -> training-row index
    import math

    for i, c in enumerate(commits):
        if c.is_fix:
            fixes.append((i, set(c.files)))
            continue  # fix commits are excluded from training rows and from
                      # features like file_hotness, so the model cannot cheat
                      # by recognizing fix-style subjects

        row_of[i] = len(rows)
        hour = (c.ts // 3600) % 24
        touched = set(c.files)
        known = [f for f in touched if f in file_first_seen]
        ages = [(c.ts - file_first_seen[f]) / 86400.0 for f in known]
        hots = [file_touch_count.get(f, 0) for f in touched]
        fam = sum(author_file.get((c.author, f), 0) for f in touched)

        nfiles_ratio = len(touched) / max(1, len(file_first_seen))

        rows.append([
            math.log1p(c.la),
            math.log1p(c.ld),
            len(touched),
            nfiles_ratio,
            c.la / max(1, c.la + c.ld),   # add/delete balance: pure-addition commits (new code) vs pure-deletion (cleanup)
            math.log1p(author_count.get(c.author, 0)),
            math.log1p(fam),
            (sum(hots) / len(hots)) if hots else 0.0,
            (sum(ages) / len(ages)) if ages else 0.0,
            math.sin(2 * math.pi * hour / 24),
            math.cos(2 * math.pi * hour / 24),
        ])
        labels.append(0)  # filled in second pass
        meta.append({"hash": c.hash, "subject": c.subject, "ts": c.ts})

        for f in touched:
            file_first_seen.setdefault(f, c.ts)
            file_touch_count[f] = file_touch_count.get(f, 0) + 1
            key = (c.author, f)
            author_file[key] = author_file.get(key, 0) + 1
        author_count[c.author] = author_count.get(c.author, 0) + 1

    WINDOW = 14 * 86400
    for i, files in fixes:
        for j in range(i):
            if commits[i].ts - commits[j].ts > WINDOW:
                continue
            if files & set(commits[j].files) and j in row_of:
                labels[row_of[j]] = 1

    import numpy as np
    X = np.asarray(rows, dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.asarray(labels, dtype=float)
    return X, y, meta, commits
