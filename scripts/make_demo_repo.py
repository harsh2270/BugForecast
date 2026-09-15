"""Generate a synthetic git repo with realistic defect dynamics for the demo.

Simulates: a hot core module, an inexperienced author, Friday-night commits,
and fixes that follow risky commits - so the miner can learn real signal.
"""
import os
import random
import subprocess

random.seed(42)
REPO = os.path.join(os.path.dirname(__file__), "..", "demo_repo")
REPO = os.path.abspath(REPO)

FILES = [f"src/module_{i}.py" for i in range(1, 11)] + ["src/core.py", "src/api.py"]
AUTHORS = ["alice", "bob", "carol", "dave"]
T0 = 1_600_000_000  # ~2020-09

SUBJECTS_NORMAL = ["add feature", "update docs", "refactor helper",
                   "improve logging", "cleanup imports", "bump version"]


def git(*args, env=None):
    e = dict(os.environ, GIT_AUTHOR_DATE="", GIT_COMMITTER_DATE="")
    e.update(env or {})
    subprocess.run(["git", "-C", REPO, *args], check=True, capture_output=True, env=e)


def run():
    if os.path.exists(REPO):
        subprocess.run(["rm", "-rf", REPO], check=True)
    os.makedirs(os.path.join(REPO, "src"))
    git("init", "-q")
    git("config", "user.email", "demo@bugforecast.dev")
    git("config", "user.name", "demo")
    for f in FILES:
        with open(os.path.join(REPO, f), "w") as fh:
            fh.write("# init\n")
    git("add", "-A")

    ts = T0
    risky_until = 0  # if ts < risky_until, next fix lands soon
    pending_fix_files: list[str] = []
    for i in range(400):
        ts += random.randint(3600, 3600 * 30)
        hour = random.randint(0, 23)
        author = random.choice(AUTHORS)
        env = {"GIT_AUTHOR_DATE": f"{ts + hour*60} +0000",
               "GIT_COMMITTER_DATE": f"{ts + hour*60} +0000"}

        if pending_fix_files and random.random() < 0.6:
            files = pending_fix_files
            subject = f"fix crash in {files[0].split('/')[-1]}"
            pending_fix_files = []
        else:
            n = random.choices([1, 2, 5], weights=[70, 20, 10])[0]
            files = random.sample(FILES, n)
            if "src/core.py" in files and random.random() < 0.5:
                files = ["src/core.py"] + files[:1]
            subject = f"{random.choice(SUBJECTS_NORMAL)} ({i})"
            if author == "dave" and (hour >= 22 or hour <= 4) and "src/core.py" in files:
                pending_fix_files = files  # this one will need a fix

        for f in files:
            with open(os.path.join(REPO, f), "a") as fh:
                fh.write(f"\n# change {i}\ndef fn_{i}():\n    pass\n")
        git("add", "-A")
        git("commit", "-q", "-m", subject, env=env)

    print(f"demo repo ready at {REPO} ({i+1} commits)")


if __name__ == "__main__":
    run()
