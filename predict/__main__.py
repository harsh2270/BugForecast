"""CLI: train / score / report / web — defect-risk analysis for git repos."""
from __future__ import annotations

import argparse

from .pipeline import analyze_repo


def _show(stage: str, pct: int) -> None:
    print(f"  [{pct:3d}%] {stage}")


def cmd_train(repo: str, max_count: int) -> None:
    result = analyze_repo(repo, max_count, progress=_show)
    print("\nbacktest results (time-safe, out-of-fold):")
    for m in result["report"]["models"]:
        print(f"  {m['name']:26s} AUC {m['auc']:.3f}  F1 {m['f1']:.3f}  "
              f"Brier {m['brier']:.3f}")
    print("\nrun `python -m predict report --repo <path>` for the dashboard, "
          "or `python -m predict web` for the browser app")


def cmd_report(repo: str, out_path: str) -> None:
    result = analyze_repo(repo, progress=_show)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result["html"])
    print(f"dashboard written to {out_path}")


def cmd_score(repo: str, top: int) -> None:
    result = analyze_repo(repo, progress=_show)
    risky = result["report"]["risky"]
    print(f"\ntop {min(top, len(risky))} riskiest commits (stacked ensemble):")
    for r in risky[:top]:
        print(f"  {r['p']*100:5.1f}%  {r['hash'][:9]}  {r['subject'][:64]}")
    print("\nprioritize review of the above; run `report` for the full dashboard")


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="bugforecast",
        description="Predictive defect-risk engine for git repos")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("train", "score", "report", "web"):
        s = sub.add_parser(name)
        if name != "web":
            s.add_argument("--repo", default=".")
        if name == "train":
            s.add_argument("--max-count", type=int, default=5000)
        if name == "score":
            s.add_argument("--top", type=int, default=10)
        if name == "report":
            s.add_argument("--out", default="bugforecast_report.html")
        if name == "web":
            s.add_argument("--port", type=int, default=5050)
    args = ap.parse_args()
    try:
        if args.cmd == "train":
            cmd_train(args.repo, args.max_count)
        elif args.cmd == "score":
            cmd_score(args.repo, args.top)
        elif args.cmd == "report":
            cmd_report(args.repo, args.out)
        else:
            from .server import run as run_server
            run_server(port=args.port)
    except ValueError as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
