"""Command line.

    python -m lha.cli run "Track the 5 largest open-weight model releases this month" --run-id research1
    python -m lha.cli run --run-id research1          # resume after a crash / days later
    python -m lha.cli state research1                 # print the working state
    python -m lha.cli recall research1 "mistral"      # search cold storage
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .agent import StatefulAgent
from .archive import Archive
from .compactor import Compactor
from .llm import from_env, summarizer_from_env
from .sinks import JsonlSink, sinks_from_env
from .state import WorkingState
from .tools import ToolBox, nimble_tools

RUNS = Path(os.environ.get("LHA_RUNS_DIR", "runs"))


def cmd_run(a) -> None:
    ckpt = RUNS / a.run_id / "state.json"
    goal = a.goal or (json.loads(ckpt.read_text())["goal"] if ckpt.exists() else None)
    if not goal:
        raise SystemExit("give a goal (or a --run-id with an existing checkpoint to resume)")
    tools = ToolBox(nimble_tools() if os.environ.get("NIMBLE_CREDENTIALS") else [])
    agent = StatefulAgent(from_env(), tools, goal, run_dir=RUNS, run_id=a.run_id,
                          sinks=[JsonlSink(RUNS / a.run_id / "steps.jsonl"), *sinks_from_env()],
                          max_steps=a.max_steps, verbose=True,
                          compactor=Compactor(budget_tokens=a.budget, summarizer=summarizer_from_env()))
    if agent.state.step:
        print(f"resuming {a.run_id} at step {agent.state.step}")
    print(json.dumps(agent.run(), indent=1, default=str))


def cmd_state(a) -> None:
    print(WorkingState.from_dict(json.loads((RUNS / a.run_id / "state.json").read_text())).render())


def cmd_recall(a) -> None:
    for h in Archive(RUNS / a.run_id / "archive.jsonl", a.run_id).recall(a.query, k=a.k):
        print(f"[step {h.step}] {h.kind} {h.key}: {json.dumps(h.payload, default=str)[:300]}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="lha")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("goal", nargs="?")
    r.add_argument("--run-id", required=True)
    r.add_argument("--budget", type=int, default=2000)
    r.add_argument("--max-steps", type=int, default=100)
    r.set_defaults(fn=cmd_run)
    s = sub.add_parser("state")
    s.add_argument("run_id")
    s.set_defaults(fn=cmd_state)
    q = sub.add_parser("recall")
    q.add_argument("run_id")
    q.add_argument("query")
    q.add_argument("-k", type=int, default=5)
    q.set_defaults(fn=cmd_recall)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
