"""Command line.

    python -m lha.cli run "Track the 5 largest open-weight model releases this month" --run-id research1
    python -m lha.cli run --run-id research1          # resume after a crash / days later
    python -m lha.cli run "Design a 2U rack blanking panel" --run-id cad1 --cad   # + build123d CAD tools
    python -m lha.cli state research1                 # print the working state
    python -m lha.cli recall research1 "mistral"      # search cold storage
    python -m lha.cli models                          # which Bedrock Claude models this account can call
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
    if a.cad:
        from .cad import cad_tools  # needs build123d (pip install -e ".[cad]")

        for t in cad_tools(RUNS / a.run_id / "cad"):
            tools.add(t)
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


# Newest first. Short IDs go through Bedrock's Messages API, versioned ones through Converse.
BEDROCK_CANDIDATES = [
    "anthropic.claude-opus-5-5", "anthropic.claude-opus-5", "anthropic.claude-sonnet-5",
    "anthropic.claude-opus-4-8", "anthropic.claude-opus-4-7", "anthropic.claude-haiku-4-5",
    "{geo}.anthropic.claude-opus-4-1-20250805-v1:0", "{geo}.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "{geo}.anthropic.claude-haiku-4-5-20251001-v1:0",
]


def _reason(e: Exception) -> str:
    body = getattr(e, "body", None)  # anthropic SDK errors
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        return str(body["error"].get("message", e))
    resp = getattr(e, "response", None)  # botocore ClientError
    if isinstance(resp, dict) and "Error" in resp:
        return str(resp["Error"].get("Message", e))
    return str(e).splitlines()[0]


def cmd_models(a) -> None:
    """Send a tiny request to each candidate and report which ones this account can use (costs < $0.01)."""
    from .llm import BedrockLLM, BedrockMantleLLM

    geo = {"eu": "eu", "ap": "apac"}.get(a.region[:2], "us")
    ok = []
    for mid in (c.format(geo=geo) for c in BEDROCK_CANDIDATES):
        try:
            llm = BedrockLLM(mid, a.region) if mid.endswith(":0") else BedrockMantleLLM(mid, a.region)
            llm.complete("Reply with one word.", "hi", max_tokens=64)
            ok.append(mid)
            print(f"OK   {mid}")
        except Exception as e:  # noqa: BLE001 - any failure means "not usable here"
            print(f"no   {mid}  ({type(e).__name__}: {_reason(e)[:110]})")
    if ok:
        print(f"\nexport AWS_REGION={a.region}\nexport LHA_BEDROCK_MODEL_ID={ok[0]}")
    else:
        print(f"\nNo Claude model is usable in {a.region}. Request access in the Bedrock console "
              "(Model access / Model catalog) or try --region us-west-2.")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="lha")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("goal", nargs="?")
    r.add_argument("--run-id", required=True)
    r.add_argument("--budget", type=int, default=2000)
    r.add_argument("--max-steps", type=int, default=100)
    r.add_argument("--cad", action="store_true", help="add build123d CAD tools (cad_build, cad_measure, ...)")
    r.set_defaults(fn=cmd_run)
    s = sub.add_parser("state")
    s.add_argument("run_id")
    s.set_defaults(fn=cmd_state)
    q = sub.add_parser("recall")
    q.add_argument("run_id")
    q.add_argument("query")
    q.add_argument("-k", type=int, default=5)
    q.set_defaults(fn=cmd_recall)
    m = sub.add_parser("models", help="probe which Bedrock Claude models this account can call")
    m.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"))
    m.set_defaults(fn=cmd_models)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
