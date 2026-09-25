"""Stateful vs naive on IncidentDesk at increasing horizons.

    python -m lha.bench.run                       # offline, deterministic policies
    python -m lha.bench.run --live --sizes 50,200 # real model (Bedrock or Liquid via env)
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from ..agent import NaiveAgent, StatefulAgent
from ..compactor import Compactor
from ..config import load_dotenv
from ..llm import ScriptedLLM, from_env, summarizer_from_env
from ..sinks import JsonlSink, sinks_from_env
from ..tools import ToolBox
from .incident_desk import IncidentDesk, naive_policy, stateful_policy


def run_one(kind: str, n: int, seed: int, budget: int, live: bool, out: Path) -> dict:
    env = IncidentDesk(n_messages=n, seed=seed)
    if live:
        llm = from_env()
    else:
        llm = ScriptedLLM(stateful_policy if kind == "stateful" else naive_policy, name=f"scripted-{kind}")
    run_id = f"{kind}-n{n}-s{seed}"
    shutil.rmtree(out / "runs" / run_id, ignore_errors=True)  # benchmarks start fresh (agents would resume)
    sinks = [JsonlSink(out / "steps.jsonl"), *sinks_from_env()]
    common = dict(llm=llm, tools=ToolBox(env.tools()), goal=env.goal(), run_dir=out / "runs",
                  run_id=run_id, sinks=sinks, max_steps=n + 60)
    if kind == "stateful":
        agent = StatefulAgent(**common, compactor=Compactor(budget_tokens=budget,
                              summarizer=summarizer_from_env() if live else None))
    else:
        agent = NaiveAgent(**common)
    answer = agent.run()
    p = agent.stats.prompt_tokens_per_step
    return {
        "agent": kind, "messages": n, "seed": seed, "score": env.score(answer), "steps": agent.stats.steps,
        "total_input_tokens": agent.stats.input_tokens, "peak_prompt_tokens": max(p) if p else 0,
        "final_prompt_tokens": p[-1] if p else 0, "parse_errors": agent.stats.parse_errors,
        "curve": p,
    }


def main(argv=None) -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="50,200,1000,3000")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--budget", type=int, default=450, help="stateful working-state token budget")
    ap.add_argument("--live", action="store_true", help="use a real model from env instead of scripted policies")
    ap.add_argument("--out", default="results")
    ap.add_argument("--skip-naive-above", type=int, default=10**9, help="naive gets expensive fast with --live")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "steps.jsonl").unlink(missing_ok=True)
    rows = []
    for n in map(int, a.sizes.split(",")):
        for seed in map(int, a.seeds.split(",")):
            for kind in ("stateful", "naive"):
                if kind == "naive" and n > a.skip_naive_above:
                    continue
                r = run_one(kind, n, seed, a.budget, a.live, out)
                rows.append(r)
                print(f"{kind:9s} n={n:<5d} seed={seed} score={r['score']:.2f} steps={r['steps']:<5d} "
                      f"peak_prompt={r['peak_prompt_tokens']:<7d} total_in={r['total_input_tokens']:,}")
    (out / "results.json").write_text(json.dumps(rows, indent=1))
    (out / "report.md").write_text(report(rows))
    print(f"\nwrote {out}/results.json and {out}/report.md")
    print(report(rows))


def report(rows: list[dict]) -> str:
    lines = ["| messages | agent | score | steps | peak prompt tok | total input tok | cost vs naive |",
             "|---:|---|---:|---:|---:|---:|---:|"]
    naive = {(r["messages"], r["seed"]): r["total_input_tokens"] for r in rows if r["agent"] == "naive"}
    for r in rows:
        base = naive.get((r["messages"], r["seed"]))
        ratio = f"{r['total_input_tokens'] / base:.1%}" if base else "n/a"
        lines.append(f"| {r['messages']} | {r['agent']} | {r['score']:.2f} | {r['steps']} | "
                     f"{r['peak_prompt_tokens']:,} | {r['total_input_tokens']:,} | {ratio} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
