"""The agent loop, in two flavours that share everything except memory:

StatefulAgent  - prompt = rendered WorkingState (bounded). The model edits its
                 own state with explicit ops; old material goes to the archive.
                 State is checkpointed every step, so a run can be killed and
                 resumed hours or days later from exactly where it was.
NaiveAgent     - the usual baseline: prompt = goal + the full transcript.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .archive import Archive
from .compactor import Compactor
from .llm import LLM, Usage
from .state import OPS_HELP, WorkingState, apply_ops
from .tokens import estimate_tokens
from .tools import ToolBox, recall_tool

RESPONSE_FORMAT = """\
Reply with ONE JSON object and nothing else:
{"thought": "<one or two sentences>",
 "state_ops": [ ...edit ops... ],
 "action": {"tool": "<tool name or finish>", "args": {...}}}
Use {"tool": "finish", "args": {"answer": ...}} when the goal is achieved."""


def parse_response(text: str) -> dict:
    """Extract the first balanced JSON object from model output."""
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in response")
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if in_str:
            esc = (ch == "\\") and not esc
            if ch == '"' and not esc:
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("unbalanced JSON in response")


@dataclass
class RunStats:
    steps: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    prompt_tokens_per_step: list[int] = field(default_factory=list)
    parse_errors: int = 0
    op_errors: int = 0

    def add(self, u: Usage) -> None:
        self.input_tokens += u.input_tokens
        self.output_tokens += u.output_tokens
        self.prompt_tokens_per_step.append(u.input_tokens)


class _BaseAgent:
    kind = "base"

    def __init__(self, llm: LLM, tools: ToolBox, goal: str, run_dir: str | Path | None = None,
                 run_id: str | None = None, sinks=(), max_steps: int = 200, verbose: bool = False):
        self.llm = llm
        self.tools = tools
        self.goal = goal
        self.run_id = run_id or uuid.uuid4().hex[:10]
        self.run_dir = Path(run_dir) / self.run_id if run_dir else None
        self.sinks = list(sinks)
        self.archive = Archive(self.run_dir / "archive.jsonl" if self.run_dir else None, self.run_id, self.sinks)
        self.max_steps = max_steps
        self.verbose = verbose
        self.stats = RunStats()

    def system_prompt(self) -> str:
        raise NotImplementedError

    def _emit(self, rec: dict) -> None:
        rec = {"type": "step", "run_id": self.run_id, "agent": self.kind, "model": self.llm.name,
               "ts": time.time(), **rec}
        for s in self.sinks:
            s.emit(rec)

    def run(self) -> Any:
        while not self.finished and self.stats.steps < self.max_steps:
            self.step()
        self.archive.flush()
        return self.answer


class StatefulAgent(_BaseAgent):
    kind = "stateful"

    def __init__(self, *args, compactor: Compactor | None = None, **kw):
        super().__init__(*args, **kw)
        self.compactor = compactor or Compactor()
        self.tools.add(recall_tool(self.archive))
        self.state = self._load() or WorkingState(goal=self.goal)
        self._feedback = ""

    # ---------------------------------------------------------------- persistence
    def _ckpt(self) -> Path | None:
        return self.run_dir / "state.json" if self.run_dir else None

    def _load(self) -> WorkingState | None:
        p = self._ckpt()
        if p and p.exists():
            return WorkingState.from_dict(json.loads(p.read_text()))
        return None

    def _save(self) -> None:
        p = self._ckpt()
        if p:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.state.to_dict(), default=str, indent=1))
            os.replace(tmp, p)  # atomic: a crash never leaves a half-written state

    @property
    def finished(self) -> bool:
        return self.state.done

    @property
    def answer(self) -> Any:
        return self.state.answer

    def system_prompt(self) -> str:
        return (
            "You are a long-running agent. You have NO conversation history: the working state below is "
            "your entire memory. Anything you do not write into it is gone (except via `recall`). "
            "Keep it small, current and true: overwrite stale facts, drop what no longer matters, "
            "close finished tasks.\n\n" + OPS_HELP + "\nTools:\n" + self.tools.help() + "\n\n" + RESPONSE_FORMAT
        )

    def step(self) -> None:
        s = self.state
        s.step += 1
        prompt = s.render() + (f"\n\n# FEEDBACK FROM LAST STEP\n{self._feedback}" if self._feedback else "")
        self._feedback = ""
        if self.verbose and self.stats.steps == 0:
            print(f"[{self.run_id}] calling {self.llm.name} (first reply can take a while)...", flush=True)
        t0 = time.time()
        text, usage = self.llm.complete(self.system_prompt(), prompt)
        self._last_secs = time.time() - t0
        self.stats.steps += 1
        self.stats.add(usage)

        tool, args, op_errors = "none", {}, []
        try:
            resp = parse_response(text)
            op_errors = apply_ops(s, resp.get("state_ops", []),
                                  lambda kind, key, payload: self._archive(kind, key, payload))
            action = resp.get("action") or {}
            tool, args = action.get("tool", "none"), action.get("args", {}) or {}
        except (ValueError, json.JSONDecodeError) as e:
            self.stats.parse_errors += 1
            self._feedback = f"Your last reply was not valid JSON ({e}). Follow the response format exactly."

        if op_errors:
            self.stats.op_errors += len(op_errors)
            self._feedback += "\nState op errors: " + "; ".join(op_errors)

        if tool == "finish":
            s.done, s.answer = True, args.get("answer")
            s.observation = ""
        elif tool != "none":
            obs = self.tools.call(tool, args)
            self._archive("observation", tool, {"args": args, "result": obs})
            s.observation = f"{tool}({json.dumps(args, default=str)[:200]}) ->\n{obs}"

        evicted = self.compactor.enforce(s, self.archive)
        self._save()
        self._emit({
            "step": s.step, "tool": tool, "prompt_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
            "state_tokens": s.tokens(), "n_facts": len(s.facts), "n_tasks": len(s.tasks),
            "n_archived": len(self.archive.items), "evictions": len(evicted), "op_errors": len(op_errors),
        })
        if self.verbose:
            result = s.observation.split("->\n", 1)[-1].splitlines()[0][:100] if s.observation else ""
            flag = " PARSE ERROR" if self._feedback.startswith("Your last reply") else ""
            print(f"[{self.run_id} step {s.step}] {self._last_secs:.1f}s tool={tool} prompt={usage.input_tokens}tok "
                  f"out={usage.output_tokens}tok facts={len(s.facts)}{flag} | {result} {'; '.join(evicted)}", flush=True)

    def _archive(self, kind: str, key: str, payload: dict) -> None:
        self.archive.put(self.state.step, kind, key, payload)


class NaiveAgent(_BaseAgent):
    """Baseline: replays the whole transcript every step (no state, no eviction)."""

    kind = "naive"

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.history: list[str] = []
        self._done, self._answer = False, None

    @property
    def finished(self) -> bool:
        return self._done

    @property
    def answer(self) -> Any:
        return self._answer

    def system_prompt(self) -> str:
        return ("You are an agent. The transcript below is your memory.\nTools:\n" + self.tools.help()
                + "\n\n" + RESPONSE_FORMAT + '\n(You may omit "state_ops".)')

    def step(self) -> None:
        prompt = f"# GOAL\n{self.goal}\n\n# TRANSCRIPT\n" + "\n".join(self.history)
        text, usage = self.llm.complete(self.system_prompt(), prompt)
        self.stats.steps += 1
        self.stats.add(usage)
        tool = "none"
        try:
            action = parse_response(text).get("action") or {}
            tool, args = action.get("tool", "none"), action.get("args", {}) or {}
        except (ValueError, json.JSONDecodeError) as e:
            self.stats.parse_errors += 1
            self.history.append(f"[step {self.stats.steps}] ERROR: invalid JSON ({e})")
        if tool == "finish":
            self._done, self._answer = True, args.get("answer")
        elif tool != "none":
            obs = self.tools.call(tool, args)
            self.history.append(f"[step {self.stats.steps}] you: {text.strip()}\n[step {self.stats.steps}] {tool} -> {obs}")
        self._emit({"step": self.stats.steps, "tool": tool, "prompt_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens, "state_tokens": estimate_tokens(prompt)})
        if self.verbose:
            print(f"[{self.run_id} step {self.stats.steps}] tool={tool} prompt={usage.input_tokens}tok", flush=True)
