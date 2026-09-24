"""Explicit, mutable working state.

This is the core idea of the project: the agent does NOT carry its history
forward. Each step it sees a *rendered view* of a small, typed, editable state,
and it changes that state through explicit edit operations. Anything that falls
out of the state goes to the archive (cold storage) where it can be recalled
on demand, but is never replayed by default.

State sections, from most to least durable:

    goal        - immutable task statement (set once)
    facts       - key -> value beliefs with provenance; set_fact overwrites,
                  and the superseded value is archived, so stale beliefs never
                  sit next to current ones in the prompt
    tasks       - the plan: id, title, status, result
    questions   - open unknowns the agent still has to resolve
    focus       - one line: what the agent is doing right now
    notes       - volatile scratchpad, bounded ring buffer
    observation - ONLY the latest tool result (truncated); older ones are archived
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from .tokens import estimate_tokens

TASK_STATUSES = ("todo", "doing", "done", "blocked", "dropped")


@dataclass
class Fact:
    key: str
    value: Any
    source: str = ""
    step: int = 0
    confidence: float = 1.0
    pinned: bool = False
    last_used: int = 0  # step the fact was last written or referenced


@dataclass
class Task:
    id: str
    title: str
    status: str = "todo"
    result: str = ""
    parent: str | None = None
    step: int = 0


@dataclass
class WorkingState:
    goal: str = ""
    step: int = 0
    focus: str = ""
    facts: dict[str, Fact] = field(default_factory=dict)
    tasks: dict[str, Task] = field(default_factory=dict)
    questions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    observation: str = ""
    archived_keys: list[str] = field(default_factory=list)  # breadcrumbs to recall
    done: bool = False
    answer: Any = None
    max_notes: int = 8

    # ---------------------------------------------------------------- serialization
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "WorkingState":
        d = dict(d)
        d["facts"] = {k: Fact(**v) for k, v in d.get("facts", {}).items()}
        d["tasks"] = {k: Task(**v) for k, v in d.get("tasks", {}).items()}
        return cls(**d)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    # ---------------------------------------------------------------- rendering
    def render(self) -> str:
        """The ONLY view of the past the model gets each step."""
        out = [f"# GOAL\n{self.goal}", f"# STEP {self.step}"]
        if self.focus:
            out.append(f"# FOCUS\n{self.focus}")
        if self.tasks:
            lines = []
            for t in self.tasks.values():
                if t.status in ("done", "dropped"):
                    res = f" -> {t.result}" if t.result else ""
                    lines.append(f"- [{t.status}] {t.id}: {t.title}{res}")
                else:
                    lines.append(f"- [{t.status}] {t.id}: {t.title}")
            out.append("# PLAN\n" + "\n".join(lines))
        if self.facts:
            lines = []
            for f in sorted(self.facts.values(), key=lambda f: (not f.pinned, f.key)):
                pin = "*" if f.pinned else ""
                src = f" (src: {f.source}, step {f.step})" if f.source else f" (step {f.step})"
                lines.append(f"- {pin}{f.key} = {json.dumps(f.value, default=str)}{src}")
            out.append("# FACTS (current beliefs; superseded values are archived)\n" + "\n".join(lines))
        if self.questions:
            out.append("# OPEN QUESTIONS\n" + "\n".join(f"- {q}" for q in self.questions))
        if self.notes:
            out.append("# NOTES (volatile)\n" + "\n".join(f"- {n}" for n in self.notes))
        if self.archived_keys:
            shown = self.archived_keys[-12:]
            more = len(self.archived_keys) - len(shown)
            tail = f" (+{more} more)" if more > 0 else ""
            out.append("# ARCHIVED (use the recall tool to fetch)\n" + ", ".join(shown) + tail)
        if self.observation:
            out.append(f"# LAST OBSERVATION\n{self.observation}")
        return "\n\n".join(out)

    def tokens(self) -> int:
        return estimate_tokens(self.render())


# -------------------------------------------------------------------- edit ops
class StateOpError(ValueError):
    pass


ArchiveFn = Callable[[str, str, dict], None]  # (kind, key, payload)


def apply_ops(state: WorkingState, ops: list[dict], archive: ArchiveFn | None = None) -> list[str]:
    """Apply model-proposed edits. Returns human-readable errors (never raises),
    so a single bad op cannot derail a long run; errors are fed back next step."""
    errors: list[str] = []
    for op in ops or []:
        try:
            _apply_one(state, op, archive or (lambda *a: None))
        except (StateOpError, KeyError, TypeError) as e:
            errors.append(f"{op.get('op', '?')}: {e}")
    return errors


def _apply_one(s: WorkingState, op: dict, archive: ArchiveFn) -> None:
    kind = op.get("op")
    if kind == "set_fact":
        key = str(op["key"])
        old = s.facts.get(key)
        if old is not None and old.value != op.get("value"):
            archive("superseded_fact", key, asdict(old))
        s.facts[key] = Fact(
            key=key,
            value=op.get("value"),
            source=str(op.get("source", "")),
            step=s.step,
            confidence=float(op.get("confidence", 1.0)),
            pinned=bool(op.get("pin", old.pinned if old else False)),
            last_used=s.step,
        )
    elif kind == "drop_fact":
        key = str(op["key"])
        if key not in s.facts:
            raise StateOpError(f"no fact {key!r}")
        f = s.facts.pop(key)
        archive("dropped_fact", key, {**asdict(f), "reason": op.get("reason", "")})
        s.archived_keys.append(key)
    elif kind in ("pin", "unpin"):
        key = str(op["key"])
        if key not in s.facts:
            raise StateOpError(f"no fact {key!r}")
        s.facts[key].pinned = kind == "pin"
    elif kind == "add_task":
        tid = str(op.get("id") or f"t{len(s.tasks) + 1}")
        s.tasks[tid] = Task(id=tid, title=str(op["title"]), parent=op.get("parent"), step=s.step)
    elif kind == "update_task":
        tid = str(op["id"])
        if tid not in s.tasks:
            raise StateOpError(f"no task {tid!r}")
        status = op.get("status", s.tasks[tid].status)
        if status not in TASK_STATUSES:
            raise StateOpError(f"bad status {status!r}")
        s.tasks[tid].status = status
        if "result" in op:
            s.tasks[tid].result = str(op["result"])
    elif kind == "add_question":
        q = str(op["text"])
        if q not in s.questions:
            s.questions.append(q)
    elif kind == "resolve_question":
        q = str(op["text"])
        if q in s.questions:
            s.questions.remove(q)
            archive("resolved_question", q[:60], {"question": q, "answer": op.get("answer")})
    elif kind == "note":
        s.notes.append(str(op["text"]))
        while len(s.notes) > s.max_notes:
            archive("note", f"note@{s.step}", {"text": s.notes.pop(0)})
    elif kind == "clear_notes":
        for n in s.notes:
            archive("note", f"note@{s.step}", {"text": n})
        s.notes.clear()
    elif kind == "set_focus":
        s.focus = str(op["text"])
    else:
        raise StateOpError(f"unknown op {kind!r}")


OPS_HELP = """\
State edit ops (list them in "state_ops"; applied before your action runs):
  {"op":"set_fact","key":K,"value":V,"source":S,"pin":bool}  overwrite a belief (old value is archived)
  {"op":"drop_fact","key":K,"reason":R}                        remove a belief that no longer matters
  {"op":"pin","key":K} / {"op":"unpin","key":K}                pinned facts are never evicted
  {"op":"add_task","id":ID,"title":T} / {"op":"update_task","id":ID,"status":todo|doing|done|blocked|dropped,"result":R}
  {"op":"add_question","text":Q} / {"op":"resolve_question","text":Q,"answer":A}
  {"op":"note","text":T}   volatile scratch; oldest notes roll off automatically
  {"op":"clear_notes"}
  {"op":"set_focus","text":T}
"""
