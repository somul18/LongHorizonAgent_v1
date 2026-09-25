"""Memory State Inspector: watch an agent replace its history with a small, mutable state.

    python -m lha.inspect                              # replay the most recent run in ./results
    python -m lha.inspect design_stateful-n3000-s0 --speed 60
    python -m lha.inspect live_design_stateful-n200-s0 --follow   # while a live run is going
    python -m lha.bench.run --env design --live --sizes 200 --inspect   # render during the run

Every step the stateful agent appends a record to runs/<run>/trace.jsonl (see
StatefulAgent._trace). This module renders those records as one terminal screen:
the incoming event, the working state (current facts, plan, open questions), the state
operation it caused (what was overwritten, what was superseded, evicted or recalled),
the memory counters against the naive transcript, and finally the path from
thousands of events to a validated CAD part.
"""

from __future__ import annotations

import argparse
import json
import textwrap
import os
import re
import shutil
import sys
import time
from pathlib import Path

W = 78

# ------------------------------------------------------------------ styling
_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _c(code: str, text: str) -> str:
    return f"\x1b[{code}m{text}\x1b[0m" if _COLOR else text


def bold(t): return _c("1", t)
def dim(t): return _c("2", t)
def green(t): return _c("32", t)
def yellow(t): return _c("33", t)
def red(t): return _c("31", t)
def blue(t): return _c("34", t)
def orange(t): return _c("38;5;208", t)


def _vis(s: str) -> int:
    return len(re.sub(r"\x1b\[[0-9;]*m", "", s))


def rule(title: str = "") -> str:
    return bold(title) + ("\n" if title else "") + dim("─" * W)


def fmt_tok(n: float) -> str:
    return f"{n / 1000:.1f}K" if n >= 10_000 or (n >= 1000 and n % 1000) else f"{int(n):,}"


def _val(v) -> str:
    s = v if isinstance(v, str) else json.dumps(v)
    return s if len(s) <= 28 else s[:27] + "…"


UNITS = {"length": "mm", "width": "mm", "thickness": "mm", "hole_diameter": "mm", "mass_g": "g"}


def _with_unit(key: str, v) -> str:
    u = UNITS.get(key.rsplit(".", 1)[-1])
    return f"{_val(v)} {u}" if u and not str(v).endswith(u) else _val(v)


# ------------------------------------------------------------------ model of the run
class Inspector:
    """Keeps a little context across records (the entity in focus, the previous event)."""

    def __init__(self, run_id: str = "", total_messages: int | None = None, summary: dict | None = None,
                 intent: dict | None = None):
        self.run_id = run_id
        self.total = total_messages
        self.summary = summary
        self.prev: dict | None = None
        self.entity: str | None = None
        self.inbox = (0, total_messages)
        self.history: list[tuple[int, int, int]] = []  # (step, prompt, naive) for the bar history
        self.builds: dict[str, dict] = {}  # part -> the agent's own check of its latest build
        self.intent = intent  # {"part", "request", "interpretation"} when a part started as design intent
        self.intent_changes: list[tuple[str, str, object, object]] = []  # (source, attr, old, new) since the intent
        self.archived_facts = 0  # facts moved to the archive so far: superseded, evicted or dropped
        self.recalls = 0         # recall operations so far
        from .describe import tracker_for

        self.design = tracker_for(run_id)  # "Current Design Understanding", derived from the state
        self.bench = ("DESIGNDESK" if "design_" in run_id else
                      "INCIDENTDESK" if re.match(r"(live_)?(stateful|naive)-n\d+", run_id) else "LONG HORIZON AGENT")

    def feed(self, rec: dict) -> str:
        frame = self.render(rec)
        self.prev = rec
        return frame

    # -------------------------------------------------------------- sections
    def render(self, rec: dict) -> str:
        event = (self.prev or {}).get("observation", "")
        if m := re.match(r"msg (\d+)/(\d+):", event):
            self.inbox = (int(m[1]), int(m[2]))
        changed = {k for k, *_ in rec["changes"]}
        first = next((op.get("key") for op in rec["ops"] or [] if op.get("op") == "set_fact"), None)
        if first or changed:  # follow the part the agent is working on
            self.entity = (first or sorted(changed)[0]).split(".")[0]
        if built := re.search(r"built '([\w-]+)': (\{.*\})", event):
            self.builds[built[1]] = json.loads(built[2])
        self.history.append((rec["step"], rec["prompt_tokens"], rec["naive_tokens"]))
        self.archived_facts += sum(1 for kind, *_ in rec["archived"] if kind.endswith("_fact"))
        self.recalls += rec["tool"] == "recall"

        design = self.design.feed(rec)
        if self.intent:
            p = self.intent["part"] + "."
            srcs = rec.get("sources", {})
            for k, old, new in rec["changes"]:
                src = srcs.get(k, "")
                # every change that came from an ECO, even when the old value was archived at the time;
                # not the intent itself, not recalls (same value, back from the archive), not build results
                if k.startswith(p) and old != new and not k.endswith(".mass_g") and src and "recall" not in src \
                        and src not in ("design_intent", "cad_build"):
                    self.intent_changes.append((src, k[len(p):], old, new))
        out = [self._header(rec), ""]
        if self.intent:
            out += [self._intent(rec), ""]
        out += [self._event(event, rec), "", self._state(rec, changed), ""]
        if design:
            out += [self._understanding(design), ""]
        out += [self._operation(rec), "", self._memory(rec)]
        if self._building(rec, event):
            out += ["", self._build(rec)]
        return "\n".join(out)

    def _header(self, rec: dict) -> str:
        read, total = self.inbox
        msg = f" · MESSAGE {read:,} / {total:,}" if total else ""
        left = bold(f"{self.bench} — STEP {rec['step']:,}{msg}")
        right = dim(rec.get("model", ""))
        return left + " " * max(1, W - _vis(left) - _vis(right)) + right

    def _event(self, event: str, rec: dict) -> str:
        lines = [rule("INCOMING EVENT")]
        text = re.sub(r"^msg \d+/\d+: ", "", event.splitlines()[0] if event else "(start of run)")
        prev_tool = (self.prev or {}).get("tool")
        if prev_tool == "recall":
            lines.append(green("RECALL RESULT (from the archive)"))
        elif prev_tool == "cad_build":
            lines.append(blue("BUILD RESULT"))
        if m := re.match(r"\[(ECO-\d+)\] ([\w-]+): (\w+) changed to ([\w.]+)", text):
            lines.append(bold(f"{m[1]} APPROVED"))
            old = next((o for k, o, _ in rec["changes"] if k == f"{m[2]}.{m[3]}"), None)
            arrow = f"{_with_unit(m[3], old)} → " if old is not None else ""
            note = "" if old is not None else dim("   (not in working state before: first value, or old one archived)")
            lines.append(f"{m[2]}.{m[3]}: {arrow}{bold(_with_unit(m[3], m[4]))}{note}")
        elif text.startswith("INBOX EMPTY"):
            lines += [bold("INBOX EMPTY: build requests"), text.split("->", 1)[-1].strip()]
        elif rec["tool"] == "finish" or (self.prev or {}).get("tool") in ("cad_build", "recall"):
            lines.append(text[:W])
        else:
            lines += [text[:W], dim("no state change: noise, a rejected proposal, or already known")
                      if not rec["changes"] and self.prev and self.prev["tool"] == "next_message" else ""]
        return "\n".join(x for x in lines if x != "")

    def _state(self, rec: dict, changed: set[str]) -> str:
        facts, pinned = rec["facts"], set(rec["pinned"])
        head = bold("WORKING STATE")
        tok = f"~{rec['state_tokens']:,} tokens"
        lines = [head + " " * (W - _vis(head) - len(tok)) + tok, dim("─" * W), dim("CURRENT FACTS")]
        shown = sorted(k for k in facts if self.entity and k.startswith(self.entity + "."))
        if not shown:
            shown = sorted(facts)[:6]
        recalled = {op.get("key") for op in rec["ops"] or [] if op.get("op") == "set_fact" and "recall" in str(op.get("source", ""))}
        for k in shown:
            mark = green("  RECALLED") if k in recalled else yellow("  UPDATED") if k in changed else ""
            pin = " *" if k in pinned else "  "
            src = _provenance(rec.get("sources", {}).get(k, ""))
            lines.append(f"  {k:<32}{_with_unit(k, facts[k]):>12}{pin} {dim('← ' + src) if src else ''}{mark}")
        others = len(facts) - len(shown)
        parts = len({k.split('.')[0] for k in facts})
        lines.append(dim(f"  + {others} more facts across {parts} parts · {rec['archived_keys']} keys archived")
                     if others > 0 else dim(f"  {rec['archived_keys']} keys archived"))
        if rec["tasks"]:
            lines.append(dim("CURRENT PLAN"))
            icon = {"done": green("✓"), "doing": blue("▸"), "blocked": red("!"), "dropped": dim("×")}
            lines += [f"  {icon.get(st, '○')} {title}" for _, title, st in rec["tasks"]]
        lines.append(dim("OPEN QUESTIONS"))
        lines += [f"  ? {q}" for q in rec["questions"]] or ["  None"]
        return "\n".join(lines)

    def _intent(self, rec: dict) -> str:
        """Original design intent -> the ECOs that changed that part since -> its current spec."""
        part, facts = self.intent["part"], rec["facts"]
        head = bold("ORIGINAL DESIGN INTENT")
        lines = [head + " " * max(1, W - _vis(head) - len(part)) + part, dim("─" * W)]
        lines += [f'"{x}' if i == 0 else f" {x}" for i, x in enumerate(textwrap.wrap(self.intent["request"] + '"', W - 2))]
        spec0 = (self.intent.get("interpretation") or {}).get("facts", {})
        now = {a: facts.get(f"{part}.{a}") for a in ("length", "width", "thickness", "hole_diameter", "material")}
        lines.append(f"  {'intent':<12}{_spec_line(spec0)}")
        ch = self.intent_changes
        if len(ch) > 4:
            lines.append(dim(f"  … {len(ch) - 4} earlier change{'s' if len(ch) - 4 > 1 else ''}"))
        for src, attr, old, new in ch[-4:]:
            was = f"{_with_unit(attr, old)} → " if old is not None else dim("(was archived) ") + "→ "
            lines.append(f"  {_provenance(src):<12}{attr.replace('_', ' ')} {was}{_with_unit(attr, new)}")
        lines.append(f"  {bold('now'):<{12 + len(bold('now')) - 3}}{_spec_line(now)}")
        return "\n".join(lines)

    @staticmethod
    def _understanding(design: dict) -> str:
        head = bold("CURRENT DESIGN UNDERSTANDING")
        tag = design["part"]
        lines = [head + " " * max(1, W - _vis(head) - len(tag)) + tag, dim("─" * W)]
        lines += textwrap.wrap(design["text"], W)
        lines.append(dim("derived from the working state above; regenerated every step, never stored"))
        return "\n".join(lines)

    def _operation(self, rec: dict) -> str:
        lines = [rule("STATE MUTATION")]
        ops = rec["ops"] or []
        old_of = {k: old for k, old, _ in rec["changes"]}
        forks = 0
        for op in ops[:6]:
            kind = op.get("op", "?")
            key = op.get("key")
            if kind == "set_fact" and old_of.get(key) is not None and forks < 2:
                # an overwrite: the new value stays active, the old one goes to the archive
                forks += 1
                old, new = _with_unit(key, old_of[key]), _with_unit(key, op.get("value"))
                lines += [f"{bold('set_fact')}  {key}", f"          {old} → {bold(new)}",
                          f"          ├─ {blue('ACTIVE ')}  {new}", f"          └─ {orange('ARCHIVE')}  {old}  " + dim("(superseded)")]
            elif kind == "set_fact":
                pin = ", pin=True" if op.get("pin") else ""
                lines.append(f'set_fact("{op.get("key")}", {json.dumps(op.get("value"))}{pin})')
            elif kind == "update_task":
                lines.append(f'update_task("{op.get("id")}", {op.get("status")})')
            elif kind in ("add_question", "resolve_question", "note", "set_focus"):
                lines.append(f'{kind}("{str(op.get("text", ""))[:50]}")')
            else:
                lines.append(f'{kind}({", ".join(f"{k}={json.dumps(v)}" for k, v in op.items() if k != "op")[:60]})')
        if len(ops) > 6:
            lines.append(dim(f"… and {len(ops) - 6} more"))
        if not ops:
            lines.append(dim("none"))
        sup = [(k, v) for kind, k, v in rec["archived"] if kind == "superseded_fact"]
        evi = [k for kind, k, _ in rec["archived"] if kind in ("evicted_fact", "dropped_fact")]
        if len(sup) > forks:
            lines.append("Superseded:  " + ", ".join(f"{k} = {_with_unit(k, v)} → archive" for k, v in sup)[:W - 13])
        if evi:
            lines.append(orange("Evicted to archive (over budget):  ") + ", ".join(evi)[:W - 35])
        restored = [op.get("key") for op in ops if op.get("op") == "set_fact" and "recall" in str(op.get("source", ""))]
        if restored:
            lines.append(green("Restored from archive:  ") + ", ".join(restored)[:W - 24])
        lines.append(dim("Action:  ") + _action(rec))
        if rec["op_errors"]:
            lines.append(red("op errors: " + "; ".join(rec["op_errors"])[:W]))
        if rec.get("parse_error"):
            lines.append(red("reply was not valid JSON; the error goes back to the model next step"))
        return "\n".join(lines)

    def _memory(self, rec: dict) -> str:
        naive, active = rec["naive_tokens"], rec["prompt_tokens"]
        red_pct = 100 * (1 - active / naive) if naive else 0
        growing = len(self.history) > 1 and naive > self.history[-2][2]
        rows = [("Events processed", f"{rec['step']:,}"), ("Active state (prompt)", f"{active:,} tokens"),
                ("Naive context (est.)", f"{naive:,} tokens" + (" ↑" if growing else "  ")),
                ("Context reduction", f"{red_pct:.1f}%"), ("Archived facts", f"{self.archived_facts:,}"),
                ("Recall operations", f"{self.recalls:,}")]
        lines = [rule("LONG-HORIZON MEMORY")] + [f"{k:<30}{v:>22}" for k, v in rows]
        top = max(naive, active, 1)
        bw = W - 18

        def bar(n):
            return "█" * max(1, round(bw * n / top))

        lines += ["", f"{'Naive':<8}{orange(bar(naive))} {fmt_tok(naive)}", f"{'LHA':<8}{blue(bar(active))} {fmt_tok(active)}"]
        lines += ["", dim("prompt size over the run, same scale:"),
                  f"{'Naive':<8}{orange(self._spark(2))}", f"{'LHA':<8}{blue(self._spark(1))}"]
        return "\n".join(lines)

    def _spark(self, col: int) -> str:
        """History of prompt sizes as one line of blocks; both rows share the naive maximum."""
        h, width = self.history, W - 18
        if not h:
            return ""
        top = max(max(x[2] for x in h), 1)
        pick = [h[min(len(h) - 1, i * len(h) // width)] for i in range(min(width, len(h)))]
        return "".join("▁▂▃▄▅▆▇█"[min(7, round(7 * p[col] / top))] for p in pick)

    # -------------------------------------------------------------- the payoff
    @staticmethod
    def _building(rec: dict, event: str) -> bool:
        return rec["tool"] in ("cad_build", "finish") or event.startswith("INBOX EMPTY") or bool(rec["questions"])

    def _build(self, rec: dict) -> str:
        part = next(reversed(self.builds), None)
        if rec["tool"] == "cad_build":
            part = rec["args"].get("name", part)
        events = self.inbox[0] or rec["step"]
        lines = [rule("BUILD"),
                 f"{events:,} historical events",
                 "        │", "        ▼",
                 f"~{rec['state_tokens']:,} token working state   " + dim(f"(naive transcript: ~{fmt_tok(rec['naive_tokens'])})"),
                 "        │", "        ▼",
                 f"CAD agent ({rec.get('model', 'model')}) writes a build123d script",
                 "        │", "        ▼",
                 f"build123d / OpenCascade → {part + '.step' if part else 'part'}"]
        if self.builds:
            lines += ["        │", "        ▼", "agent's own check of each build (against its working state)"]
            for name, m in self.builds.items():
                spec = [rec["facts"].get(f"{name}.{a}") for a in ("length", "width", "thickness")]
                dims_ok = all(v is not None and abs(float(v) - b) < 0.01 for v, b in zip(spec, m["bbox_mm"]))
                holes = m.get("z_through_holes", m.get("cyl_faces"))
                dims = "×".join(f"{b:g}" for b in m["bbox_mm"])
                lines.append(f"   {name:<16} {_ok(dims_ok)} {dims} mm   {_ok(holes == 4)} {holes}/4 through-holes   "
                             f"{_ok(m.get('valid'))} solid · {m.get('mass_g', '?')} g")
        grade = (self.summary or {}).get("grade") if rec["done"] else None
        if grade:
            lines += ["        │", "        ▼", bold("Geometry validator (against the true spec)")]
            for g in grade.values():
                both = g["geometry"] and g["mass"]
                lines.append(f"   {_ok(both)} {g['part']:<16} {_ok(g['geometry'])} geometry & dimensions   "
                             f"{_ok(g['mass'])} mass & material")
            lines.append(bold(f"   score {self.summary.get('score', 0):.2f}"))
        return "\n".join(lines)


def _provenance(src: str) -> str:
    """How a fact's source is shown: DESIGN INTENT, ECO-1234, ECO-1234 · recalled, BUILD."""
    if not src:
        return ""
    base, recalled = src.replace(" via recall", ""), "via recall" in src
    base = {"design_intent": "DESIGN INTENT", "cad_build": "BUILD", "recall": "archive", "eco": "ECO", "user": "USER"}.get(base, base)
    return base + (" · recalled" if recalled else "")


def _spec_line(spec: dict) -> str:
    d = [spec.get(a) for a in ("length", "width", "thickness")]
    dims = " × ".join(_val(x) if x is not None else "?" for x in d) + " mm"
    mat = spec.get("material") or "?"
    hole = f"Ø{_val(spec['hole_diameter'])}" if spec.get("hole_diameter") is not None else "Ø?"
    return f"{dims} · {mat} · 4 × {hole} holes"


def _action(rec: dict) -> str:
    tool, args = rec["tool"], rec["args"] or {}
    if tool == "cad_build":
        return f'cad_build("{args.get("name")}", material="{args.get("material", "")}")  + build123d script'
    if tool == "recall":
        return f'recall("{args.get("query", "")}")'
    if tool == "finish":
        return "finish(answer)"
    return f"{tool}({', '.join(f'{k}={json.dumps(v)}' for k, v in args.items())[:50]})"


def _ok(v) -> str:
    return green("✓") if v else red("✗")


# ------------------------------------------------------------------ running it
def find_run(name: str | None, results: Path) -> Path:
    runs = results / "runs"
    if name and Path(name).is_dir():
        return Path(name)
    if name:
        d = runs / name
        if not d.is_dir():
            raise SystemExit(f"no run {name!r} in {runs}")
        return d
    traces = sorted(runs.glob("*/trace.jsonl"), key=lambda p: p.stat().st_mtime)
    if not traces:
        raise SystemExit(f"no traced runs in {runs}. Run: python -m lha.bench.run --env design --sizes 200")
    return traces[-1].parent


def _meta(run_dir: Path) -> tuple[int | None, dict | None]:
    m = re.search(r"-n(\d+)-", run_dir.name)
    s = run_dir / "summary.json"
    return (int(m[1]) if m else None), (json.loads(s.read_text()) if s.exists() else None)


def show(frame: str) -> None:
    if sys.stdout.isatty():
        sys.stdout.write("\x1b[H\x1b[2J" + frame + "\n")
    else:
        sys.stdout.write(frame + "\n" + "=" * W + "\n")
    sys.stdout.flush()


def replay(run_dir: Path, speed: float, follow: bool, start: int) -> None:
    total, summary = _meta(run_dir)
    intent = json.loads((run_dir / "intent.json").read_text()) if (run_dir / "intent.json").exists() else None
    ins = Inspector(run_dir.name, total, summary, intent)
    path = run_dir / "trace.jsonl"
    while not path.exists():
        if not follow:
            raise SystemExit(f"{path} does not exist (runs from before the inspector have no trace)")
        time.sleep(0.5)
    with path.open() as f:
        while True:
            line = f.readline()
            if not line:
                if not follow:
                    break
                time.sleep(0.3)
                continue
            rec = json.loads(line)
            if rec["done"]:  # the runner writes summary.json right after the last step
                for _ in range(20):
                    if (run_dir / "summary.json").exists():
                        ins.summary = json.loads((run_dir / "summary.json").read_text())
                        break
                    time.sleep(0.25)
            frame = ins.feed(rec)
            if rec["step"] < start:
                continue
            show(frame)
            if rec["done"]:
                break
            if speed > 0 and not follow:
                slow = rec["tool"] in ("cad_build", "recall") or rec["changes"] and "mass_g" in str(rec["changes"])
                time.sleep((4 if slow else 1) / speed)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m lha.inspect", description=__doc__.split("\n\n")[0])
    ap.add_argument("run", nargs="?", help="run id (e.g. design_stateful-n200-s0) or directory; default: latest")
    ap.add_argument("--results", default="results")
    ap.add_argument("--speed", type=float, default=30, help="steps per second when replaying (0 = no delay)")
    ap.add_argument("--follow", action="store_true", help="keep reading as a running benchmark appends steps")
    ap.add_argument("--start", type=int, default=0, help="skip to this step")
    a = ap.parse_args(argv)
    global W
    W = max(60, min(96, shutil.get_terminal_size((W, 40)).columns - 2))
    try:
        replay(find_run(a.run, Path(a.results)), a.speed, a.follow, a.start)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
