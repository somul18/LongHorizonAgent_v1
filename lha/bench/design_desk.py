"""DesignDesk: a long-horizon CAD task with drifting requirements.

The agent drains an inbox of N engineering messages (one per `next_message`
call). Most are chatter: design reviews, supplier quotes, proposals that were
rejected. Some are engineering change orders (ECOs) that change a dimension or
the material of one of several mounting plates, and each attribute changes
repeatedly over the run, so early values go stale. When the inbox is empty the
env asks for a few parts to be built with the *current* spec. The agent builds
each one with `cad_build` (build123d) and finishes with the mass of each part.

Every part is a rectangular plate, length x width x thickness (mm), centred on
the origin, with 4 through-holes of the given diameter whose centres sit
HOLE_INSET mm in from both edges at each corner. Grading is geometric: the
built solid must match a reference solid (same bounding box, empty symmetric
difference up to translation), and the reported mass must be within 1%.

Compared to IncidentDesk, the agent now has to turn its remembered facts into
a working artifact, and a single stale value anywhere (say, a thickness last
changed 900 messages ago) produces a wrong part.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..cad import DENSITIES, CadWorkspace, geometry_matches, measure
from ..tools import Tool

HOLE_INSET = 6.0
PARTS = ["base-plate", "motor-mount", "sensor-bracket", "battery-tray", "lid", "gusset", "rail-clamp", "hinge-plate"]
ATTRS = {
    "length": [float(v) for v in range(50, 125, 5)],
    "width": [float(v) for v in range(30, 85, 5)],
    "thickness": [2.0, 3.0, 4.0, 5.0, 6.0, 8.0],
    "hole_diameter": [3.2, 4.3, 5.3, 6.4],  # M3..M6 clearance
    "material": ["al6061", "steel", "ti6al4v", "abs", "pla"],
}
TEMPLATES = {
    "length": "[ECO-{n}] {part}: length changed to {val} mm (approved by {who})",
    "width": "[ECO-{n}] {part}: width changed to {val} mm (approved by {who})",
    "thickness": "[ECO-{n}] {part}: thickness changed to {val} mm (approved by {who})",
    "hole_diameter": "[ECO-{n}] {part}: hole_diameter changed to {val} mm (approved by {who})",
    "material": "[ECO-{n}] {part}: material changed to {val} (approved by {who})",
}
UPDATE_RE = re.compile(r"\[ECO-\d+\] (?P<part>[\w-]+): (?P<attr>\w+) changed to (?P<val>[\w.]+)")
PEOPLE = ["priya", "marco", "yuki", "sam", "lena", "omar"]
NOISE = [
    "[review] {part}: {who} asked whether the fillets are worth it; parked until next sprint",
    "[proposal] {part}: thickness {a} mm suggested by {who} -- REJECTED at design review, no change",
    "[proposal] {part}: switch to {mat}? {who} to get quotes; not approved, keep current material",
    "[supplier] quote for {c}00x {part}: ${a}.{b}{c} each, lead time {c} weeks",
    "[cam] toolpath for {part} regenerated in {a}s; {b} warnings (all cosmetic)",
    "[test] {part} vibration test run {a}: peak {b}.{c} g, within envelope",
]


def fmt_val(attr: str, val) -> str:
    return val if attr == "material" else f"{val:g}"


def parse_update(text: str) -> tuple[str, str, str] | None:
    m = UPDATE_RE.search(text)
    return (m["part"], m["attr"], m["val"]) if m else None


def plate_script(length: float, width: float, thickness: float, hole_diameter: float) -> str:
    """The canonical build123d script for a DesignDesk plate (also used by the policies)."""
    return (f"L, W, T, D, INSET = {length:g}, {width:g}, {thickness:g}, {hole_diameter:g}, {HOLE_INSET:g}\n"
            "with BuildPart() as p:\n"
            "    Box(L, W, T)\n"
            "    with Locations(*[(x, y) for x in (-L/2 + INSET, L/2 - INSET) for y in (-W/2 + INSET, W/2 - INSET)]):\n"
            "        Hole(D / 2)\n"
            "result = p.part\n")


@dataclass
class DesignDesk:
    n_messages: int = 200
    update_rate: float = 0.3
    n_questions: int = 3
    seed: int = 0
    out_dir: str | Path | None = None
    messages: list[str] = field(default_factory=list)
    truth: dict[str, str] = field(default_factory=dict)
    questions: dict[str, str] = field(default_factory=dict)  # qid -> part
    cursor: int = 0

    def __post_init__(self) -> None:
        rng = random.Random(self.seed)
        self.workspace = CadWorkspace(self.out_dir)
        # Every (part, attr) gets an initial value up front, so early facts exist.
        for part in PARTS:
            for attr, vals in ATTRS.items():
                self._update(rng, part, attr, rng.choice(vals))
        while len(self.messages) < self.n_messages:
            if rng.random() < self.update_rate:
                part, attr = rng.choice(PARTS), rng.choice(list(ATTRS))
                self._update(rng, part, attr, rng.choice(ATTRS[attr]))
            else:
                self.messages.append(rng.choice(NOISE).format(
                    part=rng.choice(PARTS), who=rng.choice(PEOPLE), mat=rng.choice(ATTRS["material"]),
                    a=rng.randint(2, 99), b=rng.randint(1, 9), c=rng.randint(1, 9)))
        # Ask for the most-changed parts AND the part whose spec went quiet earliest.
        last_touch, churn = {}, {p: 0 for p in PARTS}
        for i, m in enumerate(self.messages):
            if u := parse_update(m):
                last_touch[u[0]] = i
                churn[u[0]] += 1
        stale_first = sorted(PARTS, key=lambda p: last_touch[p])
        by_churn = sorted(PARTS, key=lambda p: -churn[p])
        picks = list(dict.fromkeys([stale_first[0], *by_churn]))[: self.n_questions]
        rng.shuffle(picks)
        self.questions = {f"q{i + 1}": p for i, p in enumerate(picks)}

    def _update(self, rng: random.Random, part: str, attr: str, val) -> None:
        self.truth[f"{part}.{attr}"] = fmt_val(attr, val)
        self.messages.append(TEMPLATES[attr].format(part=part, val=fmt_val(attr, val),
                                                    n=rng.randint(1000, 9999), who=rng.choice(PEOPLE)))

    # ------------------------------------------------------------------ reference
    def spec(self, part: str) -> dict:
        return {a: self.truth[f"{part}.{a}"] for a in ATTRS}

    def reference(self, part: str):
        s = self.spec(part)
        script = plate_script(*(float(s[a]) for a in ("length", "width", "thickness", "hole_diameter")))
        return CadWorkspace().build(part, script)

    # ------------------------------------------------------------------ agent API
    def goal(self) -> str:
        return ("You are the CAD engineer for a set of mounting plates. Drain the engineering inbox by calling "
                "next_message until it is empty. Track the CURRENT length, width, thickness, hole_diameter (mm) and "
                "material of every part; ECO messages change them and only the latest ECO counts (proposals and "
                "reviews change nothing). Every part is a rectangular plate length x width x thickness centred on "
                f"the origin, with 4 through-holes of hole_diameter whose centres are {HOLE_INSET:g} mm in from both "
                "edges at each corner. Keep one fact per value, keyed <part>.<attr> (e.g. base-plate.thickness); "
                "evicted facts come back with recall on that key. When the inbox is empty you get build requests: "
                "pin the requested parts' facts, then build each requested part "
                "with cad_build (name = the part, pass its material). Check the result against the spec: bbox_mm "
                "must equal length, width, thickness and z_through_holes must be 4; if not, fix the script and "
                "rebuild. When a part is right, set_fact <part>.mass_g to its mass so you do not build it again. "
                'When every requested part has a mass fact, finish {"answer": {"q1": {"part": <name>, '
                '"mass_g": <mass from cad_build>}, ...}}.')

    def question_text(self) -> str:
        qs = "; ".join(f"{qid}: build {part}" for qid, part in self.questions.items())
        return f"INBOX EMPTY. Build requests -> {qs}"

    def tools(self) -> list[Tool]:
        def next_message(_args: dict) -> str:
            if self.cursor >= len(self.messages):
                return self.question_text()
            self.cursor += 1
            return f"msg {self.cursor}/{len(self.messages)}: {self.messages[self.cursor - 1]}"

        return [Tool("next_message", "read the next inbox message. args: {}", next_message),
                *self.workspace.tools()]

    def grade(self, answer) -> dict[str, dict]:
        """Per requested part: geometry (built solid vs reference) and reported mass."""
        answer = answer if isinstance(answer, dict) else {}
        out = {}
        for qid, part in self.questions.items():
            ref = self.reference(part)
            ref_mass = measure(ref, DENSITIES[self.truth[f"{part}.material"]])["mass_g"]
            a = answer.get(qid) if isinstance(answer.get(qid), dict) else {}
            try:
                mass_ok = abs(float(a.get("mass_g")) - ref_mass) <= 0.01 * ref_mass
            except (TypeError, ValueError):
                mass_ok = False
            geom_ok = geometry_matches(self.workspace.parts.get(part), ref)
            out[qid] = {"part": part, "geometry": geom_ok, "mass": mass_ok, "ref_mass_g": ref_mass}
        return out

    def score(self, answer) -> float:
        g = self.grade(answer)
        return sum((r["geometry"] + r["mass"]) / 2 for r in g.values()) / len(g)


# ====================================================================== policies
# Deterministic "agents" that read the exact prompt a real model would get.
# They make the benchmark reproducible offline; with --live a real model is used.

REQUEST_RE = re.compile(r"(q\d+): build ([\w-]+)")
BUILT_RE = re.compile(r"built '([\w-]+)': (\{.*\})")


def _section(prompt: str, name: str) -> str:
    m = re.search(rf"# {name}[^\n]*\n(.*?)(?=\n# |\Z)", prompt, re.S)
    return m.group(1) if m else ""


def _next_action(requests: list[tuple[str, str]], spec, built: dict[str, float]) -> dict:
    """Shared end-game: build the first unbuilt part, else finish. `spec(part)` -> dict or missing key."""
    for _, part in requests:
        if part in built:
            continue
        s = spec(part)
        if isinstance(s, str):
            return {"tool": "recall", "args": {"query": s, "k": 5, "kinds": ["evicted_fact", "dropped_fact"]}}
        script = plate_script(*(float(s[a]) for a in ("length", "width", "thickness", "hole_diameter")))
        return {"tool": "cad_build", "args": {"name": part, "script": script, "material": s["material"]}}
    return {"tool": "finish", "args": {"answer": {qid: {"part": p, "mass_g": built[p]} for qid, p in requests}}}


def stateful_policy(system: str, prompt: str) -> str:
    facts = dict(re.findall(r"^- \*?([\w.-]+) = \"([^\"]*)\"", _section(prompt, "FACTS"), re.M))
    obs = _section(prompt, "LAST OBSERVATION")
    requests = REQUEST_RE.findall(_section(prompt, "OPEN QUESTIONS"))
    ops: list[dict] = []

    def set_fact(key: str, val: str, source: str, pin: bool = False) -> None:
        ops.append({"op": "set_fact", "key": key, "value": val, "source": source, **({"pin": True} if pin else {})})
        facts[key] = val

    if (u := parse_update(obs)) and "INBOX EMPTY" not in obs:
        set_fact(f"{u[0]}.{u[1]}", u[2], "eco")
    if "INBOX EMPTY" in obs:
        for qid, part in REQUEST_RE.findall(obs):
            ops.append({"op": "add_question", "text": f"{qid}: build {part}"})
            requests.append((qid, part))
            ops += [{"op": "pin", "key": f"{part}.{a}"} for a in ATTRS if f"{part}.{a}" in facts]
    if obs.startswith("recall("):
        # Newest archived value wins, but never overwrite a live fact (it is newer than any eviction).
        best: dict[str, tuple[int, str]] = {}
        for step, key, val in re.findall(r"\[step (\d+)\] (?:evicted|dropped)_fact ([\w.-]+) = \"([^\"]*)\"", obs):
            if key not in facts and int(step) >= best.get(key, (-1, ""))[0]:
                best[key] = (int(step), val)
        for key, (_, val) in best.items():
            set_fact(key, val, "recall", pin=True)
    if (b := BUILT_RE.search(obs)) and "mass_g" in b[2]:
        set_fact(f"{b[1]}.mass_g", str(json.loads(b[2])["mass_g"]), "cad_build", pin=True)

    if not requests:
        action = {"tool": "next_message", "args": {}}
    else:
        def spec(part):
            missing = [f"{part}.{a}" for a in ATTRS if f"{part}.{a}" not in facts]
            return missing[0] if missing else {a: facts[f"{part}.{a}"] for a in ATTRS}
        built = {p: float(facts[f"{p}.mass_g"]) for _, p in requests if f"{p}.mass_g" in facts}
        action = _next_action(requests, spec, built)
    ops += _plan_ops(_section(prompt, "PLAN"), action["tool"], bool(requests))
    return json.dumps({"thought": action["tool"], "state_ops": ops, "action": action})


PLAN = [("track", "Track engineering changes"), ("cad", "Produce final CAD"), ("validate", "Validate geometry")]


def _plan_ops(plan: str, tool: str, building: bool) -> list[dict]:
    """A short plan, kept current: what a real model is expected to do with its PLAN section.
    Finished tasks may have been folded into a digest by the compactor; those are left alone."""
    status = dict((t, st) for st, t in re.findall(r"^- \[(\w+)\] ([\w-]+):", plan, re.M))
    if not plan.strip():
        return [{"op": "add_task", "id": t, "title": title} for t, title in PLAN] + [
            {"op": "update_task", "id": "track", "status": "doing"}]
    want = {"track": "doing", "cad": "todo", "validate": "todo"}
    if building:
        want = {"track": "done", "cad": "doing", "validate": "todo"}
    if tool == "finish":
        want = {"track": "done", "cad": "done", "validate": "done"}
    return [{"op": "update_task", "id": t, "status": st} for t, st in want.items() if t in status and status[t] != st]


def naive_policy(system: str, prompt: str) -> str:
    """Perfect reader of the transcript: last ECO per key, then builds what was requested."""
    requests = REQUEST_RE.findall(prompt) if "INBOX EMPTY" in prompt else []
    if not requests:
        return json.dumps({"thought": "keep draining", "action": {"tool": "next_message", "args": {}}})
    requests = list(dict.fromkeys(requests))
    latest: dict[str, str] = {}
    built: dict[str, float] = {}
    for line in prompt.splitlines():
        if " next_message -> " in line and (u := parse_update(line)):
            latest[f"{u[0]}.{u[1]}"] = u[2]
        elif " cad_build -> " in line and (b := BUILT_RE.search(line)):
            built[b[1]] = json.loads(b[2]).get("mass_g")
    action = _next_action(requests, lambda p: {a: latest[f"{p}.{a}"] for a in ATTRS}, built)
    return json.dumps({"thought": action["tool"], "action": action})
