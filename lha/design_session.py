"""A design that starts from Natural-Language Design Intent and then evolves.

    Natural-Language Intent -> Structured State -> Current Design Understanding -> CAD Geometry

A session is a directory under ``results/runs/`` (so the dashboard's CAD parts tab lists it too):

    intent.json     the original request and how it was interpreted (never changes)
    state.json      the working state: facts with their source (design_intent, ECO-…, user)
    archive.jsonl   superseded values, exactly as in a long agent run
    log.jsonl       one line per operation: intent, change, build
    cad/            <part>.py / .brep / .step of the latest build
    summary.json    the latest validation, in the benchmark's format

The working state is the only source of truth. The request is kept as the *original* intent;
changes overwrite facts with ``set_fact`` (the old value goes to the archive) and answer open
questions, and the description and CAD are always derived from the current state.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict
from pathlib import Path

from .archive import Archive
from .describe import Change, describe_part
from .intent import ATTRS, HOLE_INSET, INTENT_SOURCE, Interpretation, interpret, interpret_change, question
from .state import WorkingState, apply_ops

PREFIX = "intent_"


def _safe(s: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", s.lower()).strip("-")[:40] or "part"


class DesignSession:
    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir)
        meta = json.loads((self.dir / "intent.json").read_text())
        self.request, self.part = meta["request"], meta["part"]
        self.interpretation = meta["interpretation"]
        self.state = WorkingState.from_dict(json.loads((self.dir / "state.json").read_text()))
        self.archive = Archive(self.dir / "archive.jsonl", self.dir.name)

    # ---------------------------------------------------------------- lifecycle
    @classmethod
    def create(cls, request: str, runs_dir: Path, part: str | None = None, llm=None,
               interpretation: Interpretation | None = None) -> "DesignSession":
        it = interpretation or interpret(request, part=part, llm=llm)
        run_dir = Path(runs_dir) / f"{PREFIX}{_safe(it.part)}-{time.strftime('%Y%m%d-%H%M%S')}"
        n = 1
        while run_dir.exists():
            n += 1
            run_dir = run_dir.with_name(f"{run_dir.name.rsplit('~', 1)[0]}~{n}")
        run_dir.mkdir(parents=True)
        (run_dir / "intent.json").write_text(json.dumps(
            {"part": it.part, "request": it.request, "interpretation": it.to_dict(), "created": time.time()}, indent=1))
        state = WorkingState(goal=f'Design intent for {it.part}: "{it.request}"')
        (run_dir / "state.json").write_text(json.dumps(state.to_dict(), indent=1))
        s = cls(run_dir)
        s._apply(it.ops(), "intent", INTENT_SOURCE, request)
        return s

    @classmethod
    def load(cls, runs_dir: Path, sid: str) -> "DesignSession":
        if not re.fullmatch(rf"{PREFIX}[\w.~-]+", sid):
            raise ValueError("bad design id")
        d = (Path(runs_dir) / sid).resolve()
        if d.parent != Path(runs_dir).resolve() or not (d / "intent.json").exists():
            raise FileNotFoundError(f"no design {sid!r}")
        return cls(d)

    @staticmethod
    def list(runs_dir: Path) -> list[dict]:
        out = []
        for d in sorted(Path(runs_dir).glob(f"{PREFIX}*/intent.json")):
            meta = json.loads(d.read_text())
            out.append({"id": d.parent.name, "part": meta["part"], "request": meta["request"],
                        "created": meta.get("created", d.stat().st_mtime)})
        return sorted(out, key=lambda x: -x["created"])

    # ---------------------------------------------------------------- state operations
    def _apply(self, ops: list[dict], kind: str, source: str, text: str) -> list[list]:
        s = self.state
        s.step += 1
        before = {k: f.value for k, f in s.facts.items()}
        errors = apply_ops(s, ops, lambda k, key, payload: self.archive.put(s.step, k, key, payload))
        changes = [[k, before.get(k), f.value] for k, f in s.facts.items() if before.get(k) != f.value]
        self._save()
        self._log({"step": s.step, "kind": kind, "source": source, "text": text, "ops": ops,
                   "changes": changes, "errors": errors, "ts": time.time()})
        return changes

    def change(self, text: str) -> dict:
        """An engineering change or an answer to an open question, e.g. 'ECO-1847: thickness 6 mm -> 3 mm'."""
        values, source = interpret_change(text, self.part)
        if not values:
            raise ValueError("no recognisable change: name an attribute and a value, e.g. 'ECO-1847: thickness 3 mm'")
        ops = [{"op": "set_fact", "key": f"{self.part}.{a}", "value": v, "source": source} for a, v in values.items()]
        ops += [{"op": "resolve_question", "text": question(self.part, a), "answer": v}
                for a, v in values.items() if question(self.part, a) in self.state.questions]
        changes = self._apply(ops, "change", source, text)
        return {"source": source, "values": values, "changes": changes}

    # ---------------------------------------------------------------- views
    def spec(self) -> dict:
        p = self.part + "."
        return {k[len(p):]: f.value for k, f in self.state.facts.items() if k.startswith(p)}

    def missing(self) -> list[str]:
        return [a for a in ATTRS if a not in self.spec()]

    def log(self) -> list[dict]:
        p = self.dir / "log.jsonl"
        return [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []

    def last_change(self) -> Change | None:
        for e in reversed(self.log()):
            if e["kind"] == "change":
                for key, old, new in e["changes"]:
                    if old is not None and key.startswith(self.part + "."):
                        return Change(key.split(".", 1)[1], old, new, e["source"] if e["source"] != "user" else "")
        return None

    def understanding(self) -> str:
        return describe_part(self.part, {**self.spec(), "type": "plate"}, last_change=self.last_change(),
                             hole_inset=HOLE_INSET, expected=ATTRS, missing="unspecified")

    def evolution(self) -> list[dict]:
        """Intent -> each change -> now, as a list of snapshots of the part's spec."""
        steps, spec = [], {}
        for e in self.log():
            if e["kind"] == "build":
                continue
            for key, _, new in e["changes"]:
                if key.startswith(self.part + "."):
                    spec[key.split(".", 1)[1]] = new
            label = "Design intent" if e["kind"] == "intent" else e["source"] if e["source"] != "user" else "User"
            steps.append({"label": label, "text": e["text"], "spec": dict(spec), "step": e["step"],
                          "changes": [c for c in e["changes"] if c[0].startswith(self.part + ".")]})
        return steps

    def view(self) -> dict:
        build = json.loads((self.dir / "build.json").read_text()) if (self.dir / "build.json").exists() else None
        if build:
            build["stale"] = build.get("spec") != self.spec()  # the state changed after this build
        facts = [{"key": k, "attr": k.split(".", 1)[-1], "value": f.value, "source": f.source, "step": f.step}
                 for k, f in sorted(self.state.facts.items())]
        return {"id": self.dir.name, "part": self.part, "request": self.request, "interpretation": self.interpretation,
                "facts": facts, "questions": list(self.state.questions), "missing": self.missing(),
                "understanding": self.understanding(), "evolution": self.evolution(), "build": build,
                "archived": [asdict(it) for it in self.archive.items][-20:]}

    # ---------------------------------------------------------------- CAD
    def build(self) -> dict:
        """Build the part from the current state with the parametric plate template, then validate it
        against an independently constructed reference. Refuses while a requirement is missing."""
        from .bench.design_desk import plate_script
        from .cad import CadWorkspace, DENSITIES, geometry_matches, measure

        spec = self.spec()
        if miss := self.missing():
            raise ValueError(f"can't build yet: {', '.join(miss)} not specified. Answer with e.g. '{miss[0]} 5 mm'.")
        dims = [float(spec[a]) for a in ("length", "width", "thickness", "hole_diameter")]
        ws = CadWorkspace(self.dir / "cad")
        part = ws.build(self.part, plate_script(*dims))
        density = DENSITIES.get(str(spec["material"]))
        m = measure(part, density)
        {t.name: t for t in ws.tools()}["cad_export"].fn({"name": self.part, "format": "step"})  # -> <part>.step
        reference = _reference(*dims)  # built differently (solid minus cylinders), so the check means something
        checks = {
            "dimensions": all(abs(b - x) < 0.01 for b, x in zip(m["bbox_mm"], dims[:3])),
            "through_holes": m["z_through_holes"] == 4,
            "valid_solid": bool(m["valid"]) and m["solids"] == 1,
            "geometry": geometry_matches(part, reference),
            "mass": density is not None and abs(m["mass_g"] - float(reference.volume) * density / 1000) < 0.01 * m["mass_g"],
        }
        result = {"spec": spec, "measure": m, "checks": checks, "ok": all(checks.values()),
                  "step_file": str(self.dir / "cad" / f"{self.part}.step"), "state_step": self.state.step}
        (self.dir / "build.json").write_text(json.dumps(result, indent=1))
        # the benchmark's summary format, so the CAD parts tab shows this design too
        (self.dir / "summary.json").write_text(json.dumps({
            "env": "intent", "agent": "design-intent", "run_id": self.dir.name, "model": self.interpretation.get("interpreter"),
            "score": 1.0 if result["ok"] else 0.0, "spec": {self.part: spec},
            "grade": {"q1": {"part": self.part, "geometry": checks["geometry"] and checks["dimensions"],
                             "mass": checks["mass"], "ref_mass_g": m.get("mass_g")}},
            "answer": {"q1": {"part": self.part, "mass_g": m.get("mass_g")}}}, indent=1))
        self._log({"step": self.state.step, "kind": "build", "source": "cad_build", "text": "Build CAD",
                   "ops": [], "changes": [], "errors": [], "ts": time.time(), "ok": result["ok"]})
        return result

    # ---------------------------------------------------------------- files
    def _save(self) -> None:
        tmp = self.dir / "state.tmp"
        tmp.write_text(json.dumps(self.state.to_dict(), indent=1, default=str))
        tmp.replace(self.dir / "state.json")

    def _log(self, rec: dict) -> None:
        with (self.dir / "log.jsonl").open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")


def _reference(length: float, width: float, thickness: float, hole_diameter: float):
    from .cad import _b3d

    b = _b3d()
    x, y = length / 2 - HOLE_INSET, width / 2 - HOLE_INSET
    ref = b.Box(length, width, thickness)
    for px in (-x, x):
        for py in (-y, y):
            ref = ref - b.Pos(px, py, 0) * b.Cylinder(hole_diameter / 2, thickness)
    return ref
