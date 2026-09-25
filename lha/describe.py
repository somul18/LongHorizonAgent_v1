"""Current Design Understanding: a plain-English view of a part, derived from the working state.

The structured working state is the source of truth. This module turns the facts about one
part (``<part>.<attr> = value``) into a readable description of its geometry, dimensions,
material, features and latest engineering change. It is a pure function of that state:
nothing here is stored or fed back to the agent, and it is regenerated whenever the state
changes, so the text can never drift from the facts.

    describe_part("motor-mount", {"length": "100", "width": "80", "thickness": "4",
                                  "hole_diameter": "5.3", "material": "al6061"},
                  last_change=Change("thickness", "6", "4", "ECO-1847"), hole_inset=6)
    -> "The motor mount is a 100 × 80 × 4 mm rectangular mounting plate made from 6061 aluminum,
        with four 5.3 mm through-holes, one near each corner, 6 mm in from each edge. The latest
        approved engineering change (ECO-1847) reduced the thickness from 6 mm to 4 mm."

Three views of one design: structured state (truth) → this description (understanding) →
CAD geometry (realization).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# name, class. The class only drives a short manufacturing note.
MATERIALS = {
    "al6061": ("6061 aluminum", "metal"), "steel": ("steel", "metal"), "ss304": ("304 stainless steel", "metal"),
    "ti6al4v": ("Ti-6Al-4V titanium", "metal"), "abs": ("ABS", "polymer"), "pla": ("PLA", "polymer"),
    "pa12": ("PA12 nylon", "polymer"), "pet": ("PET", "polymer"), "hdpe": ("HDPE", "polymer"),
    "pp": ("polypropylene", "polymer"), "brass": ("brass", "metal"), "silver": ("sterling silver", "metal"),
    "gold": ("gold", "metal"),
}
UNITS = {"length": "mm", "width": "mm", "height": "mm", "depth": "mm", "thickness": "mm", "wall": "mm",
         "hole_diameter": "mm", "mass_g": "g",
         "diameter": "mm", "neck_diameter": "mm", "neck_height": "mm", "bar_width": "mm", "crossbar_from_top": "mm",
         "size": "mm", "line_width": "mm"}
NUMBERS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
# Attributes a plate-like part needs before it can be built (DesignDesk's spec).
PLATE_ATTRS = ("length", "width", "thickness", "hole_diameter", "material")


@dataclass
class Change:
    attr: str
    old: object
    new: object
    source: str = ""  # e.g. "ECO-1847"


def _num(v) -> float | None:
    try:
        return float(str(v).split()[0])
    except (TypeError, ValueError, IndexError):
        return None


def _fmt(v) -> str:
    n = _num(v)
    return f"{n:g}" if n is not None else str(v)


def _qty(attr: str, v) -> str:
    u = UNITS.get(attr)
    return f"{_fmt(v)} {u}" if u else str(v)


def _count(n) -> str:
    n = int(n)
    return NUMBERS[n] if 0 <= n < len(NUMBERS) else str(n)


HUMAN = {"crossbar_from_top": "crossbar position (from the top)", "wall": "wall thickness", "size": "size (point to point)",
         "neck_diameter": "neck diameter", "diameter": "body diameter"}


def _human(attr: str) -> str:
    return HUMAN.get(attr) or attr.replace("_g", "").replace("_", " ")


def _material(v) -> tuple[str, str | None]:
    return MATERIALS.get(str(v).strip().lower(), (str(v), None))


def describe_part(name: str, attrs: dict, *, last_change: Change | None = None, hole_inset: float | None = None,
                  expected: tuple[str, ...] = (), missing: str = "archived") -> str:
    """missing: why an expected value is absent: "archived" (evicted from the working state; the
    agent can recall it) or "unspecified" (the design intent never stated it)."""
    """One paragraph about one part. ``attrs`` maps attribute -> value, straight from the state."""
    a = {k: v for k, v in attrs.items() if v not in (None, "")}
    kind = str(a.pop("type", "")) or _kind_from_name(name) or ("plate" if "thickness" in a else "part")
    shape = {"plate": "rectangular mounting plate", "bracket": "mounting bracket"}.get(kind, kind)

    dims = [a.get(k) for k in ("length", "width", "thickness")] if "length" in a or "thickness" in a else \
           [a.get(k) for k in ("width", "height", "depth")]
    dims_txt = " × ".join(_fmt(d) if d is not None else "?" for d in dims) + " mm" if any(d is not None for d in dims) else ""
    mat_name, mat_class = _material(a["material"]) if "material" in a else (None, None)

    body = " ".join(x for x in [dims_txt, shape] if x)
    first = (f"The {name.replace('-', ' ')} is {_article(body)} {body}" if name
             else f"{_article(body).capitalize()} {body}")
    if mat_name:
        first += f" made from {mat_name}"
    features = []
    if "wall" in a:
        features.append(f"{_qty('wall', a['wall'])} walls")
    holes = a.get("holes", a.get("mounting_holes"))
    if holes is None and "hole_diameter" in a and hole_inset is not None:
        holes = 4  # the part family always has one hole per corner
    if holes is not None:
        size = f"{_qty('hole_diameter', a['hole_diameter'])} " if "hole_diameter" in a else ""
        where = f", one near each corner, {_fmt(hole_inset)} mm in from each edge" if hole_inset is not None and int(holes) == 4 \
            else ""
        features.append(f"{_count(holes)} {size}{'through-' if hole_inset is not None else 'mounting '}holes{where}")
    sentences = [first + (", with " + " and ".join(features) if features else "") + "."]

    if mat_class == "polymer":
        sentences.append(f"As {_article(mat_name)} {mat_name} part it suits additive manufacturing.")
    if "mass_g" in a:
        sentences.append(f"As built, it weighs {_fmt(a['mass_g'])} g.")
    if last_change is not None:
        sentences.append(_change_sentence(last_change))
    known = set(attrs) | {"type"}
    absent = [_human(k) for k in expected if k not in known or attrs.get(k) in (None, "")]
    if absent and missing == "unspecified":
        sentences.append(f"Missing requirement{'s' if len(absent) > 1 else ''}: {', '.join(absent)} "
                         f"(not specified in the design intent; nothing is assumed).")
    elif absent:
        sentences.append(f"Not in the working state right now: {', '.join(absent)} "
                         f"(archived; the agent has to recall {'it' if len(absent) == 1 else 'them'} before building).")
    extra = {k: v for k, v in a.items() if k not in {*PLATE_ATTRS, "height", "depth", "wall", "holes",
                                                      "mounting_holes", "mass_g"}}
    if extra:
        sentences.append("Other recorded properties: " + ", ".join(f"{_human(k)} {_qty(k, v)}" for k, v in extra.items()) + ".")
    return " ".join(sentences)


def _kind_from_name(name: str) -> str:
    n = name.lower()
    if "bracket" in n:
        return "bracket"
    return "plate" if name else ""


def _article(phrase: str) -> str:
    """'a' or 'an' by sound: an 80 mm, an 11 mm, an ABS part; a 100 mm, a PLA part."""
    w = phrase.split(" ")[0] if phrase else ""
    if re.match(r"(8|11|18)(\D|$)|8\d", w) and not re.match(r"(1[0-79]|1\d\d)", w):
        return "an"
    if re.match(r"(11|18)(\d\d)?(\D|$)", w):  # eleven, eighteen, eleven/eighteen hundred
        return "an"
    return "an" if w[:1].upper() in "AEIO" else "a"


def _change_sentence(c: Change) -> str:
    src = f" ({c.source})" if c.source else ""
    if c.attr == "material":
        return (f"The latest approved engineering change{src} switched the material from "
                f"{_material(c.old)[0]} to {_material(c.new)[0]}.")
    if c.old is None:  # the previous value was archived when the change arrived
        return f"The latest approved engineering change{src} set the {_human(c.attr)} to {_qty(c.attr, c.new)}."
    old, new = _num(c.old), _num(c.new)
    verb = "changed" if old is None or new is None or old == new else "reduced" if new < old else "increased"
    return (f"The latest approved engineering change{src} {verb} the {_human(c.attr)} "
            f"from {_qty(c.attr, c.old)} to {_qty(c.attr, c.new)}.")


# ------------------------------------------------------------------ from a trace / working state
def part_attrs(facts: dict, part: str) -> dict:
    """{attr: value} for one part from flat ``<part>.<attr>`` facts."""
    p = part + "."
    return {k[len(p):]: v for k, v in facts.items() if k.startswith(p)}


def focus_part(rec: dict, previous: str | None) -> str | None:
    """The part the agent is working on at this step: the first fact it set, else the first one that changed."""
    first = next((op.get("key") for op in rec.get("ops") or [] if op.get("op") == "set_fact" and op.get("key")), None)
    changed = sorted(k for k, *_ in rec.get("changes", []))
    key = first or (changed[0] if changed else None)
    return key.split(".")[0] if key and "." in key else previous


ECO = re.compile(r"\[(ECO-\d+)\]")


class DesignTracker:
    """Follows a trace and keeps, per part, the latest overwrite and where it came from,
    so a description can say what the most recent engineering change did."""

    def __init__(self, hole_inset: float | None = None, expected: tuple[str, ...] = (), kind: str | None = None):
        self.hole_inset, self.expected, self.kind = hole_inset, expected, kind
        self.last: dict[str, Change] = {}
        self.focus: str | None = None
        self._event = ""

    def feed(self, rec: dict) -> dict | None:
        """Update from one trace record; return {"part", "text"} for the part in focus (or None)."""
        eco = ECO.search(self._event)
        srcs = rec.get("sources", {})
        for key, old, new in rec.get("changes", []):
            src = srcs.get(key, "")
            if old == new or "." not in key or "recall" in src or src in ("design_intent", "cad_build"):
                continue  # a recall restores a value; the intent and builds are not engineering changes
            part, attr = key.split(".", 1)
            if attr != "mass_g" and (old is not None or src.startswith("ECO")):
                self.last[part] = Change(attr, old, new, src if src.startswith("ECO") else (eco[1] if eco else ""))
        self.focus = focus_part(rec, self.focus)
        self._event = rec.get("observation", "") or ""
        if not self.focus:
            return None
        return {"part": self.focus, "text": self.describe(rec.get("facts", {}), self.focus)}

    def describe(self, facts: dict, part: str) -> str:
        attrs = part_attrs(facts, part)
        if self.kind and "type" not in attrs:
            attrs["type"] = self.kind
        return describe_part(part, attrs, last_change=self.last.get(part),
                             hole_inset=self.hole_inset, expected=self.expected)


def tracker_for(run_id: str) -> DesignTracker:
    """DesignDesk runs know their part family (4 corner holes, 6 mm inset, 5 spec attributes)."""
    if "design_" in run_id:
        from .bench.design_desk import HOLE_INSET
        return DesignTracker(HOLE_INSET, PLATE_ATTRS, kind="plate")  # every DesignDesk part is a plate
    return DesignTracker()
