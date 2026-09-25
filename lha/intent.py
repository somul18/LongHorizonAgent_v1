"""Natural-Language Design Intent: human intent -> structured working state.

The entry point of the four-stage flow

    Natural-Language Intent -> Structured State -> Current Design Understanding -> CAD Geometry

A plain-English design request is interpreted into ordinary state operations: one
``set_fact`` per explicit requirement, all with ``source = "design_intent"``, and one
``add_question`` per requirement the request leaves open. Nothing is guessed. Conventions
the interpreter does apply (an M4 hole means a 4.3 mm clearance hole) are listed as notes,
so the reader can see them.

The request itself is kept as the *original* design intent. It is not a second source of
truth: later engineering changes overwrite facts in the state as usual, and the state wins.

Two interpreters share one output schema:
    interpret_rules(text)        deterministic, offline, used for demos and tests
    interpret_model(text, llm)   any LLM from lha.llm; its output is validated against the
                                 same schema, and anything outside it becomes a note
    interpret(text)              the model when one is configured, else the rules
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

INTENT_SOURCE = "design_intent"
# The part family the CAD pipeline builds: a rectangular plate with 4 corner through-holes.
ATTRS = ("length", "width", "thickness", "hole_diameter", "material")
HOLE_COUNT, HOLE_INSET = 4, 6.0
MATERIAL_WORDS = [  # (pattern, canonical id) - first match wins, so specific names come first
    (r"\b(?:al(?:uminium|uminum)?[\s-]*6061|6061[\s-]*(?:t6\s*)?al(?:uminium|uminum)?|al6061)\b", "al6061"),
    (r"\b(?:ti[\s-]*6al[\s-]*4v|ti6al4v|titanium)\b", "ti6al4v"),
    (r"\b(?:ss\s*304|304\s*stainless(?:\s*steel)?|stainless(?:\s*steel)?)\b", "ss304"),
    (r"\b(?:pa\s*12|nylon(?:\s*12)?)\b", "pa12"),
    (r"\babs\b", "abs"),
    (r"\bpla\b", "pla"),
    (r"\bsteel\b", "steel"),
    (r"\balumin(?:ium|um)\b", "al6061"),  # plain "aluminum": the family's only aluminum alloy (noted)
]
METRIC_CLEARANCE = {"m3": 3.2, "m4": 4.3, "m5": 5.3, "m6": 6.4}  # ISO 273 medium-fit clearance holes
PART_NAMES = [  # phrase -> DesignDesk part id, so a request can target an existing part
    (r"motor[\s-]*mount", "motor-mount"), (r"sensor[\s-]*(?:mount(?:ing)?[\s-]*)?(?:bracket|plate)", "sensor-bracket"),
    (r"base[\s-]*plate", "base-plate"), (r"battery[\s-]*tray", "battery-tray"), (r"\blid\b", "lid"),
    (r"gusset", "gusset"), (r"rail[\s-]*clamp", "rail-clamp"), (r"hinge[\s-]*plate", "hinge-plate"),
]
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "eight": 8}
NUM = r"(\d+(?:\.\d+)?)"


@dataclass
class Interpretation:
    request: str
    part: str
    facts: dict = field(default_factory=dict)      # attr -> value (numbers as strings, like the benchmark)
    unknown: list = field(default_factory=list)    # required attrs the request does not state
    notes: list = field(default_factory=list)      # conventions applied, or requests the family can't honour
    interpreter: str = "rules"

    def ops(self) -> list[dict]:
        """The state operations that load this intent into a working state."""
        ops = [{"op": "set_fact", "key": f"{self.part}.{a}", "value": v, "source": INTENT_SOURCE}
               for a, v in self.facts.items()]
        ops += [{"op": "add_question", "text": question(self.part, a)} for a in self.unknown]
        return ops

    def to_dict(self) -> dict:
        return asdict(self)


def question(part: str, attr: str) -> str:
    return f"{part}.{attr} is not specified in the design intent"


def _mm(v: float) -> str:
    return f"{v:g}"


def normalize_material(text: str) -> str | None:
    t = text.lower()
    for pat, mid in MATERIAL_WORDS:
        if re.search(pat, t):
            return mid
    return None


def part_name(text: str, default: str = "part") -> str:
    t = text.lower()
    for pat, pid in PART_NAMES:
        if re.search(pat, t):
            return pid
    m = re.search(r"\b([a-z]+)[\s-]+(?:mount(?:ing)?[\s-]+)?(plate|bracket|mount|clamp|tray)\b", t)
    if m and m[1] not in {"a", "an", "the", "mounting", "rectangular", "steel", "aluminum", "aluminium", "abs", "pla"}:
        return f"{m[1]}-{m[2]}"
    return default


# ------------------------------------------------------------------ rule-based interpreter
def interpret_rules(text: str, part: str | None = None) -> Interpretation:
    t = " " + text.lower().replace("×", "x").replace("*", "x") + " "
    it = Interpretation(request=text.strip(), part=part or part_name(text))
    f = it.facts

    # "100 x 80 x 4 mm" (length x width x thickness) or "80 x 60 mm" (length x width)
    if m := re.search(rf"{NUM}\s*(?:mm)?\s*x\s*{NUM}\s*(?:mm)?(?:\s*x\s*{NUM})?\s*mm", t):
        f["length"], f["width"] = _mm(float(m[1])), _mm(float(m[2]))
        if m[3]:
            f["thickness"] = _mm(float(m[3]))
    # "100 mm long", "80 mm wide", "4 mm thick" (and "length of 100 mm")
    for attr, words in (("length", r"long|length"), ("width", r"wide|width"), ("thickness", r"thick|thickness")):
        if m := re.search(rf"{NUM}\s*mm\s*(?:{words})\b", t) or re.search(rf"(?:{words})\s*(?:of|=|:)?\s*{NUM}\s*mm", t):
            f[attr] = _mm(float(m[1]))

    if (mid := normalize_material(t)) is not None:
        f["material"] = mid
        if re.search(r"\balumin(?:ium|um)\b", t) and "6061" not in t:
            it.notes.append("'aluminum' read as 6061 aluminum, the only aluminum alloy this part family supports")

    # holes: "four 5.3 mm through-holes", "4x M4 holes", "M4 mounting holes"
    hole_ctx = re.search(r"hole", t)
    if hole_ctx:
        count = None
        if m := re.search(r"\b(\d+|one|two|three|four|five|six|eight)\s*(?:x\s*)?(?:m\d\s*)?(?:[\d.]+\s*mm\s*)?"
                          r"(?:mounting\s*|through[\s-]*|clearance\s*|bolt\s*)*holes?\b", t):
            count = int(m[1]) if m[1].isdigit() else NUMBER_WORDS[m[1]]
        not_edge = r"(?!\s*(?:in\s*)?from)"  # "holes 8 mm from the edges" is an inset, not a diameter
        if m := re.search(rf"{NUM}\s*mm\s*(?:diameter\s*)?(?:mounting\s*|through[\s-]*|clearance\s*|corner\s*)*holes?", t) \
                or re.search(rf"holes?\s*(?:of\s*)?(?:diameter\s*)?(?:ø|dia\.?\s*)?{NUM}\s*mm{not_edge}", t) \
                or re.search(rf"(?:ø|diameter\s*(?:of\s*)?){NUM}\s*mm{not_edge}", t):
            f["hole_diameter"] = _mm(float(m[1]))
        elif m := re.search(r"\b(m[3-6])\b", t):
            f["hole_diameter"] = _mm(METRIC_CLEARANCE[m[1]])
            it.notes.append(f"{m[1].upper()} holes read as {METRIC_CLEARANCE[m[1]]:g} mm clearance holes (ISO 273, medium fit)")
        if count is not None and count != HOLE_COUNT:
            it.notes.append(f"the request asks for {count} holes; this part family always has {HOLE_COUNT} corner holes")
        if m := re.search(rf"{NUM}\s*mm\s*(?:in\s*)?from\s*(?:the\s*)?(?:adjacent\s*|each\s*|both\s*)?edges?", t):
            if float(m[1]) != HOLE_INSET:
                it.notes.append(f"hole centres requested {m[1]} mm from the edges; this part family uses {HOLE_INSET:g} mm")
    else:
        it.notes.append("no holes mentioned; this part family has four corner holes, so their diameter is needed")

    it.unknown = [a for a in ATTRS if a not in f]
    it.facts = {a: f[a] for a in ATTRS if a in f}
    return it


# ------------------------------------------------------------------ model-based interpreter
MODEL_SYSTEM = f"""You translate a mechanical design request into structured CAD requirements.
The part family is a rectangular plate (length x width x thickness, in mm) with {HOLE_COUNT} through-holes,
one near each corner, {HOLE_INSET:g} mm from the edges. Attributes: length, width, thickness, hole_diameter (mm),
material (one of: al6061, steel, ss304, ti6al4v, abs, pla, pa12).

Rules:
- Record ONLY values the request states explicitly. Never guess or apply defaults.
- A metric screw size means its clearance hole: M3=3.2, M4=4.3, M5=5.3, M6=6.4 mm. Add a note when you do this.
- Anything the family cannot honour (another hole count, another inset, a feature) goes in notes.
- part: a short kebab-case name for the part (e.g. motor-mount).

Reply with ONE JSON object and nothing else:
{{"part": "...", "facts": {{"length": 100, ...}}, "notes": ["..."]}}"""


def interpret_model(text: str, llm, part: str | None = None) -> Interpretation:
    from .agent import parse_response

    out, _ = llm.complete(MODEL_SYSTEM, f"Design request:\n{text.strip()}", 800)
    data = parse_response(out)
    it = Interpretation(request=text.strip(), part=part or _slug(data.get("part")) or part_name(text),
                        interpreter=getattr(llm, "name", "model"))
    raw = data.get("facts") or {}
    for a in ATTRS:
        v = raw.get(a)
        if v in (None, ""):
            continue
        if a == "material":
            mid = normalize_material(str(v)) or (str(v).lower() if str(v).lower() in {m for _, m in MATERIAL_WORDS} else None)
            if mid:
                it.facts[a] = mid
            else:
                it.notes.append(f"material {v!r} is not one this part family supports")
        else:
            try:
                it.facts[a] = _mm(float(str(v).lower().replace("mm", "").strip()))
            except ValueError:
                it.notes.append(f"{a} = {v!r} is not a number of millimetres")
    it.notes += [str(n) for n in data.get("notes") or []]
    it.notes += [f"ignored unknown attribute {k!r}" for k in raw if k not in ATTRS]
    it.unknown = [a for a in ATTRS if a not in it.facts]
    return it


def _slug(s) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(s or "").lower()).strip("-")[:40]


def interpret(text: str, part: str | None = None, llm=None) -> Interpretation:
    """Use the configured model when there is one (LHA_BEDROCK_MODEL_ID / LHA_LIQUID_BASE_URL), else the rules."""
    if llm is None:
        try:
            from .llm import from_env

            llm = from_env()
        except Exception:  # noqa: BLE001 - no model configured: the rules are the offline interpreter
            llm = None
    if llm is not None:
        try:
            return interpret_model(text, llm, part)
        except Exception as e:  # noqa: BLE001 - a failed model call must not lose the request
            it = interpret_rules(text, part)
            it.notes.append(f"model interpreter failed ({type(e).__name__}); used the rule-based interpreter")
            return it
    return interpret_rules(text, part)


# ------------------------------------------------------------------ changes after the intent
def interpret_change(text: str, part: str) -> tuple[dict, str]:
    """An engineering change or an answer, e.g. 'ECO-1847: reduce thickness to 3 mm' or 'thickness 5 mm'.
    Returns ({attr: value}, source) where source is the ECO id when one is given, else 'user'."""
    t = text.lower().replace("×", "x")
    src = (m[0].upper() if (m := re.search(r"eco[\s-]*\d+", t)) else "user").replace(" ", "-")
    out = {}
    for attr, words in (("length", r"length|long"), ("width", r"width|wide"), ("thickness", r"thickness|thick"),
                        ("hole_diameter", r"hole(?:[\s_-]*diameter)?s?|holes?")):
        # the new value: after "to" / an arrow if there is one ("from 6 mm to 4 mm", "95 mm -> 110 mm")
        if m := re.search(rf"(?:{words})(?:[^.;]|\.(?=\d))*?(?:\bto\b|→|->)\s*{NUM}\s*mm", t) or \
                re.search(rf"(?:{words})[^0-9]{{0,40}}?{NUM}\s*mm", t) or re.search(rf"{NUM}\s*mm\s*(?:{words})", t):
            out[attr] = _mm(float(m[1]))
    if "hole" in t and "hole_diameter" not in out and (m := re.search(r"\b(m[3-6])\b", t)):
        out["hole_diameter"] = _mm(METRIC_CLEARANCE[m[1]])
    if re.search(r"material|switch|change|make|use|\bto\b|→|->", t):
        # the target material is the last one named ("ABS -> Al6061", "switch from steel to PLA")
        found = sorted((m.start(), mid) for pat, mid in MATERIAL_WORDS for m in re.finditer(pat, t))
        if found:
            out["material"] = found[-1][1]
    return out, src


def intent_sentence(part: str, spec: dict) -> str:
    """Write a design request from a spec (used to seed DesignDesk's --intent part)."""
    from .describe import MATERIALS

    mat = MATERIALS.get(spec["material"], (spec["material"],))[0]
    return (f"Create a {part.replace('-', ' ')} {spec['length']} mm long, {spec['width']} mm wide and "
            f"{spec['thickness']} mm thick from {mat}. Add four {spec['hole_diameter']} mm through-holes, "
            f"one near each corner, with the hole centers {HOLE_INSET:g} mm from the adjacent edges.")


def dump(it: Interpretation) -> str:
    return json.dumps(it.to_dict(), indent=1)
