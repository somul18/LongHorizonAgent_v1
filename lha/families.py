"""Part families the design-intent flow can build.

Each family says what a request must pin down (its attributes), how to read them from plain
English, how to build the part (a build123d script, the same kind the agent writes), how to
build an independent reference to validate against, what to check, and how to describe it.

    plate   rectangular mounting plate with four corner through-holes (the DesignDesk part)
    bottle  hollow body of revolution: body, 45° shoulder, open neck
    cross   Latin cross (crossbar placed from the top) or Greek cross (crossbar centred)
    star    Star of David: two interlaced equilateral triangles, solid or as an outline

The rules are the same for every family: a value the request states becomes a fact; a value
it doesn't state stays unknown and the part is not built until it is answered.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

NUM = r"(\d+(?:\.\d+)?)"


def _f(v) -> float:
    return float(str(v).split()[0])


def _g(v: float) -> str:
    return f"{v:g}"


def _find(t: str, *patterns: str) -> str | None:
    for p in patterns:
        if m := re.search(p, t):
            return _g(float(m[1]))
    return None


@dataclass
class Family:
    name: str
    label: str
    attrs: list[tuple[str, str, str]]           # (attr, unit, label)
    words: str                                  # regex that selects this family
    required: list[str] = field(default_factory=list)

    def detect(self, t: str) -> bool:
        return bool(re.search(self.words, t))

    def attr_names(self) -> list[str]:
        return [a for a, _, _ in self.attrs]

    def unit(self, attr: str) -> str:
        return next((u for a, u, _ in self.attrs if a == attr), "")

    # overridden per family -------------------------------------------------
    def parse(self, t: str) -> tuple[dict, list[str]]:
        raise NotImplementedError

    def needed(self, facts: dict) -> list[str]:
        return [a for a in self.required if a not in facts]

    def script(self, s: dict) -> str:
        raise NotImplementedError

    def reference(self, s: dict):
        raise NotImplementedError

    def sanity(self, s: dict) -> str | None:
        return None

    def expected_bbox(self, s: dict) -> list[float]:
        raise NotImplementedError

    def extra_checks(self, part, s: dict, m: dict) -> dict:
        return {}

    def describe(self, name: str, s: dict) -> str:
        raise NotImplementedError

    def part_name(self, t: str) -> str:
        return self.name

    def understanding(self, name: str, s: dict, last_change=None) -> str:
        """Current Design Understanding: derived from the spec, the latest change, and what is still open."""
        from .describe import _change_sentence, _human

        out = [self.describe(name, s)]
        if last_change is not None:
            out.append(_change_sentence(last_change))
        if absent := [_human(a) for a in self.needed(s)]:
            out.append(f"Missing requirement{'s' if len(absent) > 1 else ''}: {', '.join(absent)} "
                       f"(not specified in the design intent; nothing is assumed).")
        return " ".join(out)

    def model_system(self) -> str:
        """The prompt for a model interpreter: same rules as the plate family, this family's attributes."""
        from .intent import MATERIAL_WORDS

        mats = ", ".join(dict.fromkeys(m for _, m in MATERIAL_WORDS))
        attrs = "; ".join(f"{a} ({u})" if u else a for a, u, _ in self.attrs if a != "material")
        extra = ' style is "solid" or "outline".' if "style" in self.attr_names() else ""
        return (f"You translate a design request for a {self.label} into structured CAD requirements.\n"
                f"Attributes: {attrs}; material (one of: {mats}).{extra}\n\n"
                "Rules:\n- Record ONLY values the request states explicitly. Never guess or apply defaults.\n"
                "- Anything this part family cannot honour goes in notes.\n"
                "- part: a short kebab-case name for the part.\n\n"
                'Reply with ONE JSON object and nothing else:\n{"part": "...", "facts": {...}, "notes": ["..."]}')

    def spec_line(self, s: dict) -> str:
        from .describe import _material

        def one(a, u, label):
            if a not in s:
                return f"{label} ?"
            return _material(s[a])[0] if a == "material" else f"{label} {s[a]}{' ' + u if u else ''}"

        return " · ".join(one(a, u, label) for a, u, label in self.attrs
                          if not (a == "line_width" and s.get("style") == "solid"))

    # change parsing: "height 260 mm", "wall from 2 mm to 1.5 mm", "ECO-12: neck diameter -> 32 mm"
    synonyms: dict = field(default_factory=dict)

    def parse_change(self, t: str) -> dict:
        """Each clause changes one attribute: the one named most specifically (the longest keyword,
        then the earliest), set to the value after "to"/an arrow if there is one, else its value."""
        out = {}
        for clause in re.split(r"[,;]|\band\b", t):
            m = re.search(rf"(?:\bto\b|→|->)\s*{NUM}\s*mm", clause) or re.search(rf"{NUM}\s*mm", clause)
            if not m:
                continue
            hits = [(-len(k[0]), k.start(), a) for a, unit, label in self.attrs if unit
                    for k in [re.search(self.synonyms.get(a, re.escape(label)), clause)] if k]
            if hits:
                out[min(hits)[2]] = _g(float(m[1]))
        if "style" in self.attr_names():
            if re.search(r"\bsolid\b|filled", t):
                out["style"] = "solid"
            elif re.search(r"outline|\blines?\b", t):
                out["style"] = "outline"
        return out


# ---------------------------------------------------------------------- plate
class Plate(Family):
    def __init__(self):
        super().__init__("plate", "rectangular mounting plate",
                         [("length", "mm", "length"), ("width", "mm", "width"), ("thickness", "mm", "thickness"),
                          ("hole_diameter", "mm", "hole Ø"), ("material", "", "material")],
                         r"plate|mount|bracket|clamp|tray|lid|gusset",
                         ["length", "width", "thickness", "hole_diameter", "material"])

    # parsing, building and describing plates predate this registry: see intent.py, design_desk.plate_script
    # and describe.describe_part. The plate family delegates to them.


# ---------------------------------------------------------------------- bottle
class Bottle(Family):
    def __init__(self):
        super().__init__("bottle", "water bottle",
                         [("height", "mm", "height"), ("diameter", "mm", "body Ø"), ("wall", "mm", "wall"),
                          ("neck_diameter", "mm", "neck Ø"), ("neck_height", "mm", "neck height"), ("material", "", "material")],
                         r"bottle|flask|canteen", ["height", "diameter", "wall", "neck_diameter", "neck_height", "material"],
                         synonyms={"height": r"(?<!neck )height|tall", "diameter": r"(?<!neck )diameter|body", "wall": r"wall",
                                   "neck_diameter": r"neck(?:\s*diameter)?|opening|mouth", "neck_height": r"neck\s*height|neck\s*length"})

    def part_name(self, t):
        return "water-bottle" if "water" in t else "bottle"

    def parse(self, t):
        notes = []
        s = {"height": _find(t, rf"{NUM}\s*mm\s*(?:tall|high)\b", rf"height\s*(?:of\s*)?{NUM}\s*mm"),
             "diameter": _find(t, rf"{NUM}\s*mm\s*(?:body\s*)?diameter", rf"(?:body\s*)?diameter\s*(?:of\s*)?{NUM}\s*mm",
                               rf"{NUM}\s*mm\s*wide"),
             "wall": _find(t, rf"{NUM}\s*mm\s*(?:thick\s*)?walls?", rf"wall(?:\s*thickness)?\s*(?:of\s*)?{NUM}\s*mm"),
             "neck_diameter": _find(t, rf"{NUM}\s*mm\s*(?:wide\s*)?(?:neck|opening|mouth)",
                                    rf"(?:neck|opening|mouth)\s*(?:diameter\s*)?(?:of\s*)?{NUM}\s*mm(?!\s*(?:tall|high|long))"),
             "neck_height": _find(t, rf"neck[^.;]{{0,25}}?{NUM}\s*mm\s*(?:tall|high|long)", rf"neck\s*height\s*(?:of\s*)?{NUM}\s*mm",
                                  rf"{NUM}\s*mm\s*(?:tall|high|long)\s*neck")}
        if re.search(r"\d\s*(?:ml|l|litre|liter)\b", t):
            notes.append("capacity is not used to size the bottle (that would mean guessing its proportions); "
                         "the capacity of the built bottle is reported instead")
        return {k: v for k, v in s.items() if v is not None}, notes

    def sanity(self, s):
        H, D, w, dn, hn = (_f(s[a]) for a in ("height", "diameter", "wall", "neck_diameter", "neck_height"))
        if not 0 < 2 * w < dn < D:
            return "needs 2 × wall < neck diameter < body diameter"
        if hn + (D - dn) / 2 + w >= H:
            return "the neck and shoulder don't fit in that height"
        return None

    def _profile(self, s):
        H, D, w, dn, hn = (_f(s[a]) for a in ("height", "diameter", "wall", "neck_diameter", "neck_height"))
        R, rn = D / 2, dn / 2
        zs = H - hn - (R - rn)          # where the 45° shoulder starts
        zn = zs + (R - rn)              # where the neck starts
        return H, R, rn, w, zs, zn

    def script(self, s):
        H, R, rn, w, zs, zn = self._profile(s)
        pts = [(0, 0), (R, 0), (R, zs), (rn, zn), (rn, H), (rn - w, H), (rn - w, zn), (R - w, zs), (R - w, w), (0, w)]
        return (f"# wall cross-section (radius, height) revolved about Z: body, 45° shoulder, open neck\n"
                f"profile = {[(round(x, 4), round(z, 4)) for x, z in pts]}\n"
                "with BuildPart() as p:\n"
                "    with BuildSketch(Plane.XZ):\n"
                "        Polygon(*profile, align=None)\n"
                "    revolve(axis=Axis.Z)\n"
                "result = p.part\n")

    def reference(self, s):
        from .cad import _b3d

        b = _b3d()
        H, R, rn, w, zs, zn = self._profile(s)
        # solid outer shape minus the inner cavity, each revolved on its own
        outer = b.revolve(b.Plane.XZ * b.Polygon((0, 0), (R, 0), (R, zs), (rn, zn), (rn, H), (0, H), align=None), b.Axis.Z)
        inner = b.revolve(b.Plane.XZ * b.Polygon((0, w), (R - w, w), (R - w, zs), (rn - w, zn), (rn - w, H + 1), (0, H + 1),
                                                 align=None), b.Axis.Z)
        return outer - inner

    def expected_bbox(self, s):
        return [_f(s["diameter"]), _f(s["diameter"]), _f(s["height"])]

    def extra_checks(self, part, s, m):
        H, R, rn, w, zs, zn = self._profile(s)
        cavity = math.pi * (R - w) ** 2 * (zs - w) + math.pi / 3 * (zn - zs) * ((R - w) ** 2 + (R - w) * (rn - w) + (rn - w) ** 2) \
            + math.pi * (rn - w) ** 2 * (H - zn)
        m["capacity_ml"] = round(cavity / 1000, 1)
        return {}

    def describe(self, name, s):
        from .describe import _article, _material

        dims = f"{s.get('height', '?')} mm tall, {s.get('diameter', '?')} mm diameter"
        mat = f" made from {_material(s['material'])[0]}" if "material" in s else ""
        neck = (f", with a {s.get('neck_diameter', '?')} mm neck {s.get('neck_height', '?')} mm tall"
                if "neck_diameter" in s or "neck_height" in s else "")
        wall = f" and {s['wall']} mm walls" if "wall" in s else ""
        return (f"The {name.replace('-', ' ')} is {_article(dims)} {dims} hollow bottle{mat}{neck}{wall}. The body is a cylinder "
                f"that narrows through a 45° shoulder into an open neck.")


# ---------------------------------------------------------------------- cross
class Cross(Family):
    def __init__(self):
        super().__init__("cross", "cross",
                         [("height", "mm", "height"), ("width", "mm", "width"), ("bar_width", "mm", "bar width"),
                          ("thickness", "mm", "thickness"), ("crossbar_from_top", "mm", "crossbar from top"),
                          ("material", "", "material")],
                         r"\bcross\b|crucifix|plus sign", ["height", "width", "bar_width", "thickness", "crossbar_from_top", "material"],
                         synonyms={"height": r"height|tall", "width": r"(?<!bar )width|\bwide\b|across", "bar_width": r"bar\s*width|\bbars?\b|\barms?\b",
                                   "thickness": r"thickness|thick", "crossbar_from_top": r"crossbar(?:\s*from\s*(?:the\s*)?top)?|from\s*the\s*top"})

    def part_name(self, t):
        m = re.search(r"\bcross[\s-]+(pendant|charm|ornament|badge|plaque|sign)\b", t)
        return f"cross-{m[1]}" if m else "greek-cross" if "greek" in t else "cross"

    def parse(self, t):
        notes = []
        s = {"height": _find(t, rf"{NUM}\s*mm\s*(?:tall|high)\b", rf"height\s*(?:of\s*)?{NUM}\s*mm"),
             "width": _find(t, rf"{NUM}\s*mm\s*(?:wide|across)\b(?!\s*(?:bars?|arms?))", rf"(?<!bar )width\s*(?:of\s*)?{NUM}\s*mm"),
             "bar_width": _find(t, rf"{NUM}\s*mm\s*(?:wide\s*)?(?:bars?|arms?|beams?)", rf"(?:bars?|arms?)\s*(?:are\s*|that\s*are\s*)?{NUM}\s*mm\s*wide",
                                rf"bar\s*width\s*(?:of\s*)?{NUM}\s*mm"),
             "thickness": _find(t, rf"{NUM}\s*mm\s*thick", rf"thickness\s*(?:of\s*)?{NUM}\s*mm"),
             "crossbar_from_top": _find(t, rf"crossbar[^.;]{{0,30}}?{NUM}\s*mm\s*(?:down\s*)?from\s*the\s*top",
                                        rf"{NUM}\s*mm\s*from\s*the\s*top")}
        if s["height"] and re.search(r"greek|plus sign|equal arms|centred crossbar|centered crossbar", t):
            s["crossbar_from_top"] = _g(_f(s["height"]) / 2)
            notes.append("Greek cross: the crossbar is centred, so it sits at half the height")
        if re.search(r"\bcross\b", t) and not re.search(r"greek|plus", t) and not s["crossbar_from_top"]:
            notes.append("a Latin cross needs its crossbar position (e.g. 'crossbar 25 mm from the top'); it is not assumed")
        return {k: v for k, v in s.items() if v is not None}, notes

    def sanity(self, s):
        h, w, b, c = (_f(s[a]) for a in ("height", "width", "bar_width", "crossbar_from_top"))
        if not (b < w and b < h and b / 2 <= c <= h - b / 2):
            return "needs bar width < width and height, and the crossbar inside the upright"
        return None

    def script(self, s):
        h, w, b, t, c = (_f(s[a]) for a in ("height", "width", "bar_width", "thickness", "crossbar_from_top"))
        return (f"H, W, B, T, C = {h:g}, {w:g}, {b:g}, {t:g}, {c:g}  # crossbar centre C mm below the top\n"
                "upright = Box(B, H, T)\n"
                "crossbar = Pos(0, H / 2 - C, 0) * Box(W, B, T)\n"
                "result = upright + crossbar\n")

    def reference(self, s):
        from .cad import _b3d

        b3 = _b3d()
        h, w, b, t, c = (_f(s[a]) for a in ("height", "width", "bar_width", "thickness", "crossbar_from_top"))
        y = h / 2 - c  # the outline of the cross as one 12-sided polygon, extruded
        pts = [(-b / 2, h / 2), (b / 2, h / 2), (b / 2, y + b / 2), (w / 2, y + b / 2), (w / 2, y - b / 2), (b / 2, y - b / 2),
               (b / 2, -h / 2), (-b / 2, -h / 2), (-b / 2, y - b / 2), (-w / 2, y - b / 2), (-w / 2, y + b / 2), (-b / 2, y + b / 2)]
        return b3.Pos(0, 0, -t / 2) * b3.extrude(b3.Polygon(*pts, align=None), t)

    def expected_bbox(self, s):
        return [_f(s["width"]), _f(s["height"]), _f(s["thickness"])]

    def describe(self, name, s):
        from .describe import _material

        kind = "Greek cross" if "height" in s and "crossbar_from_top" in s and _f(s["crossbar_from_top"]) == _f(s["height"]) / 2 \
            else "Latin cross"
        dims = " × ".join(str(s.get(a, "?")) for a in ("height", "width", "thickness")) + " mm"
        mat = f" made from {_material(s['material'])[0]}" if "material" in s else ""
        bar = f", with {s['bar_width']} mm wide bars" if "bar_width" in s else ""
        where = f" and the crossbar centred {s['crossbar_from_top']} mm below the top" if "crossbar_from_top" in s else ""
        return f"The {name.replace('-', ' ')} is a {dims} (height × width × thickness) {kind}{mat}{bar}{where}."


# ---------------------------------------------------------------------- star of david
class Star(Family):
    def __init__(self):
        super().__init__("star", "Star of David",
                         [("size", "mm", "point to point"), ("thickness", "mm", "thickness"), ("line_width", "mm", "line width"),
                          ("style", "", "style"), ("material", "", "material")],
                         r"star\s*of\s*david|magen\s*david|hexagram|six[\s-]*pointed\s*star", ["size", "thickness", "style", "material"],
                         synonyms={"size": r"size|across|point\s*to\s*point|diameter", "thickness": r"thickness|thick",
                                   "line_width": r"line(?:\s*width)?s?"})

    def part_name(self, t):
        m = re.search(r"\b(pendant|charm|ornament|badge|plaque)\b", t)
        return f"star-of-david-{m[1]}" if m else "star-of-david"

    def parse(self, t):
        notes = []
        s = {"size": _find(t, rf"{NUM}\s*mm\s*(?:across|point\s*to\s*point|tall|high|wide|in\s*diameter)",
                           rf"(?:size|diameter)\s*(?:of\s*)?{NUM}\s*mm", rf"{NUM}\s*mm\s*(?:star|hexagram)"),
             "thickness": _find(t, rf"{NUM}\s*mm\s*thick", rf"thickness\s*(?:of\s*)?{NUM}\s*mm"),
             "line_width": _find(t, rf"{NUM}\s*mm\s*(?:wide\s*)?lines?", rf"line\s*width\s*(?:of\s*)?{NUM}\s*mm")}
        if re.search(r"outline|lines?\b|interlac|hollow|open", t) or s["line_width"]:
            s["style"] = "outline"
        elif re.search(r"\bsolid\b|filled", t):
            s["style"] = "solid"
        else:
            notes.append("say whether the star is solid or an outline (two interlaced triangles); it is not assumed")
        return {k: v for k, v in s.items() if v is not None}, notes

    def needed(self, facts):
        miss = super().needed(facts)
        if facts.get("style") == "outline" and "line_width" not in facts:
            miss.append("line_width")
        return miss

    def sanity(self, s):
        if s.get("style") == "outline" and not 0 < 2 * _f(s["line_width"]) < _f(s["size"]) / 4:
            return "the line width is too large for that size"
        return None

    @staticmethod
    def _tri(r: float, rot: float) -> list[tuple[float, float]]:
        return [(round(r * math.cos(math.radians(a + rot)), 5), round(r * math.sin(math.radians(a + rot)), 5)) for a in (90, 210, 330)]

    def script(self, s):
        r, t = _f(s["size"]) / 2, _f(s["thickness"])
        if s.get("style") == "outline":
            lw = _f(s["line_width"])
            return (f"# two interlaced triangular rings; an equilateral triangle's edges move in by w when its\n"
                    f"# circumradius shrinks by 2w\n"
                    f"up, up_in = {self._tri(r, 0)}, {self._tri(r - 2 * lw, 0)}\n"
                    f"down, down_in = {self._tri(r, 180)}, {self._tri(r - 2 * lw, 180)}\n"
                    f"ring_up = extrude(Polygon(*up, align=None) - Polygon(*up_in, align=None), {t:g})\n"
                    f"ring_down = extrude(Polygon(*down, align=None) - Polygon(*down_in, align=None), {t:g})\n"
                    f"result = Pos(0, 0, {-t / 2:g}) * (ring_up + ring_down)\n")
        return (f"up, down = {self._tri(r, 0)}, {self._tri(r, 180)}\n"
                f"result = Pos(0, 0, {-t / 2:g}) * extrude(Polygon(*up, align=None) + Polygon(*down, align=None), {t:g})\n")

    def reference(self, s):
        from .cad import _b3d

        b = _b3d()
        r, t = _f(s["size"]) / 2, _f(s["thickness"])
        if s.get("style") == "outline":
            lw = _f(s["line_width"])
            rings = [b.RegularPolygon(r, 3, rotation=rot) - b.RegularPolygon(r - 2 * lw, 3, rotation=rot) for rot in (90, -90)]
            return b.Pos(0, 0, -t / 2) * (b.extrude(rings[0], t) + b.extrude(rings[1], t))
        return b.Pos(0, 0, -t / 2) * b.extrude(b.RegularPolygon(r, 3, rotation=90) + b.RegularPolygon(r, 3, rotation=-90), t)

    def expected_bbox(self, s):
        return [_f(s["size"]) * math.sqrt(3) / 2, _f(s["size"]), _f(s["thickness"])]

    def describe(self, name, s):
        from .describe import _material

        style = {"outline": f"drawn as an outline of two interlaced triangles with {s.get('line_width', '?')} mm lines",
                 "solid": "cut as a solid six-pointed shape"}.get(s.get("style"))
        clauses = [f"{s['thickness']} mm thick" if "thickness" in s else None,
                   f"made from {_material(s['material'])[0]}" if "material" in s else None, style]
        head = ("The Star of David is" if name == "star-of-david" else
                f"The {name.replace('star-of-david', 'Star of David').replace('-', ' ')} is a Star of David")
        return (f"{head} {s.get('size', '?')} mm point to point"
                + "".join(f", {c}" for c in clauses if c) + ".")


FAMILIES = {f.name: f for f in (Star(), Cross(), Bottle(), Plate())}  # detection order: most specific first


def detect(text: str) -> Family:
    t = text.lower()
    return next((f for f in FAMILIES.values() if f.name != "plate" and f.detect(t)), FAMILIES["plate"])
