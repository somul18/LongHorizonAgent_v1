"""CAD tools on build123d (OpenCascade), headless and in-process.

The agent models parts by writing short build123d scripts. A script runs with
`from build123d import *` already in scope and must assign the finished solid
to `result`. Each build is stored under a name in a CadWorkspace and answered
with a one-line measurement (validity, bounding box, volume, hole count, mass),
so the observation stays small and every number is checkable.

    ws = CadWorkspace("runs/my-run/cad")
    tools = ToolBox([*ws.tools(), ...])

Scripts are executed with `exec`, i.e. they are trusted code. Run the agent in a
sandbox (a container like this one) when the model writing them is untrusted.

build123d is an optional dependency: `pip install -e ".[cad]"`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .tools import Tool

# g/cm^3. The agent passes a density (or one of these names) to get a mass.
DENSITIES = {"al6061": 2.70, "steel": 7.85, "ss304": 8.00, "ti6al4v": 4.43, "abs": 1.04, "pla": 1.24, "pa12": 1.01}

SCRIPT_HELP = ("build123d script; `from build123d import *` is pre-imported; assign the solid to `result`. "
               "Units are mm. Example: result = Box(80, 40, 5) - Pos(30, 10, 0) * Cylinder(2.15, 5)")


def _b3d():
    try:
        import build123d
    except ImportError as e:  # keep the core package dependency-free
        raise RuntimeError('build123d is not installed: pip install -e ".[cad]"') from e
    return build123d


def density_of(material_or_density: Any) -> float | None:
    if material_or_density in (None, ""):
        return None
    if isinstance(material_or_density, (int, float)):
        return float(material_or_density)
    key = str(material_or_density).strip().lower()
    if key in DENSITIES:
        return DENSITIES[key]
    return float(key)  # "2.7" -> 2.7; anything else raises a readable error


def measure(part, density: float | None = None) -> dict:
    """Numbers an agent (or a grader) can check. mass_g needs a density in g/cm^3."""
    b = _b3d()
    bb = part.bounding_box()
    vol = float(part.volume)
    out = {
        "valid": bool(part.is_valid),
        "solids": len(part.solids()),
        "bbox_mm": [round(bb.size.X, 3), round(bb.size.Y, 3), round(bb.size.Z, 3)],
        "volume_mm3": round(vol, 2),
        "faces": len(part.faces()),
        "cyl_faces": len(part.faces().filter_by(b.GeomType.CYLINDER)),
    }
    if density is not None:
        out["density_g_cm3"] = density
        out["mass_g"] = round(vol * density / 1000.0, 3)
    return out


def geometry_matches(part, ref, rel_tol: float = 1e-3) -> bool:
    """Same shape up to translation: bbox sizes agree and the symmetric difference
    is (numerically) empty after aligning bounding-box centres."""
    b = _b3d()
    if part is None or not part.is_valid:
        return False
    pb, rb = part.bounding_box(), ref.bounding_box()
    if any(abs(p - r) > 0.01 for p, r in zip(pb.size, rb.size)):
        return False
    moved = b.Pos(rb.center() - pb.center()) * part
    diff = (moved - ref).volume + (ref - moved).volume
    return diff <= rel_tol * ref.volume


class CadWorkspace:
    """Named parts plus the tools that build, measure, list and export them."""

    def __init__(self, out_dir: str | Path | None = None):
        self.out_dir = Path(out_dir) if out_dir else None
        self.parts: dict[str, Any] = {}
        self.scripts: dict[str, str] = {}

    def build(self, name: str, script: str) -> Any:
        b = _b3d()
        ns: dict[str, Any] = {k: getattr(b, k) for k in dir(b) if not k.startswith("_")}
        exec(compile(script, f"<cad:{name}>", "exec"), ns)
        result = ns.get("result")
        if result is not None and hasattr(result, "part") and not hasattr(result, "volume"):
            result = result.part  # a BuildPart context was assigned
        if result is None or not hasattr(result, "volume"):
            raise ValueError("script must assign a solid to `result`")
        self.parts[name] = result
        self.scripts[name] = script
        if self.out_dir:  # keep the source next to the exports; the prompt never holds it
            self.out_dir.mkdir(parents=True, exist_ok=True)
            (self.out_dir / f"{_safe(name)}.py").write_text(script)
            b.export_brep(result, str(self.out_dir / f"{_safe(name)}.brep"))  # exact shape, for the UI
        return result

    def _get(self, name: str):
        if name not in self.parts:
            raise KeyError(f"no part {name!r}; built: {', '.join(self.parts) or 'none'}")
        return self.parts[name]

    def tools(self) -> list[Tool]:
        def cad_build(args: dict) -> str:
            name = str(args.get("name") or "part")
            part = self.build(name, str(args["script"]))
            m = measure(part, density_of(args.get("density") or args.get("material")))
            return f"built {name!r}: {json.dumps(m)}"

        def cad_measure(args: dict) -> str:
            name = str(args.get("name") or "part")
            return f"{name!r}: {json.dumps(measure(self._get(name), density_of(args.get('density') or args.get('material'))))}"

        def cad_list(_args: dict) -> str:
            if not self.parts:
                return "no parts built yet"
            return "\n".join(f"- {n}: {json.dumps(measure(p))}" for n, p in self.parts.items())

        def cad_export(args: dict) -> str:
            b = _b3d()
            if not self.out_dir:
                raise RuntimeError("workspace has no out_dir")
            name, fmt = str(args.get("name") or "part"), str(args.get("format", "step")).lower()
            if fmt not in ("step", "stl"):
                raise ValueError("format must be step or stl")
            path = self.out_dir / f"{_safe(name)}.{fmt}"
            path.parent.mkdir(parents=True, exist_ok=True)
            (b.export_step if fmt == "step" else b.export_stl)(self._get(name), str(path))
            return f"exported {name!r} -> {path}"

        mats = ", ".join(DENSITIES)
        return [
            Tool("cad_build", f'build or rebuild a named part. args: {{"name": str, "script": str, '
                              f'"material": optional ({mats}) or "density": g/cm3}}. script = {SCRIPT_HELP}', cad_build),
            Tool("cad_measure", 'measure a built part. args: {"name": str, "material" or "density": optional}', cad_measure),
            Tool("cad_list", "list built parts with measurements. args: {}", cad_list),
            Tool("cad_export", 'write a built part to STEP or STL. args: {"name": str, "format": "step"|"stl"}', cad_export),
        ]


def _safe(name: str) -> str:
    return re.sub(r"[^\w.-]", "_", name)[:80] or "part"


def cad_tools(out_dir: str | Path | None = None) -> list[Tool]:
    return CadWorkspace(out_dir).tools()
