"""Local web UI: benchmark results and a 3D viewer for the parts agents built.

    python -m lha.ui                    # http://127.0.0.1:8765, reads ./results
    python -m lha.ui --results other/ --port 9000

Stdlib HTTP server, no extra dependencies beyond build123d (to tessellate parts).
It only reads files under the results directory and binds to localhost.
"""

from __future__ import annotations

import argparse
import json
import re
import webbrowser
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

INDEX = Path(__file__).with_name("index.html")
_NAME = re.compile(r"^[\w.-]{1,120}$")  # run ids and part names; no path separators


class UI:
    def __init__(self, results: Path):
        self.results = results.resolve()
        self.runs = self.results / "runs"

    # ------------------------------------------------------------------ data
    def result_files(self) -> list[dict]:
        out = []
        for p in sorted(self.results.glob("*results.json")):
            try:
                rows = json.loads(p.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            env = next((r.get("env") for r in rows if r.get("env")), "design" if "design" in p.name else "incident")
            out.append({"file": p.name, "env": env, "live": p.name.startswith("live_"),
                        "mtime": p.stat().st_mtime, "rows": rows})
        return out

    def run_list(self) -> list[dict]:
        out = []
        if not self.runs.is_dir():
            return out
        for d in sorted(self.runs.iterdir()):
            if not d.is_dir():
                continue
            s = self._summary(d)
            parts = sorted(p.stem for p in (d / "cad").glob("*.brep")) if (d / "cad").is_dir() else []
            trace = d / "trace.jsonl"
            out.append({"run_id": d.name, "parts": parts, "mtime": max(d.stat().st_mtime, trace.stat().st_mtime if trace.exists() else 0),
                        "has_trace": trace.exists(),
                        **{k: s.get(k) for k in ("env", "agent", "messages", "seed", "score", "steps", "model", "live")}})
        return out

    def run_detail(self, run_id: str) -> dict:
        d = self._run_dir(run_id)
        s = self._summary(d)
        parts = {}
        for py in sorted((d / "cad").glob("*.py")) if (d / "cad").is_dir() else []:
            parts[py.stem] = {"script": py.read_text(), "has_mesh": py.with_suffix(".brep").exists()}
        for part, m in self._measures(str(d)).items():
            parts.setdefault(part, {})["measure"] = m
        facts = {}
        if (d / "state.json").exists():
            st = json.loads((d / "state.json").read_text())
            facts = {k: {"value": f.get("value"), "pinned": f.get("pinned"), "source": f.get("source")}
                     for k, f in st.get("facts", {}).items()}
        # Current Design Understanding of each part, from the agent's final working state (+ the trace,
        # for the latest engineering change). Derived here on request; never stored.
        from ..describe import tracker_for

        design = tracker_for(run_id)
        if (d / "trace.jsonl").exists():
            for line in (d / "trace.jsonl").read_text().splitlines():
                try:
                    design.feed(json.loads(line))
                except json.JSONDecodeError:
                    break
        flat = {k: f["value"] for k, f in facts.items()}
        names = set(parts) | {g.get("part") for g in (s.get("grade") or {}).values()}
        descriptions = {p: design.describe(flat, p) for p in sorted(n for n in names if n)}
        return {"run_id": run_id, "summary": s, "parts": parts, "facts": facts, "descriptions": descriptions}

    def trace(self, run_id: str, after: int = 0) -> dict:
        """Memory-inspector records with step > after (the page polls this while a run is live)."""
        d = self._run_dir(run_id)
        p = d / "trace.jsonl"
        if not p.exists():
            raise FileNotFoundError("this run has no trace (it predates the inspector, or is a naive run)")
        from ..describe import tracker_for

        design = tracker_for(run_id)  # regenerated from the state on every request, never stored
        recs = []
        for line in p.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:  # a line still being written
                break
            view = design.feed(r)
            if r["step"] > after:
                recs.append({**r, "design": view})
        m = re.search(r"-n(\d+)-", run_id)
        summary = self._summary(d)
        intent = json.loads((d / "intent.json").read_text()) if (d / "intent.json").exists() else None
        return {"records": recs, "total_messages": int(m[1]) if m else None, "intent": intent,
                "summary": summary or None, "done": bool(summary) or any(r["done"] for r in recs)}

    # -------------------------------------------------------------- design intent sessions
    def designs(self) -> list[dict]:
        from ..design_session import DesignSession

        return DesignSession.list(self.runs) if self.runs.is_dir() else []

    def design(self, sid: str) -> dict:
        from ..design_session import DesignSession

        return DesignSession.load(self.runs, sid).view()

    def design_create(self, body: dict) -> dict:
        from ..design_session import DesignSession
        from ..intent import interpret, interpret_rules

        request = str(body.get("request", "")).strip()
        if not request or len(request) > 2000:
            raise ValueError("describe the part in 1 to 2,000 characters")
        part = _safe_part(body.get("part"))
        it = interpret_rules(request, part) if body.get("interpreter") == "rules" else interpret(request, part)
        self.runs.mkdir(parents=True, exist_ok=True)
        return DesignSession.create(request, self.runs, interpretation=it).view()

    def design_change(self, body: dict) -> dict:
        from ..design_session import DesignSession

        s = DesignSession.load(self.runs, str(body.get("id", "")))
        s.change(str(body.get("text", ""))[:500])
        return s.view()

    def design_build(self, body: dict) -> dict:
        from ..design_session import DesignSession

        s = DesignSession.load(self.runs, str(body.get("id", "")))
        s.build()
        return s.view()

    def mesh(self, run_id: str, part: str, kind: str) -> dict:
        d = self._run_dir(run_id)
        if not _NAME.match(part):
            raise ValueError("bad part name")
        if kind == "ref":
            summary = self._summary(d)
            spec = summary.get("spec", {}).get(part)
            if not spec:
                raise FileNotFoundError(f"no reference spec for {part!r} in this run")
            return _ref_mesh(json.dumps(spec, sort_keys=True), summary.get("family", {}).get(part, "plate"))
        brep = d / "cad" / f"{part}.brep"
        if not brep.exists():
            raise FileNotFoundError(f"{part!r} has no saved shape (runs from before the UI existed only kept scripts)")
        return _brep_mesh(str(brep), brep.stat().st_mtime)

    # ------------------------------------------------------------------ helpers
    def _run_dir(self, run_id: str) -> Path:
        if not _NAME.match(run_id):
            raise ValueError("bad run id")
        d = (self.runs / run_id).resolve()
        if d.parent != self.runs or not d.is_dir():
            raise FileNotFoundError(f"no run {run_id!r}")
        return d

    @staticmethod
    def _summary(d: Path) -> dict:
        p = d / "summary.json"
        try:
            return json.loads(p.read_text()) if p.exists() else {}
        except json.JSONDecodeError:
            return {}

    @staticmethod
    @lru_cache(maxsize=64)
    def _measures_cached(run_dir: str, stamp: float) -> dict:
        from ..cad import measure

        b3d = _b3d()
        return {p.stem: measure(b3d.import_brep(str(p))) for p in sorted(Path(run_dir, "cad").glob("*.brep"))}

    def _measures(self, run_dir: str) -> dict:
        cad = Path(run_dir, "cad")
        if not cad.is_dir():
            return {}
        stamp = max((p.stat().st_mtime for p in cad.glob("*.brep")), default=0.0)
        return self._measures_cached(run_dir, stamp)


def _safe_part(v) -> str | None:
    s = re.sub(r"[^a-z0-9-]+", "-", str(v or "").lower()).strip("-")[:40]
    return s or None


def _b3d():
    import build123d

    return build123d


def _to_mesh(shape) -> dict:
    verts, tris = shape.tessellate(0.05, 0.2)
    bb = shape.bounding_box()
    return {"positions": [round(c, 4) for v in verts for c in (v.X, v.Y, v.Z)],
            "indices": [i for t in tris for i in t],
            "bbox": {"min": [bb.min.X, bb.min.Y, bb.min.Z], "max": [bb.max.X, bb.max.Y, bb.max.Z]}}


@lru_cache(maxsize=128)
def _brep_mesh(path: str, _mtime: float) -> dict:
    return _to_mesh(_b3d().import_brep(path))


@lru_cache(maxsize=128)
def _ref_mesh(spec_json: str, family: str = "plate") -> dict:
    from ..bench.design_desk import plate_script
    from ..cad import CadWorkspace
    from ..families import FAMILIES

    s = json.loads(spec_json)
    if family != "plate":
        return _to_mesh(FAMILIES[family].reference(s))
    script = plate_script(*(float(s[a]) for a in ("length", "width", "thickness", "hole_diameter")))
    return _to_mesh(CadWorkspace().build("ref", script))


def make_handler(ui: UI):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj, default=str).encode(), "application/json")

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path in ("/", "/index.html"):
                    return self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
                if u.path == "/api/results":
                    return self._json(ui.result_files())
                if u.path == "/api/runs":
                    return self._json(ui.run_list())
                if u.path == "/api/run":
                    return self._json(ui.run_detail(q.get("id", "")))
                if u.path == "/api/trace":
                    return self._json(ui.trace(q.get("run", ""), int(q.get("after", 0) or 0)))
                if u.path == "/api/designs":
                    return self._json(ui.designs())
                if u.path == "/api/design":
                    return self._json(ui.design(q.get("id", "")))
                if u.path == "/api/mesh":
                    return self._json(ui.mesh(q.get("run", ""), q.get("part", ""), q.get("kind", "agent")))
                return self._json({"error": "not found"}, 404)
            except (FileNotFoundError, ValueError, KeyError) as e:
                return self._json({"error": str(e)}, 404)
            except Exception as e:  # noqa: BLE001 - surface to the page, keep serving
                return self._json({"error": f"{type(e).__name__}: {e}"}, 500)

        def do_POST(self):
            # Writes are only accepted as JSON from this page: a cross-site form or fetch can't set this
            # content type without a CORS preflight (which is never answered), and the Origin must match.
            origin = self.headers.get("Origin")
            if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json" or \
                    (origin and urlparse(origin).netloc != self.headers.get("Host")):
                return self._json({"error": "forbidden"}, 403)
            try:
                length = min(int(self.headers.get("Content-Length", 0) or 0), 20_000)
                body = json.loads(self.rfile.read(length) or b"{}")
                route = {"/api/design": ui.design_create, "/api/design/change": ui.design_change,
                         "/api/design/build": ui.design_build}.get(urlparse(self.path).path)
                if not route:
                    return self._json({"error": "not found"}, 404)
                return self._json(route(body))
            except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as e:
                return self._json({"error": str(e)}, 400)
            except Exception as e:  # noqa: BLE001
                return self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    return Handler


def main(argv=None) -> None:
    from ..config import load_dotenv

    load_dotenv()
    ap = argparse.ArgumentParser(prog="python -m lha.ui")
    ap.add_argument("--results", default="results")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    ui = UI(Path(a.results))
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(ui))
    url = f"http://127.0.0.1:{a.port}/"
    print(f"LHA UI on {url}  (results: {ui.results})  Ctrl-C to stop")
    if not a.no_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
