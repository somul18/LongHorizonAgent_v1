import json

import pytest

from lha.intent import INTENT_SOURCE, interpret_change, interpret_rules


def test_complete_specification():
    it = interpret_rules("Create a motor mounting plate 100 mm long, 80 mm wide and 4 mm thick from 6061 aluminum. "
                         "Add four 5.3 mm through-holes, one near each corner, with the hole centers 6 mm from the "
                         "adjacent edges.")
    assert it.part == "motor-mount" and it.unknown == [] and it.notes == []
    assert it.facts == {"length": "100", "width": "80", "thickness": "4", "hole_diameter": "5.3", "material": "al6061"}
    assert all(op["source"] == INTENT_SOURCE for op in it.ops())


def test_incomplete_specification_is_not_guessed():
    it = interpret_rules("Make me an 80 × 60 mm steel sensor mounting plate with four M4 mounting holes.")
    assert it.facts == {"length": "80", "width": "60", "hole_diameter": "4.3", "material": "steel"}
    assert it.unknown == ["thickness"] and any("M4" in n for n in it.notes)  # the M4 -> 4.3 mm convention is shown
    assert {"op": "add_question", "text": "sensor-bracket.thickness is not specified in the design intent"} in it.ops()


def test_family_limits_become_notes_not_facts():
    it = interpret_rules("a titanium lid 120x70x3mm, six M5 holes 8 mm from the edges")
    assert it.facts["hole_diameter"] == "5.3"  # "8 mm from the edges" is an inset, not a diameter
    assert any("6 holes" in n for n in it.notes) and any("8 mm from the edges" in n for n in it.notes)


@pytest.mark.parametrize("text,expected", [
    ("ECO-1847: Reduce motor_mount thickness from 6mm to 4mm.", ({"thickness": "4"}, "ECO-1847")),
    ("ECO-2217 material: ABS → Al6061", ({"material": "al6061"}, "ECO-2217")),
    ("ECO-6713 length: 95 mm → 110 mm", ({"length": "110"}, "ECO-6713")),
    ("ECO-9211 hole diameter 5.3 mm to 6.4 mm", ({"hole_diameter": "6.4"}, "ECO-9211")),
    ("thickness 5 mm", ({"thickness": "5"}, "user")),
])
def test_changes_take_the_new_value(text, expected):
    assert interpret_change(text, "p") == expected


def test_design_session_intent_to_validated_cad(tmp_path):
    pytest.importorskip("build123d")
    from lha.design_session import DesignSession

    text = "Make me an 80 × 60 mm steel sensor mounting plate with four M4 mounting holes."
    s = DesignSession.create(text, tmp_path, interpretation=interpret_rules(text))
    assert "Missing requirement: thickness" in s.understanding()
    with pytest.raises(ValueError, match="thickness not specified"):
        s.build()
    s.change("thickness 5 mm")
    assert s.state.questions == []                    # the answer closes the open question
    assert s.build()["ok"]
    s.change("ECO-1847: thickness 5 mm → 3 mm")
    v = s.view()
    assert {f["attr"]: f["source"] for f in v["facts"]}["thickness"] == "ECO-1847"
    assert {f["attr"]: f["source"] for f in v["facts"]}["width"] == "design_intent"
    assert v["build"]["stale"] and "reduced the thickness from 5 mm to 3 mm" in v["understanding"]
    assert [e["label"] for e in v["evolution"]] == ["Design intent", "User", "ECO-1847"]
    assert any(it.kind == "superseded_fact" and it.payload["value"] == "5" for it in s.archive.items)
    assert json.loads((s.dir / "intent.json").read_text())["request"] == text  # the original intent is untouched


def test_designdesk_intent_run_keeps_provenance(tmp_path):
    pytest.importorskip("build123d")
    from lha.bench.run import run_one

    r = run_one("stateful", 150, 3, 700, False, tmp_path, "design", intent=True)
    run_dir = tmp_path / "runs" / "design_stateful-n150-s3-intent"
    meta = json.loads((run_dir / "intent.json").read_text())
    recs = [json.loads(line) for line in (run_dir / "trace.jsonl").read_text().splitlines()]
    first = recs[0]["sources"]
    assert r["score"] == 1.0 and meta["part"] in json.loads((run_dir / "summary.json").read_text())["spec"]
    assert {k for k, v in first.items() if v == INTENT_SOURCE} == {f"{meta['part']}.{a}" for a in meta["interpretation"]["facts"]}
    assert any(src.startswith("ECO-") for r in recs for src in r["sources"].values())


def test_dashboard_accepts_design_writes_only_as_same_origin_json(tmp_path):
    pytest.importorskip("build123d")
    import threading
    import urllib.error
    import urllib.request
    from http.server import ThreadingHTTPServer

    from lha.ui.server import UI, make_handler

    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(UI(tmp_path)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/api/design"
    body = json.dumps({"request": "Create a 95 × 45 × 6 mm rail clamp from ABS with four 4.3 mm mounting holes.",
                       "interpreter": "rules"}).encode()

    def post(headers):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, body, headers, method="POST")) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, None

    try:
        assert post({"Content-Type": "text/plain"})[0] == 403                     # a plain cross-site form post
        assert post({"Content-Type": "application/json", "Origin": "https://evil.example"})[0] == 403
        code, view = post({"Content-Type": "application/json"})
        assert code == 200 and view["part"] == "rail-clamp" and not view["missing"]
    finally:
        srv.shutdown()
