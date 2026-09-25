import pytest

pytest.importorskip("build123d")

from lha.agent import NaiveAgent, StatefulAgent  # noqa: E402
from lha.bench.design_desk import DesignDesk, naive_policy, plate_script, stateful_policy  # noqa: E402
from lha.cad import CadWorkspace, geometry_matches  # noqa: E402
from lha.compactor import Compactor  # noqa: E402
from lha.llm import ScriptedLLM  # noqa: E402
from lha.tools import ToolBox  # noqa: E402


def test_cad_tools_build_measure_export(tmp_path):
    ws = CadWorkspace(tmp_path)
    box = ToolBox(ws.tools())
    obs = box.call("cad_build", {"name": "p", "script": "result = Box(80, 40, 5)", "material": "al6061"})
    assert '"bbox_mm": [80.0, 40.0, 5.0]' in obs and '"mass_g": 43.2' in obs
    assert '"mass_g": 125.6' in box.call("cad_measure", {"name": "p", "material": "steel"})
    assert "exported" in box.call("cad_export", {"name": "p", "format": "step"})
    assert (tmp_path / "p.step").stat().st_size > 0 and (tmp_path / "p.py").exists()
    # model mistakes come back as observations, not crashes
    assert box.call("cad_build", {"name": "q", "script": "x = 1"}).startswith("ERROR")
    assert box.call("cad_measure", {"name": "nope"}).startswith("ERROR")


def test_blind_holes_are_not_through_holes():
    from lha.cad import measure

    ws = CadWorkspace()
    through = ws.build("t", "result = Box(95, 75, 8) - Pos(40, 30, 0) * Cylinder(1.6, 8)")
    blind = ws.build("b", "result = Box(95, 75, 8) - Pos(40, 30, 4) * Cylinder(1.6, 8)")  # cutter half above the plate
    assert measure(through)["z_through_holes"] == 1
    assert measure(blind)["z_through_holes"] == 0 and measure(blind)["cyl_faces"] == 1


def test_geometry_match_ignores_translation_but_not_hole_position():
    ws = CadWorkspace()
    ref = ws.build("ref", plate_script(80, 40, 5, 4.3))
    moved = ws.build("moved", "result = Pos(10, -3, 7) * (" + "Box(80, 40, 5))")
    assert not geometry_matches(moved, ref)  # no holes
    shifted = ws.build("shifted", plate_script(80, 40, 5, 4.3) + "result = Pos(10, -3, 7) * result\n")
    assert geometry_matches(shifted, ref)
    wrong_hole = ws.build("wrong", plate_script(80, 40, 5, 5.3))
    assert not geometry_matches(wrong_hole, ref)


def test_design_desk_stateful_flat_and_correct(tmp_path):
    env_a, env_b = DesignDesk(n_messages=300, seed=3), DesignDesk(n_messages=300, seed=3)
    a = StatefulAgent(ScriptedLLM(stateful_policy), ToolBox(env_a.tools()), env_a.goal(), run_dir=tmp_path,
                      run_id="d1", compactor=Compactor(budget_tokens=600), max_steps=10_000)
    assert env_a.score(a.run()) == 1.0
    assert any(it.kind == "evicted_fact" for it in a.archive.items)  # it had to recall to get there
    b = NaiveAgent(ScriptedLLM(naive_policy), ToolBox(env_b.tools()), env_b.goal(), max_steps=10_000)
    assert env_b.score(b.run()) == 1.0
    pa, pb = a.stats.prompt_tokens_per_step, b.stats.prompt_tokens_per_step
    assert max(pa) < 1500 and pb[-1] > 5 * pa[-1]


def test_design_desk_grades_stale_spec_as_wrong():
    env = DesignDesk(n_messages=400, seed=5)
    qid, part = next(iter(env.questions.items()))
    s = {k: float(v) for k, v in env.spec(part).items() if k != "material"}
    s["thickness"] += 1.0  # e.g. a value from an ECO that was later superseded
    env.workspace.build(part, plate_script(s["length"], s["width"], s["thickness"], s["hole_diameter"]))
    g = env.grade({qid: {"part": part, "mass_g": 0}})
    assert g[qid] == {**g[qid], "geometry": False, "mass": False}
    assert env.score({}) == 0.0


def test_ui_serves_runs_grades_and_meshes(tmp_path):
    from lha.bench.run import run_one
    from lha.ui.server import UI

    run_one("stateful", 60, 1, 700, False, tmp_path, "design")
    ui = UI(tmp_path)
    (run,) = ui.run_list()
    assert run["run_id"] == "design_stateful-n60-s1" and run["score"] == 1.0 and len(run["parts"]) == 3
    d = ui.run_detail(run["run_id"])
    part = run["parts"][0]
    assert d["parts"][part]["measure"]["valid"] and "result" in d["parts"][part]["script"]
    agent, ref = ui.mesh(run["run_id"], part, "agent"), ui.mesh(run["run_id"], part, "ref")
    assert agent["bbox"]["max"] == pytest.approx(ref["bbox"]["max"]) and len(agent["indices"]) % 3 == 0
    with pytest.raises(ValueError):
        ui.run_detail("../etc")
