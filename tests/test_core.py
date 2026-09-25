import json

from lha.agent import NaiveAgent, StatefulAgent, parse_response
from lha.archive import Archive
from lha.bench.incident_desk import IncidentDesk, naive_policy, stateful_policy
from lha.compactor import Compactor
from lha.llm import ScriptedLLM
from lha.state import WorkingState, apply_ops
from lha.tools import ToolBox


def test_set_fact_overwrites_and_archives_stale_value(tmp_path):
    s, arc = WorkingState(goal="g"), Archive(tmp_path / "a.jsonl", "r")
    put = lambda kind, key, payload: arc.put(s.step, kind, key, payload)
    apply_ops(s, [{"op": "set_fact", "key": "db.region", "value": "us-east-1"}], put)
    s.step = 5
    apply_ops(s, [{"op": "set_fact", "key": "db.region", "value": "eu-west-1"}], put)
    assert s.facts["db.region"].value == "eu-west-1"
    assert "us-east-1" not in s.render()  # stale belief is not in the prompt...
    assert arc.recall("db.region", kinds=["superseded_fact"])[0].payload["value"] == "us-east-1"  # ...but recoverable


def test_bad_ops_are_reported_not_raised():
    s = WorkingState()
    errs = apply_ops(s, [{"op": "update_task", "id": "nope", "status": "done"}, {"op": "explode"}])
    assert len(errs) == 2


def test_notes_are_a_bounded_ring():
    s = WorkingState(max_notes=3)
    apply_ops(s, [{"op": "note", "text": str(i)} for i in range(10)])
    assert s.notes == ["7", "8", "9"]


def test_compactor_respects_budget_and_pins(tmp_path):
    s, arc = WorkingState(goal="g"), Archive(None, "r")
    apply_ops(s, [{"op": "set_fact", "key": f"k{i}", "value": "x" * 40} for i in range(50)])
    apply_ops(s, [{"op": "pin", "key": "k0"}])
    Compactor(budget_tokens=300).enforce(s, arc)
    assert s.tokens() <= 300
    assert "k0" in s.facts
    assert any(it.kind == "evicted_fact" for it in arc.items)


def test_parse_response_handles_fences_and_prose():
    assert parse_response('sure!\n```json\n{"action": {"tool": "x"}}\n```')["action"]["tool"] == "x"
    assert parse_response('ok {"a": "}{", "b": {"c": 1}} trailing')["b"]["c"] == 1


def _stateful(env, tmp_path, run_id="r1", budget=450, max_steps=10_000):
    return StatefulAgent(ScriptedLLM(stateful_policy), ToolBox(env.tools()), env.goal(), run_dir=tmp_path,
                         run_id=run_id, compactor=Compactor(budget_tokens=budget), max_steps=max_steps)


def test_stateful_prompt_is_flat_while_naive_grows(tmp_path):
    env_a, env_b = IncidentDesk(n_messages=300, seed=1), IncidentDesk(n_messages=300, seed=1)
    a = _stateful(env_a, tmp_path)
    assert env_a.score(a.run()) == 1.0
    b = NaiveAgent(ScriptedLLM(naive_policy), ToolBox(env_b.tools()), env_b.goal(), max_steps=10_000)
    assert env_b.score(b.run()) == 1.0
    pa, pb = a.stats.prompt_tokens_per_step, b.stats.prompt_tokens_per_step
    assert max(pa) < 1000 and pb[-1] > 10 * pa[-1]
    assert a.stats.input_tokens < b.stats.input_tokens / 5


def test_crash_and_resume_from_checkpoint(tmp_path):
    env = IncidentDesk(n_messages=150, seed=2)
    first = _stateful(env, tmp_path, max_steps=80)
    first.run()  # "crashes" (runs out of steps) mid-inbox
    assert not first.finished
    resumed = _stateful(env, tmp_path)  # new process, same run id
    assert resumed.state.step == 80
    assert env.score(resumed.run()) == 1.0
    assert json.loads((tmp_path / "r1" / "state.json").read_text())["done"]


def test_recall_matches_keys_however_the_model_spelled_them(tmp_path):
    from lha.tools import recall_tool

    arc = Archive(None, "r")
    arc.put(3, "evicted_fact", "base-plate_width", {"key": "base-plate_width", "value": "30", "source": "ECO-5242"})
    arc.put(4, "observation", "recall", {"args": {"query": "base-plate width"}, "result": "recall: no matches"})
    out = recall_tool(arc).fn({"query": "base-plate width"})
    assert out == '[step 3] evicted_fact base-plate_width = "30" (src: ECO-5242)'  # not its own past query
    assert "base-plate_width" in recall_tool(arc).fn({"query": "base-plate.width"})


def test_load_dotenv_skips_blanks_and_keeps_shell_values(tmp_path, monkeypatch):
    from lha.config import load_dotenv

    (tmp_path / ".env").write_text('# c\nLHA_T_A="x"\nLHA_T_EMPTY=\nexport LHA_T_B=y\nLHA_T_SHELL=file\n')
    monkeypatch.setenv("LHA_T_SHELL", "shell")
    for k in ("LHA_T_A", "LHA_T_B", "LHA_T_EMPTY"):
        monkeypatch.delenv(k, raising=False)
    load_dotenv(tmp_path / ".env")
    import os
    assert (os.environ["LHA_T_A"], os.environ["LHA_T_B"], os.environ["LHA_T_SHELL"]) == ("x", "y", "shell")
    assert "LHA_T_EMPTY" not in os.environ


def test_load_dotenv_falls_back_to_repo_root(tmp_path, monkeypatch):
    from lha import config

    (tmp_path / ".env").write_text("LHA_T_ROOT=from-repo\n")
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    monkeypatch.chdir(tmp_path / "..")  # run from a folder with no .env
    monkeypatch.delenv("LHA_T_ROOT", raising=False)
    config.load_dotenv()
    import os
    assert os.environ["LHA_T_ROOT"] == "from-repo"
