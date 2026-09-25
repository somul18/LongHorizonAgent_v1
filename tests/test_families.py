"""Water bottle, cross and Star of David: the same intent -> state -> understanding -> CAD flow as plates."""

import pytest

from lha.design_session import DesignSession
from lha.intent import interpret_change, interpret_rules

BOTTLE = ("Create a water bottle 250 mm tall with a 70 mm body diameter, a 28 mm neck that is 20 mm tall, "
          "and 2 mm walls, from PET.")
CROSS = ("Create a brass cross pendant 60 mm tall and 40 mm wide, with 8 mm wide bars, 3 mm thick, "
         "the crossbar centred 20 mm from the top.")
STAR = "Create a Star of David 50 mm across from sterling silver, 2 mm thick, drawn as an outline with 3 mm wide lines."


@pytest.mark.parametrize("text,family,part,facts", [
    (BOTTLE, "bottle", "water-bottle", {"height": "250", "diameter": "70", "wall": "2", "neck_diameter": "28",
                                        "neck_height": "20", "material": "pet"}),
    (CROSS, "cross", "cross-pendant", {"height": "60", "width": "40", "bar_width": "8", "thickness": "3",
                                       "crossbar_from_top": "20", "material": "brass"}),
    (STAR, "star", "star-of-david", {"size": "50", "thickness": "2", "line_width": "3", "style": "outline",
                                     "material": "silver"}),
])
def test_family_requests_are_read_exactly(text, family, part, facts):
    it = interpret_rules(text)
    assert (it.family, it.part, it.facts, it.unknown) == (family, part, facts, [])


def test_unstated_style_is_asked_not_assumed():
    it = interpret_rules("Make a Star of David 40 mm across and 3 mm thick in gold.")
    assert it.unknown == ["style"] and "style" not in it.facts


@pytest.mark.parametrize("family,text,values", [
    ("bottle", "ECO-12: wall 2 mm -> 1.5 mm", {"wall": "1.5"}),
    ("bottle", "neck height 25 mm", {"neck_height": "25"}),
    ("cross", "ECO-3: crossbar to 15 mm from the top", {"crossbar_from_top": "15"}),
    ("cross", "bars 6 mm wide", {"bar_width": "6"}),
    ("star", "ECO-9: switch from sterling silver to gold", {"material": "gold"}),
    ("star", "make it solid", {"style": "solid"}),
])
def test_family_changes(family, text, values):
    assert interpret_change(text, "x", family)[0] == values


def test_stainless_steel_change_is_ss304():
    assert interpret_change("switch to stainless steel", "p")[0] == {"material": "ss304"}


@pytest.mark.parametrize("text,change", [(BOTTLE, "ECO-12: wall 2 mm -> 1.5 mm"),
                                         (CROSS, "ECO-3: crossbar to 18 mm from the top"),
                                         (STAR, "ECO-5: make it solid")])
def test_build_validates_against_reference(tmp_path, text, change):
    s = DesignSession.create(text, tmp_path, interpretation=interpret_rules(text))
    assert s.build()["ok"]
    s.change(change)
    b = s.build()
    assert b["ok"] and all(b["checks"].values()), b["checks"]
    assert "latest approved engineering change" in s.understanding()


def test_bottle_reports_capacity(tmp_path):
    s = DesignSession.create(BOTTLE, tmp_path, interpretation=interpret_rules(BOTTLE))
    assert 700 < s.build()["measure"]["capacity_ml"] < 800


def test_change_that_opens_a_requirement_asks_for_it(tmp_path):
    text = "Make a Star of David 40 mm across and 3 mm thick in gold."
    s = DesignSession.create(text, tmp_path, interpretation=interpret_rules(text))
    with pytest.raises(ValueError, match="style"):
        s.build()
    s.change("outline")
    assert s.missing() == ["line_width"] and any("line_width" in q for q in s.state.questions)
    s.change("line width 2.5 mm")
    assert s.missing() == [] and s.build()["ok"]
