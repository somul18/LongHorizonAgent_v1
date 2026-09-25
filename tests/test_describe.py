from lha.describe import Change, DesignTracker, describe_part


def test_simple_plate():
    text = describe_part("", {"length": "100", "width": "80", "thickness": "4", "material": "al6061",
                              "holes": 4, "hole_diameter": "5.3"})
    assert text == ("A 100 × 80 × 4 mm rectangular mounting plate made from 6061 aluminum, "
                    "with four 5.3 mm mounting holes.")


def test_after_an_engineering_change():
    text = describe_part("motor-mount", {"length": "100", "width": "80", "thickness": "4",
                                         "hole_diameter": "5.3", "material": "al6061"},
                         last_change=Change("thickness", "6", "4", "ECO-1847"), hole_inset=6)
    assert text.startswith("The motor mount is a 100 × 80 × 4 mm rectangular mounting plate made from 6061 aluminum")
    assert "four 5.3 mm through-holes, one near each corner, 6 mm in from each edge" in text
    assert text.endswith("The latest approved engineering change (ECO-1847) reduced the thickness from 6 mm to 4 mm.")


def test_bracket_with_walls_and_polymer_note():
    text = describe_part("", {"type": "bracket", "width": "80", "height": "60", "depth": "40", "wall": "3",
                              "material": "abs", "mounting_holes": 4})
    assert text == ("An 80 × 60 × 40 mm mounting bracket made from ABS, with 3 mm walls and four mounting holes. "
                    "As an ABS part it suits additive manufacturing.")


def test_only_reports_what_the_state_holds():
    text = describe_part("lid", {"length": "110", "thickness": "4"}, hole_inset=6,
                         expected=("length", "width", "thickness", "hole_diameter", "material"))
    assert "110 × ? × 4 mm" in text and "Not in the working state right now: width, hole diameter, material" in text


def test_tracker_follows_the_trace_and_attributes_the_eco():
    t = DesignTracker(hole_inset=6, expected=(), kind="plate")
    facts = {"gusset.length": "75", "gusset.width": "45", "gusset.thickness": "4", "gusset.hole_diameter": "5.3",
             "gusset.material": "steel"}
    t.feed({"ops": [], "changes": [], "facts": {}, "observation": "msg 9/50: [ECO-4332] gusset: hole_diameter changed to 4.3"})
    view = t.feed({"ops": [{"op": "set_fact", "key": "gusset.hole_diameter", "value": "4.3"}],
                   "changes": [["gusset.hole_diameter", "5.3", "4.3"]],
                   "facts": {**facts, "gusset.hole_diameter": "4.3"}, "observation": ""})
    assert view["part"] == "gusset"
    assert "four 4.3 mm through-holes" in view["text"]
    assert "(ECO-4332) reduced the hole diameter from 5.3 mm to 4.3 mm" in view["text"]
