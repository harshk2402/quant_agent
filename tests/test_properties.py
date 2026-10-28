# SPDX-License-Identifier: MIT
"""Tests for the generated-checker stage.

This exists because of what the dev run reported: 81 of 86 units `domain_gate_failed` -- the code
ran, the files were well-formed, the numbers were wrong. The structural layer cannot reach that,
and neither can hand-written invariants: on the measured all-zeros failure not one of them fired,
because they key on column names like `price` and the file's keys were `american_put`.

A generated checker is not trusted, and most of what is tested here is the distrust: it runs on a
copy, a crash is ignored, and nothing it says fails a unit by itself.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from agent import properties  # noqa: E402


def _out(tmp_path, **files):
    d = tmp_path / "out"
    d.mkdir(exist_ok=True)
    for name, body in files.items():
        (d / name).write_text(body)
    return d


CHECKER = '''
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
try:
    v = json.loads((out / "option_values.json").read_text())
    for k, val in v.items():
        if val == 0.0:
            print(f"VIOLATION: {k} is 0.0, but an at-the-money option with a year to run cannot be worthless")
except Exception as e:
    print("VIOLATION: could not read option_values.json:", e)
'''


def test_it_reports_what_the_checker_finds(tmp_path):
    """The measured failure: eight option values, all exactly 0.0, every structural check passing."""
    out = _out(tmp_path, **{"option_values.json": json.dumps(
        {"american_put": 0.0, "european_call": 0.0})})
    found = properties.run_properties(CHECKER, out)
    assert len(found) == 2
    assert "american_put is 0.0" in found[0]


def test_a_correct_answer_reports_nothing(tmp_path):
    out = _out(tmp_path, **{"option_values.json": json.dumps(
        {"american_put": 6.23, "european_call": 5.91})})
    assert properties.run_properties(CHECKER, out) == []


def test_the_checker_cannot_touch_the_real_deliverables(tmp_path):
    """It is generated code. "Never write a file" is an instruction, not a control -- so it runs
    against a copy."""
    out = _out(tmp_path, **{"results.json": '{"a": 1}'})
    vandal = ('import pathlib, sys\n'
              'for f in pathlib.Path(sys.argv[1]).iterdir(): f.unlink()\n'
              '(pathlib.Path(sys.argv[1]) / "reward.json").write_text("{}")\n')
    properties.run_properties(vandal, out)
    assert (out / "results.json").read_text() == '{"a": 1}'
    assert not (out / "reward.json").exists()


def test_a_checker_that_crashes_reports_nothing(tmp_path):
    out = _out(tmp_path, **{"results.json": "{}"})
    assert properties.run_properties("raise SystemExit('boom')", out) == []
    assert properties.run_properties("this is not python", out) == []


def test_a_checker_that_hangs_is_abandoned(tmp_path):
    out = _out(tmp_path, **{"results.json": "{}"})
    assert properties.run_properties("import time\ntime.sleep(30)\n", out, timeout=1.0) == []


def test_an_empty_checker_is_a_no_op(tmp_path):
    out = _out(tmp_path, **{"results.json": "{}"})
    assert properties.run_properties("", out) == []
    assert properties.run_properties("   \n", out) == []


def test_a_flood_of_violations_is_capped(tmp_path):
    """A checker reporting fifty problems has misunderstood the task, and a wall of text would
    crowd the real failure out of the repair message."""
    out = _out(tmp_path, **{"results.json": "{}"})
    noisy = 'for i in range(50): print("VIOLATION: problem %d" % i)\n'
    assert len(properties.run_properties(noisy, out)) == properties.MAX_VIOLATIONS


def test_output_that_is_not_a_violation_line_is_ignored(tmp_path):
    out = _out(tmp_path, **{"results.json": "{}"})
    chatty = ('print("checking the discount factors...")\n'
              'print("VIOLATION: DF(2) exceeds DF(1)")\n'
              'print("done")\n')
    assert properties.run_properties(chatty, out) == ["DF(2) exceeds DF(1)"]


def test_the_repair_text_says_these_are_about_the_answer(tmp_path):
    text = properties.render(["american_put is 0.0"])
    assert "american_put is 0.0" in text
    assert "rather than adjusting the output to satisfy the check" in text


def test_the_instruction_asks_for_properties_of_the_task_not_the_code():
    assert "must hold for ANY correct answer" in properties.PROPERTIES_INSTRUCTION
    assert "It does not solve the task" in properties.PROPERTIES_INSTRUCTION
