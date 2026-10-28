# SPDX-License-Identifier: MIT
"""Tests for showing the finished answer to the model and asking whether it can be right.

Measured against saved failures, against which the two earlier layers were also run:

    hand-written invariants   caught 0 of 2
    generated property tests  caught 0 of 2, and raised one false alarm that cost a unit its
                              entire repair budget
    this critic               caught 2 of 3, no false alarm on the passing unit tested

Most of what is tested here is the distrust: the verdict gates everything, one `wrong` buys a
single generation, and the comparison that follows is blind.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from agent import critic  # noqa: E402


def _out(tmp_path, **files):
    d = tmp_path / "out"
    d.mkdir(exist_ok=True)
    for name, body in files.items():
        (d / name).write_text(body)
    return d


# --- reading the answer --------------------------------------------------------------------

def test_the_deliverables_are_shown(tmp_path):
    out = _out(tmp_path, **{"results.json": '{"a": 1}', "curve.csv": "t,v\n1,0.03\n"})
    body = critic.read_outputs(out)
    assert "results.json" in body and '{"a": 1}' in body
    assert "curve.csv" in body and "0.03" in body


def test_grader_files_are_never_shown(tmp_path):
    """They belong to the harness, not to us, and judging them would be nonsense."""
    out = _out(tmp_path, **{"results.json": "{}", "reward.json": '{"reward": 1.0}',
                            "pytest_report.json": "{}"})
    body = critic.read_outputs(out)
    assert "reward" not in body and "pytest_report" not in body


def test_a_binary_file_is_named_not_quoted(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "plot.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 500)
    body = critic.read_outputs(out)
    assert "plot.png" in body and "not text" in body


def test_long_files_are_truncated(tmp_path):
    out = _out(tmp_path, **{"big.csv": "x\n" + "1.0\n" * 50000})
    body = critic.read_outputs(out)
    assert len(body) < critic.MAX_TOTAL + 200
    assert "characters in total" in body


def test_an_empty_directory_yields_no_prompt(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    assert critic.build_critic_prompt([{"role": "user", "content": "task"}], out) is None


# --- the verdict ---------------------------------------------------------------------------

def test_wrong_with_problems_is_acted_on():
    wrong, problems = critic.parse_verdict(
        "PROBLEM: american_put should be above zero but is 0.0\n"
        "PROBLEM: gamma should be positive for a long option but is -0.4\n"
        "VERDICT: wrong")
    assert wrong and len(problems) == 2


def test_plausible_is_the_default_for_anything_unclear():
    for reply in ("VERDICT: plausible", "", "I think it looks fine", "VERDICT: maybe",
                  "PROBLEM: something -- without a verdict line"):
        assert critic.parse_verdict(reply) == (False, [])


def test_problems_are_ignored_when_the_verdict_is_plausible():
    """Measured: the model sometimes argues itself out of a complaint mid-reply -- "so this is not
    a problem. Wait -- rechecking". Only its conclusion is worth acting on."""
    wrong, problems = critic.parse_verdict(
        "PROBLEM: american_call should be 5 but is 4\nVERDICT: plausible")
    assert not wrong and problems == []


def test_a_wrong_verdict_with_no_problems_does_nothing():
    """Nothing to repair against, so there is nothing to spend a generation on -- and measured,
    a `wrong` with nothing supporting it is usually not a real objection."""
    assert critic.parse_verdict("VERDICT: wrong") == (False, [])


def test_a_flood_of_problems_is_capped():
    reply = ("\n".join(f"PROBLEM: v{i} should be 1 but is 0" for i in range(40))
             + "\nVERDICT: wrong")
    _, problems = critic.parse_verdict(reply)
    assert len(problems) == critic.MAX_PROBLEMS


# --- the blind comparison --------------------------------------------------------------------

def test_the_comparison_says_nothing_about_which_is_the_revision():
    """The critic is not a neutral judge of whether its own objection was met: told which output
    is the revision, it will favour the revision."""
    prompt = critic.build_compare_prompt([{"role": "user", "content": "task"}],
                                         "FIRST ANSWER", "SECOND ANSWER")
    text = prompt[-1]["content"]
    assert "FIRST ANSWER" in text and "SECOND ANSWER" in text
    for giveaway in ("revision", "repair", "original", "previous", "corrected", "problem"):
        assert giveaway not in text.lower()


def test_the_choice_is_read_from_the_answer_line():
    assert critic.parse_choice("ANSWER: A\nREASON: the boundary is right") == "A"
    assert critic.parse_choice("ANSWER: B\nREASON: ...") == "B"


def test_an_unclear_choice_is_no_choice():
    for reply in ("", "both look fine", "REASON: A is better", "ANSWER: C"):
        assert critic.parse_choice(reply) == ""


def test_the_repair_text_points_at_the_computation():
    text = critic.render(["american_put 0.0 -- cannot be worthless"])
    assert "american_put 0.0" in text
    assert "Do not adjust the values on the way out" in text


# --- the invariants, moved to where they can be tested rather than nodded at ----------------------

def test_the_checklist_appears_in_the_ask_not_just_the_context(tmp_path):
    """They were always in the critic's context, in the original task prompt -- but as advice to
    someone about to write code. With numbers in front of you the same sentence is a test."""
    out = _out(tmp_path, **{"results.json": '{"call_price": 5.2, "put_price": 3.1}'})
    prompt = critic.build_critic_prompt([{"role": "user", "content": "task"}], out,
                                        category="derivatives-pricing")
    ask = prompt[-1]["content"]
    assert "put-call parity" in ask.lower()
    assert "each one is a test you can run" in ask


def test_the_checklist_is_routed_by_category(tmp_path):
    out = _out(tmp_path, **{"results.json": "{}"})
    credit = critic.build_critic_prompt([{"role": "user", "content": "t"}], out,
                                        category="credit")[-1]["content"]
    assert "survival probability" in credit.lower()
    assert "put-call parity" not in credit.lower()


def test_the_task_is_declared_to_outrank_the_checklist(tmp_path):
    """Four times in measured runs a universal rule overrode what a task actually specified."""
    out = _out(tmp_path, **{"results.json": "{}"})
    ask = critic.build_critic_prompt([{"role": "user", "content": "t"}], out,
                                     category="risk-management")[-1]["content"]
    assert "the task wins" in ask


def test_no_category_means_no_checklist(tmp_path):
    out = _out(tmp_path, **{"results.json": "{}"})
    ask = critic.build_critic_prompt([{"role": "user", "content": "t"}], out)[-1]["content"]
    assert "each one is a test you can run" not in ask


# --- a verdict that contradicts its own reasoning ------------------------------------------------
#
# Measured live: the critic returned `VERDICT: wrong` and then five "problems" that each concluded
# the value was CORRECT -- "which is correct", "which matches", "is plausible". All five went into
# a repair turn as defects. It had committed to a verdict and then failed to support it.

def test_a_problem_must_state_the_contradiction():
    wrong, problems = critic.parse_verdict(
        "PROBLEM: var_99 should be 0.0131 but is 0.0044\nVERDICT: wrong")
    assert wrong and problems == ["var_99 should be 0.0131 but is 0.0044"]


def test_a_confirmation_cannot_be_read_as_a_problem():
    """The measured false alarm, verbatim in shape: five lines that each say the value is right."""
    reply = (
        "PROBLEM: var_99_parametric -- 0.01310 / 0.005635 = 2.326, which is correct.\n"
        "PROBLEM: annualized_vol -- 0.005635 * sqrt(252) = 0.0894, which matches.\n"
        "VERDICT: wrong")
    assert critic.parse_verdict(reply) == (False, [])


def test_the_verdict_is_asked_for_last():
    """It stated a conclusion and then reasoned; the reasoning disagreed and the conclusion won."""
    assert "on the LAST line" in critic.CRITIC_INSTRUCTION
    assert critic.CRITIC_INSTRUCTION.index("List any problems FIRST") < \
           critic.CRITIC_INSTRUCTION.index("VERDICT: plausible | wrong")


def test_correct_findings_are_asked_to_be_omitted():
    assert "write nothing about it" in critic.CRITIC_INSTRUCTION


def test_a_problem_whose_two_numbers_match_is_not_a_problem():
    """Measured, with the required wording already in place: `annualized_vol should be
    0.266599766168436 but is 0.266599766168436`. The template asks for a contradiction and it
    filled both halves with the same value."""
    reply = ("PROBLEM: total_return should be 0.1772 but is 1.5898\n"
             "PROBLEM: annualized_vol should be 0.266599766168436 but is 0.266599766168436\n"
             "VERDICT: wrong")
    wrong, problems = critic.parse_verdict(reply)
    assert wrong
    assert len(problems) == 1 and "total_return" in problems[0]


def test_all_problems_matching_means_no_objection():
    reply = ("PROBLEM: a should be 1.5 but is 1.5\nPROBLEM: b should be 2 but is 2.0\n"
             "VERDICT: wrong")
    assert critic.parse_verdict(reply) == (False, [])


def test_a_non_numeric_problem_is_kept():
    """The test only applies where there are two numbers to compare."""
    reply = ("PROBLEM: the exercise boundary should be monotonically decreasing but is flat at 300\n"
             "VERDICT: wrong")
    wrong, problems = critic.parse_verdict(reply)
    assert wrong and len(problems) == 1
