# SPDX-License-Identifier: MIT
"""Tests for the feedback message sent to the model after a failed attempt.

Two kinds of failure reach it: a traceback, and output that ran cleanly but does not satisfy the
contract. The second has to be stated explicitly, because nothing crashed and the model has no
other way to know anything is wrong.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from agent.prompt import build_repair_message, strip_canary  # noqa: E402
from agent.task_context import parse_task  # noqa: E402
from agent.validate import validate_outputs  # noqa: E402

KIT = pathlib.Path(__file__).resolve().parents[2] / "track1-coding-public"
UNITS = KIT / "units"
pytestmark = pytest.mark.skipif(not UNITS.is_dir(), reason="practice kit not found beside repo")


def _csv_ctx(tmp_path, rows: str, filename="trades.csv"):
    """A context whose single deliverable is the CSV we just wrote."""
    ctx = parse_task(UNITS / "t1-sma-crossover-spy", probe_filesystem=False)
    ctx.outputs = [o for o in ctx.outputs if o.filename == filename]
    (tmp_path / filename).write_text(rows)
    return ctx


def _codes(rep):
    return {f.code for f in rep.findings}


# --- the repair message ----------------------------------------------------------------------

def test_repair_message_carries_a_traceback():
    msg = build_repair_message(exec_failure="ZeroDivisionError: division by zero")
    assert "ZeroDivisionError" in msg
    assert "COMPLETE corrected script" in msg


def test_repair_message_carries_contract_errors():
    """A script that ran cleanly gives the model no reason to suspect a problem, so the contract
    failures have to be stated explicitly."""
    ctx = parse_task(UNITS / "t1-zero-coupon-bootstrapping", probe_filesystem=False)
    rep = validate_outputs(ctx, pathlib.Path("/nonexistent"))
    msg = build_repair_message(contract_errors=rep.repair_text())
    assert "results.json" in msg and "missing_file" in msg


def test_repair_message_never_leaks_a_canary():
    """Repair feeds prior output back to the model; a canary must not ride along."""
    guid = "9ccd8a7c-a5cc-4cf2-8874-0633603f95c4"
    assert guid not in strip_canary(f"Traceback ... canary {guid} ...")


# --- pointing repair at the function that failed -------------------------------------------------
#
# A traceback is the least ambiguous signal we get: exact file, line and function. Measured, a run
# was handed `IndexError ... line 148, in crank_nicolson_psor` and failed to fix it on two further
# attempts -- because we ask for the ENTIRE corrected script, so each repair regenerates five
# hundred lines and re-rolls every decision while fixing the one that broke.

SCRIPT = '''import numpy as np

def setup(n):
    return np.zeros(n)

def crank_nicolson_psor(Z, B, C, D):
    for i in range(len(Z)):
        Z[i] = (D[i] - C[i] * Z[i + 1]) / B[i]
    return Z

def main():
    print(crank_nicolson_psor(setup(3), setup(3), setup(3), setup(3)))
'''

TRACEBACK = '''Traceback (most recent call last):
  File "/tmp/qa-exec-x/solution.py", line 13, in <module>
    main()
  File "/tmp/qa-exec-x/solution.py", line 7, in crank_nicolson_psor
    Z[i] = (D[i] - C[i] * Z[i + 1]) / B[i]
IndexError: index 3 is out of bounds for axis 0 with size 3
'''


def test_the_failing_function_is_quoted_back():
    from agent.prompt import locate_failure
    text = locate_failure(TRACEBACK, SCRIPT)
    assert "crank_nicolson_psor" in text
    assert "Z[i] = (D[i] - C[i] * Z[i + 1]) / B[i]" in text
    assert "def setup" not in text, "only the offending function should be quoted"
    assert "reproduce the rest of the script" in text.lower()


def test_the_innermost_frame_is_the_one_blamed():
    """`main` appears in the traceback too; the culprit is the deepest frame in our own file."""
    from agent.prompt import locate_failure
    assert "def main" not in locate_failure(TRACEBACK, SCRIPT)


def test_a_traceback_with_no_usable_frame_yields_nothing():
    from agent.prompt import locate_failure
    assert locate_failure("MemoryError\n", SCRIPT) == ""
    assert locate_failure(TRACEBACK, "") == ""


def test_unparsable_code_is_handled():
    from agent.prompt import locate_failure
    assert locate_failure(TRACEBACK, "def broken(:\n  pass") == ""


def test_a_missing_function_still_names_the_line():
    from agent.prompt import locate_failure
    text = locate_failure(TRACEBACK, "x = 1\n")
    assert "line 7" in text and "crank_nicolson_psor" in text


def test_the_repair_message_carries_the_failing_function():
    from agent.prompt import build_repair_message
    msg = build_repair_message(exec_failure="boom", failing_function="FUNC QUOTED HERE")
    assert "FUNC QUOTED HERE" in msg


# --- the most expensive failure, and the one we said least about ---------------------------------
#
# Measured: a unit lost two consecutive attempts to 120-second timeouts. A timeout produces no
# traceback, so `locate_failure` has nothing to blame and the repair message degenerated to "it
# failed when executed" plus a timeout notice. The model regenerated something equally slow twice.

def test_a_timeout_gets_advice_instead_of_an_empty_traceback():
    from agent.prompt import build_repair_message
    msg = build_repair_message(exec_failure="timed out after 120s", timed_out=True,
                               time_limit=120.0)
    assert "exceeded 120 seconds" in msg
    assert "numpy array operations" in msg
    assert "solve_banded" in msg


def test_the_advice_says_to_speed_up_not_to_shrink_the_problem():
    """A fast wrong answer scores the same as a slow one -- the grid sizes are the task."""
    from agent.prompt import build_repair_message
    msg = build_repair_message(timed_out=True, time_limit=120.0)
    assert "make the work faster, do not make it smaller" in msg


def test_a_timeout_does_not_also_quote_a_function():
    """There is no meaningful frame to blame, and quoting one would point at the wrong place."""
    from agent.prompt import build_repair_message
    msg = build_repair_message(exec_failure="timed out", failing_function="SOME FUNCTION",
                               timed_out=True, time_limit=60.0)
    assert "SOME FUNCTION" not in msg


def test_an_ordinary_crash_still_gets_the_traceback_and_the_function():
    from agent.prompt import build_repair_message
    msg = build_repair_message(exec_failure="IndexError: boom", failing_function="SOME FUNCTION")
    assert "IndexError: boom" in msg and "SOME FUNCTION" in msg
    assert "exceeded" not in msg


# --- abandoning a repair chain that will not converge --------------------------------------------
#
# Measured three separate times: IndexError on three consecutive attempts, TIMEOUT on two,
# TypeError on three. Each attempt was handed a precise traceback and produced the identical
# failure again, because a repair carries the broken script forward and the next attempt is a small
# edit of the same wrong foundation. Independent generations do not share that fate -- five runs of
# one unit gave five different failures -- so a fresh sample is a genuinely different roll.

def test_the_same_exception_twice_is_stuck():
    from agent.solve import _stuck
    assert _stuck(["TypeError: '<' not supported between int and Timestamp",
                   "TypeError: '<' not supported between instances of 'int' and 'Timestamp'"])


def test_the_kind_is_compared_not_the_message():
    """Line numbers and values move between attempts even when the bug is identical."""
    from agent.solve import _stuck
    assert _stuck(["IndexError: index 299 is out of bounds for axis 0 with size 299",
                   "IndexError: index 41 is out of bounds for axis 0 with size 41"])


def test_two_different_failures_are_progress():
    """The chain is moving: it fixed one thing and broke another, which repair can work with."""
    from agent.solve import _stuck
    assert not _stuck(["IndexError: out of bounds", "TypeError: not subscriptable"])


def test_timeouts_match_each_other():
    from agent.solve import _stuck
    assert _stuck(["TIMEOUT after 120s", "TIMEOUT after 120s"])


def test_contract_errors_are_not_treated_as_stuck():
    """A missing file is a different kind of problem, and repairing against it does converge."""
    from agent.solve import _stuck
    assert not _stuck(["contract: missing_file", "contract: missing_file"])


def test_one_attempt_is_never_stuck():
    from agent.solve import _stuck
    assert not _stuck([]) and not _stuck(["IndexError: boom"])


def test_the_restart_shows_no_previous_code():
    """The point is to break the lineage: inheriting the broken script is what stops it converging."""
    from agent.prompt import build_restart_message
    msg = build_restart_message("IndexError: index 299 is out of bounds")
    assert "IndexError: index 299" in msg
    assert "Start again from scratch" in msg
    assert "different approach" in msg
    assert "repair" in msg.lower()


def test_the_restart_prefers_something_that_runs():
    from agent.prompt import build_restart_message
    msg = build_restart_message("TIMEOUT after 120s")
    assert "approximate answer is worth far more" in msg


# --- the two kinds of missing deliverable --------------------------------------------------------
#
# `missing_file` is our commonest contract error, and today it says the same thing whichever case
# it is: "required deliverable was not written". The two need opposite fixes.

SCRIPT_WITH_WRITE = '''import json, pathlib
out = pathlib.Path("/app/output")
rows = compute()
with open(out / "results.json", "w") as f:
    json.dump(rows, f)
print("done")
'''


def test_a_write_that_exists_is_quoted():
    from agent.prompt import describe_missing
    text = describe_missing(["results.json"], SCRIPT_WITH_WRITE, "/app/output")
    assert "DOES reference it" in text
    assert 'json.dump(rows, f)' in text
    assert "execution never reached this point" in text


def test_the_output_directory_is_named():
    """Many units' instructions name a path that is not where the harness binds the output, so
    "written somewhere else" looks identical to "never written"."""
    from agent.prompt import describe_missing
    text = describe_missing(["results.json"], SCRIPT_WITH_WRITE, "/app/output")
    assert "/app/output" in text


def test_a_file_the_script_never_mentions_is_called_out_as_forgotten():
    from agent.prompt import describe_missing
    text = describe_missing(["summary.json"], SCRIPT_WITH_WRITE, "/app/output")
    assert "does not appear anywhere" in text
    assert "Add the code that produces it" in text
    assert "execution never reached" not in text


def test_both_kinds_in_one_message():
    from agent.prompt import describe_missing
    text = describe_missing(["results.json", "summary.json"], SCRIPT_WITH_WRITE)
    assert "DOES reference it" in text and "does not appear anywhere" in text


def test_nothing_missing_says_nothing():
    from agent.prompt import describe_missing
    assert describe_missing([], SCRIPT_WITH_WRITE) == ""
    assert describe_missing(["results.json"], "") == ""


def test_the_repair_message_carries_the_missing_file_detail():
    from agent.prompt import build_repair_message
    msg = build_repair_message(contract_errors="missing_file", missing_files="MISSING DETAIL")
    assert "MISSING DETAIL" in msg
