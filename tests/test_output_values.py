# SPDX-License-Identifier: MIT
"""Tests for the value and structure checks applied to generated output.

These catch output that is the right shape of file but could not be a correct answer — a negative
price, a NaN, no rows at all. They are deliberately conservative, so the tests below check both
that real problems are caught AND that legitimate values are left alone. A false alarm is worse
than a miss here: it sends the repair loop spending attempts on a file that was never broken.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from agent.task_context import parse_task  # noqa: E402
from agent.validate import ERROR, WARNING, validate_outputs  # noqa: E402

KIT = pathlib.Path(__file__).resolve().parents[2] / "track1-coding-public"
UNITS = KIT / "units"
pytestmark = pytest.mark.skipif(not UNITS.is_dir(), reason="practice kit not found beside repo")

pd = pytest.importorskip("pandas")


def _csv_ctx(tmp_path, rows: str, filename="trades.csv"):
    """A context whose single deliverable is the CSV we just wrote."""
    ctx = parse_task(UNITS / "t1-sma-crossover-spy", probe_filesystem=False)
    ctx.outputs = [o for o in ctx.outputs if o.filename == filename]
    (tmp_path / filename).write_text(rows)
    return ctx


def _codes(rep):
    return {f.code for f in rep.findings}


# --- structure -------------------------------------------------------------------------------

def test_header_only_file_is_an_error(tmp_path):
    """A file with columns and no rows passes a parse check but cannot be an answer."""
    ctx = _csv_ctx(tmp_path, "price,qty\n")
    assert "no_rows" in _codes(validate_outputs(ctx, tmp_path))


def test_rows_present_is_not_flagged(tmp_path):
    ctx = _csv_ctx(tmp_path, "price,qty\n1.5,10\n2.5,20\n")
    assert "no_rows" not in _codes(validate_outputs(ctx, tmp_path))


# --- NaN and infinity ------------------------------------------------------------------------

def _levels(rep, code):
    return {f.level for f in rep.findings if f.code == code}


# NaN is graded by its shape, and that is not a refinement for its own sake. In the September sweep
# five units raised `nan_values` and THREE of them passed grading anyway, each having spent all
# three attempts rewriting output the grader then accepted. A check that fires on correct answers
# turns a pass into a contract error, so the only blanks that now error are the ones no convention
# excuses.

def _strict_ctx(tmp_path, rows, filename="trades.csv"):
    """A context whose instruction says nothing about blanks, so the pattern rules decide alone.

    Set explicitly because `t1-sma-crossover-spy` -- the fixture unit -- really does permit them
    ("`pnl` is only filled for SELL rows (leave empty for BUY rows)"), which would otherwise make
    these tests pass for the wrong reason.
    """
    ctx = _csv_ctx(tmp_path, rows, filename)
    ctx.instruction = "Compute the table and write it out."
    return ctx


def test_a_wholly_blank_column_is_an_error(tmp_path):
    """Almost never intended: a join that matched nothing, or a column never assigned."""
    ctx = _strict_ctx(tmp_path, "price,qty\n,10\n,20\n")
    rep = validate_outputs(ctx, tmp_path)
    assert _levels(rep, "nan_values") == {ERROR}
    assert "entirely blank" in rep.repair_text()


# Severity is decided by whether the TASK permits blanks, and by nothing else. Across every
# measured unit that produced NaN that flag separates the groups perfectly: three units with
# scattered gaps whose instructions permit nulls all PASSED grading, while three whose instructions
# do not all failed. The shapes are the same in both groups, so grading on shape invents a
# distinction the grader does not make -- and it cost a real catch, cir-bond-pricing leaving half
# of `market_yield` blank. The shape survives only in the message, as a pointer to a line of code.

@pytest.mark.parametrize("rows", [
    "price,qty\n,10\n,20\n",                        # the whole column
    "price,qty\n,10\n,20\n3.5,30\n4.5,40\n",        # a leading run
    "price,qty\n1.5,10\n,20\n3.5,30\n",             # scattered
])
def test_shape_does_not_change_the_severity(rows, tmp_path):
    assert _levels(validate_outputs(_strict_ctx(tmp_path, rows), tmp_path),
                   "nan_values") == {ERROR}


@pytest.mark.parametrize("rows", [
    "price,qty\n,10\n,20\n",
    "price,qty\n,10\n,20\n3.5,30\n4.5,40\n",
    "price,qty\n1.5,10\n,20\n3.5,30\n",
])
def test_a_task_that_permits_blanks_only_ever_warns(rows, tmp_path):
    ctx = _strict_ctx(tmp_path, rows)
    ctx.instruction = "Leave the cell blank where no value applies."
    assert _levels(validate_outputs(ctx, tmp_path), "nan_values") == {WARNING}


def test_the_message_names_a_likely_cause(tmp_path):
    """"column x has 96 NaN" gives the repair loop nothing to act on; a cause points at a line."""
    ctx = _strict_ctx(tmp_path, "price,qty\n,10\n,20\n3.5,30\n4.5,40\n")
    text = str(validate_outputs(ctx, tmp_path).findings)
    assert "min_periods" in text or "warm-up" in text


def test_a_task_that_requires_blanks_does_not_get_an_error(tmp_path):
    ctx = _csv_ctx(tmp_path, "price,qty\n,10\n,20\n")
    ctx.instruction = ("If history length is below the minimum, all process parameters "
                       "must be null.")
    rep = validate_outputs(ctx, tmp_path)
    assert _levels(rep, "nan_values") == {WARNING}
    assert "do permit blanks" in str(rep.findings)


def test_permission_is_read_from_the_tasks_own_words():
    from agent.validate import nulls_permitted
    assert nulls_permitted("Non-applicable methods should return NaN.")
    assert nulls_permitted("all process parameters must be null")
    assert nulls_permitted("Leave the cell blank when no trade occurred.")
    # Measured on the roster: 7 of 87 instructions permit blanks somewhere, including this phrasing
    # from the fixture unit, which is why the tests above set the instruction themselves.
    assert nulls_permitted("`pnl` is only filled for SELL rows (leave empty for BUY rows)")
    assert not nulls_permitted("Compute the rolling mean over 60 days.")


def test_infinity_is_an_error(tmp_path):
    ctx = _csv_ctx(tmp_path, "price,qty\n1.5,10\ninf,20\n")
    assert "inf_values" in _codes(validate_outputs(ctx, tmp_path))


def test_nan_in_json_is_an_error(tmp_path):
    ctx = parse_task(UNITS / "t1-zero-coupon-bootstrapping", probe_filesystem=False)
    (tmp_path / "results.json").write_text('{"zero_rates": {"1": NaN}}')
    assert "nan_values" in _codes(validate_outputs(ctx, tmp_path))


# --- value ranges ----------------------------------------------------------------------------

def test_negative_price_is_an_error(tmp_path):
    ctx = _csv_ctx(tmp_path, "price,qty\n1.5,10\n-4.0,20\n")
    rep = validate_outputs(ctx, tmp_path)
    assert "value_out_of_range" in _codes(rep)
    assert not rep.ok


def test_negative_pnl_is_left_alone(tmp_path):
    """P&L legitimately goes negative, so the price rule must not match it."""
    ctx = _csv_ctx(tmp_path, "pnl,qty\n-40.0,10\n12.0,20\n")
    assert "value_out_of_range" not in _codes(validate_outputs(ctx, tmp_path))


def test_probability_above_one_is_an_error(tmp_path):
    ctx = _csv_ctx(tmp_path, "probability,qty\n0.4,10\n1.7,20\n")
    assert "value_out_of_range" in _codes(validate_outputs(ctx, tmp_path))


def test_plausible_volatility_passes(tmp_path):
    ctx = _csv_ctx(tmp_path, "sigma,qty\n0.2,10\n0.35,20\n")
    assert "value_out_of_range" not in _codes(validate_outputs(ctx, tmp_path))


def test_implausible_volatility_only_warns(tmp_path):
    """A wild vol is suspicious but not impossible in a synthetic task, so it must not be fatal."""
    ctx = _csv_ctx(tmp_path, "sigma,qty\n0.2,10\n50.0,20\n")
    rep = validate_outputs(ctx, tmp_path)
    assert "value_out_of_range" in _codes(rep)
    assert rep.ok, "an implausible-but-possible value must not fail the task"


def test_a_df_column_is_not_read_as_a_discount_factor(tmp_path):
    """Measured on a PASSING unit: `markov_test.csv` has a `df` column holding chi-square degrees
    of freedom (24), and a 0-1 discount-factor bound failed a correct answer. The abbreviation is
    ambiguous and its commonest expansions -- degrees of freedom, and a pandas DataFrame -- are
    both unbounded."""
    ctx = _strict_ctx(tmp_path, "rating,df,p_value\nAAA,24,0.4\nAA,18,0.7\n")
    assert "value_out_of_range" not in _codes(validate_outputs(ctx, tmp_path))


def test_a_named_discount_factor_is_still_bounded(tmp_path):
    ctx = _strict_ctx(tmp_path, "maturity,discount_factor\n1,0.97\n2,1.4\n")
    assert "value_out_of_range" in _codes(validate_outputs(ctx, tmp_path))


# --- output that did not actually compute anything -----------------------------------------------
#
# The failure this was written for: a unit produced a 534-line PDE solver that ran cleanly, wrote
# all seven files, and satisfied every check we had -- while reporting an american put, american
# call, european put and european call, with and without dividends, as EXACTLY 0.0, all eight. The
# script reversed its time axis for readability and then read the answer off the wrong end, getting
# the payoff at expiry instead of the value today; the options were at the money, so every payoff
# was zero. Files parsed, columns right, values in range, no NaN -- and a price of 0.0 satisfies
# "prices are non-negative" perfectly. The only signal was that every number was the same number.

def test_a_file_where_every_number_is_the_same_is_an_error(tmp_path):
    ctx = _strict_ctx(tmp_path, "grid,price\nfine,0.0\ncoarse,0.0\nrichardson,0.0\n")
    rep = validate_outputs(ctx, tmp_path)
    assert "degenerate_values" in _codes(rep)
    assert "did not happen" in rep.repair_text()


def test_the_same_applies_to_json(tmp_path):
    ctx = parse_task(UNITS / "t1-zero-coupon-bootstrapping", probe_filesystem=False)
    (tmp_path / "results.json").write_text(json.dumps(
        {"american_put": 0.0, "american_call": 0.0, "european_put": 0.0, "european_call": 0.0}))
    assert "degenerate_values" in _codes(validate_outputs(ctx, tmp_path))


def test_a_repeated_non_zero_value_counts_too(tmp_path):
    """It is not about zero. A file of one repeated number is a computation that did nothing."""
    ctx = _strict_ctx(tmp_path, "a,b\n7.5,7.5\n7.5,7.5\n")
    assert "degenerate_values" in _codes(validate_outputs(ctx, tmp_path))


def test_one_constant_column_beside_varying_ones_is_left_alone(tmp_path):
    """Deliberately narrow. A trade count, a violation count, or the weights of a portfolio meant
    to net to nothing are all legitimately constant -- it is the WHOLE file being constant that
    means nothing was computed."""
    ctx = _strict_ctx(tmp_path, "price,violations\n101.5,0\n102.25,0\n99.75,0\n")
    assert "degenerate_values" not in _codes(validate_outputs(ctx, tmp_path))


def test_too_few_numbers_to_judge(tmp_path):
    """Two identical values is a coincidence, not evidence."""
    ctx = _strict_ctx(tmp_path, "a,b\n1.0,1.0\n")
    assert "degenerate_values" not in _codes(validate_outputs(ctx, tmp_path))


def test_ordinary_varying_output_is_not_flagged(tmp_path):
    ctx = _strict_ctx(tmp_path, "tau,zcb_price\n0.25,0.991\n0.5,0.983\n1.0,0.967\n")
    assert "degenerate_values" not in _codes(validate_outputs(ctx, tmp_path))
