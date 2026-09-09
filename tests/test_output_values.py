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
from agent.validate import validate_outputs  # noqa: E402

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

def test_nan_in_a_numeric_column_is_an_error(tmp_path):
    """NaN almost always means a computation failed quietly rather than a deliberate answer."""
    ctx = _csv_ctx(tmp_path, "price,qty\n1.5,10\n,20\n")
    assert "nan_values" in _codes(validate_outputs(ctx, tmp_path))


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
