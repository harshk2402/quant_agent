# SPDX-License-Identifier: MIT
"""Tests for the feedback message sent to the model after a failed attempt.

Two kinds of failure reach it: a traceback, and output that ran cleanly but does not satisfy the
contract. The second has to be stated explicitly, because nothing crashed and the model has no
other way to know anything is wrong.
"""
from __future__ import annotations

import json
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
