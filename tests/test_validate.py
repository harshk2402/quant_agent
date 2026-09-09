# SPDX-License-Identifier: MIT
"""Tests for the output-contract validator (Step 0.4).

The canary tests matter most: that is the only check here that repairs rather than reports, and a
marker left in an output file fails the task outright regardless of whether the answer was right.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from agent.task_context import parse_task  # noqa: E402
from agent.validate import (  # noqa: E402
    ERROR, known_canaries, scrub_canaries, validate_outputs,
)

KIT = pathlib.Path(__file__).resolve().parents[2] / "track1-coding-public"
UNITS = KIT / "units"
UNIT = UNITS / "t1-zero-coupon-bootstrapping"
pytestmark = pytest.mark.skipif(not UNITS.is_dir(), reason="practice kit not found beside repo")

CANARY = "9ccd8a7c-a5cc-4cf2-8874-0633603f95c4"


def _ctx():
    return parse_task(UNIT, probe_filesystem=False)


def _codes(report):
    return {f.code for f in report.findings}


# --- the contract ----------------------------------------------------------------------------

def test_missing_file_is_an_error(tmp_path):
    rep = validate_outputs(_ctx(), tmp_path)
    assert "missing_file" in _codes(rep)
    assert not rep.ok


def test_empty_file_is_an_error(tmp_path):
    (tmp_path / "results.json").write_text("")
    assert "empty_file" in _codes(validate_outputs(_ctx(), tmp_path))


def test_malformed_json_is_caught(tmp_path):
    (tmp_path / "results.json").write_text('{"zero_rates": ')      # truncated
    rep = validate_outputs(_ctx(), tmp_path)
    assert "unparseable" in _codes(rep)


def test_well_formed_output_passes(tmp_path):
    (tmp_path / "results.json").write_text(json.dumps({"zero_rates": {"1": 0.04}}))
    rep = validate_outputs(_ctx(), tmp_path)
    assert rep.ok, rep.repair_text()


def test_writing_the_graders_own_file_is_an_error(tmp_path):
    (tmp_path / "results.json").write_text("{}")
    (tmp_path / "reward.json").write_text('{"reward": 1.0}')
    rep = validate_outputs(_ctx(), tmp_path)
    assert "wrote_grader_file" in _codes(rep)


def test_findings_render_for_the_repair_loop(tmp_path):
    text = validate_outputs(_ctx(), tmp_path).repair_text()
    assert "missing_file" in text and "results.json" in text


# --- canaries --------------------------------------------------------------------------------

def test_known_canaries_reads_the_units_own_marker():
    assert CANARY in known_canaries(UNIT)


def test_leaked_canary_is_detected(tmp_path):
    (tmp_path / "results.json").write_text(json.dumps({"note": f"from {CANARY}"}))
    rep = validate_outputs(_ctx(), tmp_path)
    assert "canary_suspect" in _codes(rep)
    assert any(f.level == ERROR for f in rep.findings)


def test_leaked_canary_is_removed_and_data_survives(tmp_path):
    payload = {"zero_rates": {"1": 0.044}, "note": f"generated for {CANARY}"}
    (tmp_path / "results.json").write_text(json.dumps(payload))

    cleaned = scrub_canaries(tmp_path, known_canaries(UNIT))

    assert cleaned == ["results.json"]
    body = (tmp_path / "results.json").read_text()
    assert CANARY not in body
    # Still valid JSON, and the actual answer is untouched.
    assert json.loads(body)["zero_rates"]["1"] == 0.044
    assert validate_outputs(_ctx(), tmp_path).ok


def test_scrub_leaves_unrelated_guids_alone(tmp_path):
    """Only THIS task's markers are removed; a GUID the task legitimately asked for survives."""
    other = "11111111-2222-3333-4444-555555555555"
    (tmp_path / "results.json").write_text(json.dumps({"id": other}))
    assert scrub_canaries(tmp_path, known_canaries(UNIT)) == []
    assert other in (tmp_path / "results.json").read_text()


def test_scrub_is_a_no_op_when_there_is_nothing_to_clean(tmp_path):
    (tmp_path / "results.json").write_text('{"zero_rates": {"1": 0.044}}')
    before = (tmp_path / "results.json").read_text()
    assert scrub_canaries(tmp_path, known_canaries(UNIT)) == []
    assert (tmp_path / "results.json").read_text() == before


# --- soft signals ----------------------------------------------------------------------------

def test_missing_columns_warn_but_do_not_fail(tmp_path):
    """Column extraction covers only ~56% of tabular deliverables, so a mismatch is a hint for
    the repair loop -- never grounds to reject a file that may well be correct."""
    unit = UNITS / "t1-sma-crossover-spy"
    ctx = parse_task(unit, probe_filesystem=False)
    for spec in ctx.outputs:
        p = tmp_path / spec.filename
        p.write_text("wrong_col_a,wrong_col_b\n1,2\n" if spec.fmt == "csv" else "{}")
    rep = validate_outputs(ctx, tmp_path)
    assert "columns_missing" in _codes(rep)
    assert rep.ok, "a column mismatch must not be an error"
