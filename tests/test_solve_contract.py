# SPDX-License-Identifier: MIT
"""Contract tests for the `solve` verb -- the failures that zero a unit without touching finance.

Run:  ../.venv/bin/python -m pytest tests/ -q
"""
from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from agent.solve import main, solve  # noqa: E402
from agent.task_context import parse_task  # noqa: E402

KIT = pathlib.Path(__file__).resolve().parents[2] / "track1-coding-public"
UNITS = KIT / "units"
pytestmark = pytest.mark.skipif(not UNITS.is_dir(), reason="practice kit not found beside repo")


def _units(n=None):
    us = sorted(d for d in UNITS.iterdir() if (d / "instruction.md").is_file())
    return us[:n] if n else us


def test_verb_is_accepted_as_positional(tmp_path):
    """The harness passes `solve` as the first arg. Rejecting it exits non-zero on EVERY unit."""
    assert main(["solve", "--task-dir", str(_units(1)[0]), "--out", str(tmp_path)]) == 0


def test_unknown_argument_still_exits_zero(tmp_path):
    """argparse exits 2 on a bad arg; the contract wants 0. Never forfeit a unit over an arg."""
    assert main(["solve", "--task-dir", str(_units(1)[0]), "--out", str(tmp_path),
                 "--not-a-real-flag"]) == 0


def test_missing_task_dir_does_not_crash(tmp_path):
    """A wrong answer is scored; an exception is not."""
    assert solve(str(tmp_path / "nope"), str(tmp_path)) == 0


def test_writes_every_declared_deliverable(tmp_path):
    unit = UNITS / "t1-zero-coupon-bootstrapping"
    assert solve(str(unit), str(tmp_path)) == 0
    for spec in parse_task(unit, probe_filesystem=False).outputs:
        assert (tmp_path / spec.filename).is_file(), f"missing {spec.filename}"


def test_never_writes_reward_json(tmp_path):
    """checks/test.sh writes reward.json. Writing it ourselves corrupts the reward signal."""
    solve(str(UNITS / "t1-zero-coupon-bootstrapping"), str(tmp_path))
    assert not (tmp_path / "reward.json").exists()


@pytest.mark.parametrize("unit", _units(), ids=lambda d: d.name)
def test_every_public_unit_exits_zero_and_writes_something(unit, tmp_path):
    """The floor conformance.sh checks, run in-process across all 87 units."""
    assert solve(str(unit), str(tmp_path)) == 0
    expected = [s.filename for s in parse_task(unit, probe_filesystem=False).outputs]
    assert expected, f"{unit.name}: parser found no deliverable"
    assert any((tmp_path / f).is_file() for f in expected)


def test_no_canary_guid_in_any_output(tmp_path):
    """Gate g2 scans every output text file for canary GUIDs -> instant zero if one leaks.

    66/87 units carry one in instruction.md, so this guards the whole corpus, not one unit.
    """
    import re
    guid = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
    canaries = set()
    for unit in _units():
        canaries.update(guid.findall((unit / "instruction.md").read_text(errors="replace")))
        out = tmp_path / unit.name
        out.mkdir()
        solve(str(unit), str(out))
    assert canaries, "no canaries found in the kit -- test would be vacuous"
    for f in tmp_path.rglob("*"):
        if f.is_file() and f.suffix in {".py", ".json", ".txt", ".md", ".csv", ".log"}:
            body = f.read_text(errors="replace")
            leaked = canaries & set(guid.findall(body))
            assert not leaked, f"CANARY LEAKED into {f.name}: {leaked}"
