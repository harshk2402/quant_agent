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


# --- A/B switches ------------------------------------------------------------------------------
#
# Both prompt additions are hypotheses. On the representative set, four units that passed in
# September stopped passing, and three of them made a single model call with no stitching involved
# -- so the prompt is the only thing that changed for them. A block that cannot be turned off
# cannot be measured.

def _rep_ctx():
    return parse_task(UNITS / "t1-cir-bond-pricing", probe_filesystem=False)


def test_guidance_can_be_switched_off(monkeypatch):
    from agent.prompt import build_prompt
    ctx = _rep_ctx()
    assert "Financial invariants" in build_prompt(ctx, "/app/output")[1]["content"]
    monkeypatch.setenv("QFBENCH_NO_INVARIANT_GUIDANCE", "1")
    assert "Financial invariants" not in build_prompt(ctx, "/app/output")[1]["content"]


def test_input_previews_can_be_switched_off(monkeypatch):
    from agent.prompt import build_prompt
    ctx = parse_task(UNITS / "t1-cir-bond-pricing")
    monkeypatch.setenv("QFBENCH_NO_INPUT_PREVIEW", "1")
    assert "What each input" not in build_prompt(ctx, "/app/output")[1]["content"]


def test_the_task_text_is_declared_to_outrank_the_guidance():
    """A precedence rule belongs at the top of the block, not as a footnote: the guidance is
    phrased as rules, and a task that states a different convention has to win."""
    from agent.prompt import build_prompt
    body = build_prompt(_rep_ctx(), "/app/output")[1]["content"]
    block = body.split("## Financial invariants")[1].split("##")[0]
    assert "takes precedence" in block


# --- keeping the best output ---------------------------------------------------------------------
#
# Every attempt writes into the same directory, so without this the loop has no memory of having
# been better earlier: an attempt that crashes having written nothing replaces one that wrote six
# of seven files. There is no upside to that trade.

def _ctx_with_two_outputs(tmp_path):
    ctx = parse_task(UNITS / "t1-zero-coupon-bootstrapping", probe_filesystem=False)
    return ctx


def test_output_score_prefers_fewer_errors_then_more_files(tmp_path):
    from agent.solve import _output_score
    ctx = _ctx_with_two_outputs(tmp_path)

    (tmp_path / "results.json").write_text('{"zero_rates": {"1": 0.03}}')
    good = _output_score(ctx, tmp_path)

    (tmp_path / "results.json").unlink()
    missing = _output_score(ctx, tmp_path)

    assert good > missing          # a written deliverable beats a missing one


def test_restore_puts_the_better_output_back(tmp_path):
    from agent.solve import _keep_best, _restore
    out = tmp_path / "out"
    out.mkdir()
    (out / "results.json").write_text('{"a": 1}')
    (out / "extra.csv").write_text("x\n1\n")

    snap = _keep_best(out, None)
    assert not str(snap).startswith(str(out)), "the snapshot must not live inside the output tree"

    # a later attempt wrecks it
    for f in out.iterdir():
        f.unlink()
    (out / "junk.txt").write_text("nothing useful")

    _restore(out, snap)
    assert (out / "results.json").read_text() == '{"a": 1}'
    assert (out / "extra.csv").exists()
    assert not (out / "junk.txt").exists(), "restoring must clear what the bad attempt left"


def test_restore_empties_the_directory_without_removing_it(tmp_path):
    """The harness binds this directory as both /output and /app/output. Deleting a mount point is
    not the same as emptying it."""
    from agent.solve import _keep_best, _restore
    out = tmp_path / "out"
    out.mkdir()
    (out / "a.json").write_text("{}")
    snap = _keep_best(out, None)
    inode = out.stat().st_ino

    (out / "sub").mkdir()
    (out / "sub" / "deep.txt").write_text("x")
    _restore(out, snap)

    assert out.is_dir() and out.stat().st_ino == inode
    assert not (out / "sub").exists()
    assert (out / "a.json").exists()


def test_a_later_snapshot_replaces_the_earlier_one(tmp_path):
    from agent.solve import _keep_best
    out = tmp_path / "out"
    out.mkdir()
    (out / "a.json").write_text("first")
    first = _keep_best(out, None)
    (out / "a.json").write_text("second")
    second = _keep_best(out, first)
    assert not first.exists(), "the superseded snapshot should not be left behind"
    assert (second / "a.json").read_text() == "second"


# --- deliverables before extras ------------------------------------------------------------------
#
# Several scripts computed every number correctly and then died in the plotting tail, with the
# results still only in memory. The rule can only move a failure later in the script, never
# earlier, which is why it needs no measurement to justify.

def test_the_prompt_orders_data_files_before_plots():
    from agent.prompt import SYSTEM
    assert "IMMEDIATELY write every required data file" in SYSTEM
    assert "Only after all of those are on disk, produce plots" in SYSTEM
    assert "try/except so a plotting failure can never stop the script" in SYSTEM


def test_the_prompt_asks_for_a_lean_script():
    from agent.prompt import SYSTEM
    assert "no plots beyond those the task requires" in SYSTEM


def test_the_output_contract_is_restated_last(tmp_path):
    """It appears earlier too, but earlier is the middle of a long prompt. This is the one
    constraint that costs a unit outright and is free to get right."""
    from agent.prompt import build_prompt
    ctx = parse_task(UNITS / "t1-american-option-fd-new", probe_filesystem=False)
    body = build_prompt(ctx, "/app/output")[1]["content"]
    tail = body[-1200:]
    assert "Before you finish" in tail
    for spec in ctx.outputs:
        assert spec.filename in tail
    assert "/app/output" in tail


def test_the_restatement_is_phrased_positively(tmp_path):
    """A prohibition at the top of a prompt is the easiest thing in it to forget."""
    from agent.prompt import build_prompt
    ctx = parse_task(UNITS / "t1-zero-coupon-bootstrapping", probe_filesystem=False)
    tail = build_prompt(ctx, "/app/output")[1]["content"].split("## Before you finish")[1]
    assert "Write exactly these files" in tail
    assert "character for character" in tail
