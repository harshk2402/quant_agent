# SPDX-License-Identifier: MIT
"""Run the agent across the public practice tasks and score it with their real graders.

Dev-only; never shipped in the image.

This is the only honest feedback loop available before submitting. It answers two questions:

  1. What fraction of tasks do we actually solve?  (the pass@1 proxy)
  2. WHERE do we lose them?

The second matters more than the first. A score of 40/87 means nothing on its own: if the losses
are scripts that crashed, the fix is the repair loop; if they ran cleanly and produced wrong
numbers, the fix is domain knowledge. The aggregate cannot tell those apart, so every task is
recorded with an outcome label.

Two caveats on the number it produces:

  * These are PRACTICE tasks. The ~30 that decide the ranking are sealed and authored separately.
  * The model here is a development stand-in, not the organizers' house model.

So treat the result as a measure of DIRECTION — did this change help? — and never as a prediction
of where we would place.

Usage:
    python3 tools/baseline_sweep.py                 # every task, resuming if interrupted
    python3 tools/baseline_sweep.py --limit 5       # a quick sanity pass
    python3 tools/baseline_sweep.py --units a,b     # named tasks only
    python3 tools/baseline_sweep.py --no-grade      # run without Docker grading (fast, partial)
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from agent import llm  # noqa: E402
from agent.solve import run_solve  # noqa: E402
from agent.prompt import strip_canary  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
KIT = pathlib.Path(os.environ.get("QFBENCH_KIT", REPO.parent / "track1-coding-public"))
SANDBOX_IMAGE = "finance-bench-sandbox:latest"

#: Tighter than the per-task budget on purpose. A task's card may allow 1800s+, but across 87
#: tasks a single hanging script would stall the whole sweep. Anything legitimately slower can be
#: re-run on its own with the real limit.
SWEEP_EXEC_TIMEOUT = 120.0

#: Grading runs the unit's own checker inside the sandbox image; it should take seconds.
GRADE_TIMEOUT = 300.0


@dataclasses.dataclass
class UnitResult:
    unit: str
    outcome: str = ""            # why the agent run ended as it did
    llm_error: str = ""          # verbatim client error, so quota can be told from a real fault
    reward: float | None = None  # from the unit's real checker; None when grading was skipped
    graded: bool = False
    grade_error: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_reasoning: int = 0    # of tokens_out, the share spent on the model's internal reasoning
    seconds: float = 0.0
    deliverables: int = 0
    non_empty: int = 0
    attempts: int = 0            # generation attempts, including the first
    repairs: int = 0             # of those, how many were driven by a failure signal
    requests: int = 0            # model calls spent, continuations included (allowance is 25)
    continuations: int = 0       # responses spent extending a script past the 4k output ceiling
    rewound_lines: int = 0       # lines discarded to reach a parsable prefix after truncation
    name_repairs: int = 0        # rewinds driven by a name used before it was assigned
    properties_written: bool = False  # the model produced a checker for its own answer
    property_violations: int = 0      # what that checker reported on the final attempt
    critic_verdict: str = ""          # "plausible" | "wrong" | "" (not consulted)
    critic_repaired: bool = False     # a generation was spent on the critic's objection
    critic_choice: str = ""           # which answer the blind comparison kept
    restarted: bool = False           # a stalled repair chain was abandoned for a fresh generation
    stages: list[str] = dataclasses.field(default_factory=list)  # what each layer did
    truncated: bool = False      # at least one response hit the ceiling
    stopped_early: str = ""      # set when a budget guard cut repair short
    exec_failure: str = ""       # the last attempt's stderr tail, canary-stripped, for diagnosis
    attempt_errors: list[str] = dataclasses.field(default_factory=list)  # how each attempt ended
    error_codes: list[str] = dataclasses.field(default_factory=list)
    canaries_scrubbed: list[str] = dataclasses.field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.reward is not None and self.reward >= 1.0


def grade(unit_dir: pathlib.Path, out_dir: pathlib.Path) -> tuple[float | None, str]:
    """Run the unit's own checker in Docker and read the reward it writes.

    Mirrors scripts/selfgrade.sh. Self-grading needs more mounts than running does: 36 of the 87
    checkers open paths the plain run recipe never provides -- /tests/reference_data, /app/data,
    and in a few cases a bare /app/<file>. Without them the failure is about mounts rather than
    about the answer.
    """
    mounts = [
        "-v", f"{unit_dir}:/input:ro",
        "-v", f"{out_dir}:/output",
        "-v", f"{out_dir}:/app/output",
        "-v", f"{unit_dir / 'checks'}:/checks:ro",
    ]
    data = unit_dir / "environment" / "data"
    if data.is_dir():
        mounts += ["-v", f"{data}:/app/data:ro"]
        for f in sorted(data.iterdir()):
            if f.is_file():
                mounts += ["-v", f"{f}:/app/{f.name}:ro"]
    ref = unit_dir / "checks" / "reference_data"
    if ref.is_dir():
        mounts += ["-v", f"{ref}:/tests/reference_data:ro"]

    try:
        subprocess.run(
            ["docker", "run", "--rm", "--network=none", *mounts,
             SANDBOX_IMAGE, "bash", "/checks/test.sh"],
            capture_output=True, text=True, timeout=GRADE_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return None, "grading timed out"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

    reward_file = out_dir / "reward.json"
    if not reward_file.is_file():
        return None, "checker wrote no reward.json"
    try:
        return float(json.loads(reward_file.read_text()).get("reward", 0.0)), ""
    except Exception as e:
        return None, f"unreadable reward.json: {e}"


#: Consecutive quota refusals before the sweep gives up. One can be transient; three in a row on a
#: sequential run means the daily allowance is gone and every remaining task would be recorded as a
#: failure that is really a billing limit.
QUOTA_STRIKES = 3

_QUOTA_SIGNS = ("429", "rate limit", "ratelimit", "quota", "resource_exhausted", "exhausted")


def looks_like_quota(err: str) -> bool:
    return any(s in (err or "").lower() for s in _QUOTA_SIGNS)


def run_unit(unit_dir: pathlib.Path, work: pathlib.Path, do_grade: bool) -> UnitResult:
    out_dir = work / unit_dir.name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # The token budget is per task, and SESSION_USAGE is cumulative across calls -- without a reset
    # the sweep would trip BudgetExceeded partway through and blame the wrong task.
    llm.SESSION_USAGE = llm.Usage()

    r = UnitResult(unit=unit_dir.name)
    t0 = time.time()
    try:
        sr = run_solve(str(unit_dir), str(out_dir))
        r.outcome = sr.outcome()
        r.tokens_in, r.tokens_out = sr.tokens_in, sr.tokens_out
        r.tokens_reasoning = sr.tokens_reasoning
        r.deliverables, r.non_empty = sr.deliverables, sr.non_empty
        r.attempts, r.repairs, r.stopped_early = sr.attempts, sr.repairs, sr.stopped_early
        # Recorded because "exec rc=1" alone is not diagnostic, and re-running a unit to see the
        # traceback costs model calls. Canary-stripped: a traceback quotes lines of the generated
        # script, which may carry text the model copied out of the instruction.
        r.exec_failure = strip_canary(sr.exec_failure or "")[:1500]
        r.attempt_errors = [strip_canary(e)[:200] for e in sr.attempt_errors]
        r.requests, r.continuations = sr.requests, sr.continuations
        r.rewound_lines, r.truncated = sr.rewound_lines, sr.truncated
        r.name_repairs = sr.name_repairs
        r.properties_written = sr.properties_written
        r.property_violations = sr.property_violations
        r.critic_verdict, r.critic_repaired = sr.critic_verdict, sr.critic_repaired
        r.critic_choice = sr.critic_choice
        r.restarted = sr.restarted
        r.stages = list(sr.stages)
        r.llm_error = sr.llm_error
        r.error_codes, r.canaries_scrubbed = sr.error_codes, sr.canaries_scrubbed
    except Exception as e:
        r.outcome = f"sweep_error: {type(e).__name__}: {e}"
    r.seconds = round(time.time() - t0, 1)

    if do_grade:
        r.reward, r.grade_error = grade(unit_dir, out_dir)
        r.graded = r.reward is not None
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N tasks (0 means none, for inspecting a "
                         "recorded run without running anything)")
    ap.add_argument("--units", default="", help="comma-separated task ids")
    ap.add_argument("--set", dest="setname", default="",
                    help="a named subset from tools/testsets.py (representative, smoke)")
    ap.add_argument("--no-grade", action="store_true", help="skip Docker grading")
    ap.add_argument("--force", action="store_true", help="re-run tasks already recorded")
    ap.add_argument("--include-excluded", action="store_true",
                    help="also run the tasks testsets.EXCLUDED holds out of the roster")
    ap.add_argument("--out", default=str(REPO / "sweeps"), help="where results are written")
    args = ap.parse_args()

    units_root = KIT / "units"
    if not units_root.is_dir():
        print(f"practice kit not found at {units_root}; set $QFBENCH_KIT", file=sys.stderr)
        return 2

    units = sorted(d for d in units_root.iterdir() if (d / "instruction.md").is_file())

    # Held-out tasks come off the denominator, loudly. A task that cannot be passed by anyone is a
    # fixed subtraction from every score; keeping it in makes each run look worse by the same amount
    # and buries real movement. Printing what was dropped, and why, is the part that keeps this
    # honest rather than self-serving.
    from testsets import EXCLUDED
    if not args.include_excluded:
        held = [u for u in units if u.name in EXCLUDED]
        units = [u for u in units if u.name not in EXCLUDED]
        for u in held:
            print(f"excluded from the roster: {u.name} -- {EXCLUDED[u.name]}", file=sys.stderr)

    if args.setname:
        from testsets import resolve
        wanted = set(resolve(args.setname))
        units = [u for u in units if u.name in wanted]
        missing = wanted - {u.name for u in units}
        if missing:
            print(f"warning: set names tasks not in the kit: {sorted(missing)}", file=sys.stderr)
    if args.units:
        wanted = {u.strip() for u in args.units.split(",")}
        units = [u for u in units if u.name in wanted]
    # `is not None`, not truthiness: `--limit 0` is a request for no units, and treating it as
    # "no limit given" means a command meant to inspect a recorded run instead starts a live one
    # over the whole roster. That cost five units of quota before it was caught.
    if args.limit is not None:
        units = units[:args.limit]

    # RESOLVED, not taken as given. Grading bind-mounts the output directory into Docker, and
    # `docker run -v` requires an absolute path -- a relative one is read as a named volume, so the
    # checker runs against an empty mount, writes no reward.json, and every unit comes back
    # ungraded with no sign that anything was wrong with the command rather than the agent.
    outdir = pathlib.Path(args.out).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    results_file = outdir / "results.jsonl"
    work = outdir / "work"
    work.mkdir(exist_ok=True)

    # Resume rather than restart: a sweep is minutes of model calls, and losing it to one crash
    # (or a rate limit) would be expensive.
    done: dict[str, dict] = {}
    if results_file.is_file() and not args.force:
        for line in results_file.read_text().splitlines():
            try:
                rec = json.loads(line)
                # A result recorded before the task was held out would otherwise keep the old
                # denominator alive: the row stays in results.jsonl as history, but it does not
                # count towards a pass rate the task can no longer contribute to.
                if rec["unit"] in EXCLUDED and not args.include_excluded:
                    continue
                done[rec["unit"]] = rec
            except Exception:
                continue

    todo = [u for u in units if u.name not in done]
    print(f"sweep: {len(units)} task(s), {len(done)} already recorded, {len(todo)} to run",
          file=sys.stderr)
    if not args.no_grade:
        print(f"grading in Docker with {SANDBOX_IMAGE}", file=sys.stderr)

    os.environ.setdefault("QFBENCH_EXEC_TIMEOUT", str(SWEEP_EXEC_TIMEOUT))

    strikes = 0
    with results_file.open("a") as fh:
        for i, unit in enumerate(todo, 1):
            r = run_unit(unit, work, do_grade=not args.no_grade)

            # A quota refusal is not an agent failure, and recording it as one would poison the
            # results: a sweep that hits the daily cap at task 60 would otherwise report the
            # remaining 27 as broken. Stop instead, and leave them unrecorded so that re-running
            # tomorrow resumes exactly where this left off.
            if looks_like_quota(r.llm_error):
                strikes += 1
                print(f"[{i:>3}/{len(todo)}] QUOTA  {r.unit:<46} {r.llm_error[:60]}",
                      file=sys.stderr)
                if strikes >= QUOTA_STRIKES:
                    print(f"\nSTOPPING: {strikes} consecutive quota refusals. This is the daily "
                          f"allowance, not the agent.\n"
                          f"{i - strikes} task(s) recorded and kept; the rest are untouched and "
                          f"will be picked up on the next run.", file=sys.stderr)
                    break
                continue          # do not record a quota refusal as a result
            strikes = 0

            fh.write(json.dumps(dataclasses.asdict(r)) + "\n")
            fh.flush()
            done[r.unit] = dataclasses.asdict(r)
            mark = "PASS" if r.passed else ("...." if r.reward is None else "fail")
            print(f"[{i:>3}/{len(todo)}] {mark}  {r.unit:<46} {r.outcome:<16} "
                  f"{r.seconds:>6.1f}s  {r.tokens_in + r.tokens_out:>6} tok", file=sys.stderr)

    report(list(done.values()), outdir)
    return 0


def report(records: list[dict], outdir: pathlib.Path) -> None:
    n = len(records)
    graded = [r for r in records if r.get("reward") is not None]
    passed = [r for r in graded if r["reward"] >= 1.0]

    print("\n" + "=" * 68)
    print(f"tasks run      : {n}")
    if graded:
        print(f"graded         : {len(graded)}")
        print(f"PASSED         : {len(passed)}/{len(graded)}  "
              f"= pass@1 {len(passed) / len(graded):.3f}")
    out_tokens = sum(r["tokens_out"] for r in records)
    thinking = sum(r.get("tokens_reasoning", 0) for r in records)
    print(f"tokens         : {sum(r['tokens_in'] for r in records):,} in / {out_tokens:,} out")
    if thinking:
        # The share of the per-request output ceiling that never reaches the script. A reasoning
        # model is charged for its thinking against the same 4,000 tokens as its answer, so this
        # is budget spent on something we never see and cannot reuse.
        print(f"               : {thinking:,} of those output tokens were reasoning "
              f"({thinking / max(out_tokens, 1):.0%})")
    print(f"wall time      : {sum(r['seconds'] for r in records) / 60:.1f} min")

    # How much of the outcome came from repair rather than the first attempt. Without this the
    # sweep cannot distinguish "the loop fixed it" from "the first try happened to work".
    repaired = [r for r in records if r.get("repairs", 0) > 0]
    if repaired:
        rescued = [r for r in repaired if (r.get("reward") or 0) >= 1.0]
        print(f"\nrepair: {len(repaired)} task(s) needed at least one retry, "
              f"{len(rescued)} of those ended up passing")
        print(f"        attempts per task: mean "
              f"{sum(r.get('attempts', 0) for r in records) / max(len(records), 1):.2f}")
        cut = [r["unit"] for r in records if r.get("stopped_early")]
        if cut:
            print(f"        stopped early by a budget guard: {len(cut)} ({', '.join(cut[:4])})")

    # The output ceiling, which is the thing this run exists to answer. The September baseline had
    # no way to see it: a response cut off at the cap looked exactly like a complete one, and seven
    # tasks that were pinning the limit were recorded only as crashes.
    hit = [r for r in records if r.get("truncated")]
    if hit:
        rescued = [r for r in hit if (r.get("reward") or 0) >= 1.0]
        print(f"\noutput ceiling: {len(hit)} task(s) had a response cut off at the cap, "
              f"{len(rescued)} of those still passed")
        print(f"        continuations per affected task: mean "
              f"{sum(r.get('continuations', 0) for r in hit) / max(len(hit), 1):.2f}")
        print(f"        lines rewound to a parsable prefix: "
              f"{sum(r.get('rewound_lines', 0) for r in hit)} across those tasks")
    repaired_names = [r for r in records if r.get("name_repairs")]
    if repaired_names:
        # A seam that lost a line, found by the static check before the script was run. Worth its
        # own line: it is the failure that replaced truncation once assembly was working.
        print(f"        seams that lost a line, caught before execution: {len(repaired_names)} "
              f"task(s), {sum(r['name_repairs'] for r in repaired_names)} repair(s)")
    reqs = [r.get("requests", 0) for r in records if r.get("requests")]
    if reqs:
        # Against an allowance of 25 per unit. Worth watching rather than assuming: continuation
        # spends requests, and the point of the allowance is that we have room to.
        print(f"        requests per task: mean {sum(reqs) / len(reqs):.1f}, max {max(reqs)} "
              f"(allowance 25)")

    print("\nwhere tasks ended up (agent-side outcome):")
    counts: dict[str, int] = {}
    for r in records:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        wins = sum(1 for r in records
                   if r["outcome"] == k and (r.get("reward") or 0) >= 1.0)
        print(f"   {k:<18} {v:>3}   (passed {wins})")

    codes: dict[str, int] = {}
    for r in records:
        for c in r.get("error_codes", []):
            codes[c] = codes.get(c, 0) + 1
    if codes:
        print("\ncontract errors seen:")
        for k, v in sorted(codes.items(), key=lambda kv: -kv[1]):
            print(f"   {k:<20} {v}")

    scrubbed = [r["unit"] for r in records if r.get("canaries_scrubbed")]
    if scrubbed:
        print(f"\ncanary markers removed from output on {len(scrubbed)} task(s): "
              f"{', '.join(scrubbed[:5])}")

    losses = [r for r in graded if r["reward"] < 1.0]
    if losses:
        print(f"\nfailed tasks ({len(losses)}):")
        for r in sorted(losses, key=lambda r: r["outcome"]):
            print(f"   {r['unit']:<46} {r['outcome']}")

    print(f"\nfull records: {outdir / 'results.jsonl'}")


if __name__ == "__main__":
    raise SystemExit(main())
