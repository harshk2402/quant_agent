# SPDX-License-Identifier: MIT
"""The `solve` verb — the CLI contract the harness invokes.

    solve --task-dir /input --out /app/output

Three contract rules, each of which silently zeroes every unit if broken:

1.  **The verb arrives as the first argument after the image reference.** It is consumed here as
    a positional. Getting this wrong exits 127/126 on every unit and is recorded as our failure.
2.  **Never crash.** One uncaught exception forfeits every gate for that unit at once — "a wrong
    answer is scored, an exception is not". So the solve body is wrapped, and we still write a
    well-formed file of the right name and shape on failure, then exit 0.
3.  **Never leak a canary.** 66/87 units carry a canary GUID in instruction.md; gate g2 scans every
    text file we write. So no instruction or input text is ever echoed into a deliverable, and no
    scratch file or log is written to the output directory.

The loop is: prompt → generate → stage inputs → execute → check the output → repair and retry if
either the script failed or its output does not satisfy the contract. Repair is bounded by
attempts, wall clock and token budget, whichever binds first.

Repair exists because of what the first baseline measured: across 33 tasks, 55% failed with the
generated script crashing and 0% of those scored, while of the tasks whose script ran cleanly 40%
were correct. Crashes were the largest single loss and the only one addressable generically.

Placeholders remain as the floor. Whatever the model does or fails to do, every declared
deliverable exists by the time we exit — an empty file of the right name and shape still clears
more gates than a missing one, and costs nothing when generation succeeded.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import pathlib
import sys
import tempfile
import shutil
import time
import traceback

from .execute import exec_timeout_for, run_code, stage_inputs
from .llm import LLMError
from .assemble import generate_script, plan_once
from . import critic, properties
from .llm import chat
from .prompt import (build_prompt, build_properties_prompt, build_repair_message,
                     build_restart_message, describe_missing, extract_code, locate_failure,
                     strip_canary)
from .task_context import OutputSpec, TaskContext, parse_task, resolve_output_dir
from .validate import known_canaries, scrub_canaries, validate_outputs


def _write_placeholder(spec: OutputSpec, ctx: TaskContext, out_dir) -> None:
    """Write a structurally valid, empty deliverable of the declared format.

    Content is derived only from the parsed CONTRACT (filenames, column names) -- never from
    instruction or input text, which is what keeps canary GUIDs out of the output.
    """
    path = spec.path_in(out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = spec.columns

    if spec.fmt in ("csv", "tsv"):
        sep = "\t" if spec.fmt == "tsv" else ","
        path.write_text(sep.join(cols) + "\n" if cols else "")
    elif spec.fmt in ("json", "jsonl"):
        path.write_text("{}\n" if spec.fmt == "json" else "")
    elif spec.fmt == "parquet":
        try:
            import pandas as pd
            pd.DataFrame({c: [] for c in cols} if cols else {}).to_parquet(path, index=False)
        except Exception:
            path.write_bytes(b"")          # better an empty file than no file
    elif spec.fmt == "html":
        path.write_text("<!doctype html><title>placeholder</title>\n")
    elif spec.fmt == "png":
        # 1x1 transparent PNG, so the file is a valid image rather than empty bytes.
        path.write_bytes(bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
            "890000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082"))
    else:
        path.write_text("")


@dataclasses.dataclass
class SolveResult:
    """What happened on one task, for callers that need more than an exit code.

    The evaluation harness (tools/baseline_sweep.py) needs to distinguish "the model returned no
    code" from "the code crashed" from "it ran but the answer was wrong" — those call for
    completely different work. Scraping that back out of log lines would be fragile, so the
    outcome is returned as data and `solve()` prints it.
    """
    unit_id: str = ""
    parsed: bool = False
    generated: bool = False          # a code block came back and was extracted
    exec_ok: bool = False            # that code ran to completion without raising
    exec_timeout: bool = False
    llm_error: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_reasoning: int = 0        # of tokens_out, the share the model spent thinking
    deliverables: int = 0
    non_empty: int = 0
    attempts: int = 0                # generation attempts made, including the first
    repairs: int = 0                 # follow-up attempts driven by a failure signal
    properties_written: bool = False  # the model produced a checker for its own answer
    property_violations: int = 0      # what that checker reported on the final attempt
    critic_verdict: str = ""          # "plausible" | "wrong" | "" (not consulted)
    critic_problems: list[str] = dataclasses.field(default_factory=list)
    critic_repaired: bool = False     # a generation was spent on the critic's objection
    critic_choice: str = ""           # "kept" | "took the revision" | "" (no comparison made)
    restarted: bool = False           # a stalled repair chain was abandoned for a fresh generation
    #: One line per stage that did something, so a run can be read without the full log.
    stages: list[str] = dataclasses.field(default_factory=list)
    requests: int = 0                # model calls spent, continuations included
    continuations: int = 0           # responses spent extending a script past the output ceiling
    rewound_lines: int = 0           # lines discarded to reach a parsable prefix after truncation
    name_repairs: int = 0            # rewinds driven by a name used before it was assigned
    truncated: bool = False          # at least one response hit the output ceiling
    stopped_early: str = ""          # "time" when the wall-clock guard cut repair short
    exec_failure: str = ""           # the last failing attempt's stderr tail, for the harness
    attempt_errors: list[str] = dataclasses.field(default_factory=list)  # one line per attempt
    error_codes: list[str] = dataclasses.field(default_factory=list)
    canaries_scrubbed: list[str] = dataclasses.field(default_factory=list)

    def outcome(self) -> str:
        """A single label for why this task ended as it did. Ordered most-specific first."""
        if not self.parsed:
            return "parse_failed"
        if self.llm_error:
            return "llm_error"
        if not self.generated:
            return "no_code"
        if self.exec_timeout:
            return "exec_timeout"
        if not self.exec_ok:
            return "exec_failed"
        if self.error_codes:
            return "contract_error"
        return "produced_output"


def solve(task_dir: str, out: str | None) -> int:
    """Solve one unit. Returns a process exit code -- and it is always 0 (see rule 2)."""
    res = run_solve(task_dir, out)
    if res.stages:
        print(f"[solve] stages: {' | '.join(res.stages)}", file=sys.stderr)
    print(f"[solve] done: {res.non_empty}/{res.deliverables} deliverable(s) non-empty "
          f"({res.outcome()})", file=sys.stderr)
    return 0


def run_solve(task_dir: str, out: str | None) -> SolveResult:
    """The actual pipeline. Never raises; failures are recorded in the result."""
    res = SolveResult()
    out_dir = resolve_output_dir(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        ctx = parse_task(task_dir)
        res.parsed = True
    except Exception:
        # Even the parser failing must not crash the run. Nothing is known about the contract
        # here, so there is nothing safe to write -- exit 0 and let the gates record it.
        traceback.print_exc(file=sys.stderr)
        return res

    res.unit_id = ctx.unit_id or str(task_dir)
    res.deliverables = len(ctx.outputs)
    print(f"[solve] {res.unit_id}: "
          f"{len(ctx.inputs)} input(s), {len(ctx.outputs)} deliverable(s) -> {out_dir}",
          file=sys.stderr)
    for w in ctx.warnings:
        print(f"[solve] warning: {w}", file=sys.stderr)

    if os.environ.get("QFBENCH_NO_LLM") != "1":
        try:
            _generate_and_run(ctx, out_dir, res)
        except Exception:
            # Generation is best-effort. Whatever went wrong, the placeholders below still run.
            traceback.print_exc(file=sys.stderr)

    # Validate BEFORE backfilling. Once placeholders are written every file exists, so a
    # "missing_file" finding would never fire — and that finding is exactly the signal the repair
    # loop (Step 0.7) will act on. This report describes what the model actually produced.
    try:
        report = validate_outputs(ctx, out_dir)
        res.error_codes = [f.code for f in report.errors]
        print(f"[solve] contract check: {report.summary()}", file=sys.stderr)
        for f in report.findings:
            print(f"[solve]   {f}", file=sys.stderr)
    except Exception:
        traceback.print_exc(file=sys.stderr)

    # The floor, always. Only fills what the generated code did not already produce, so a real
    # solution is never overwritten by an empty file.
    for spec in ctx.outputs:
        try:
            if not spec.path_in(out_dir).exists():
                _write_placeholder(spec, ctx, out_dir)
        except Exception:
            traceback.print_exc(file=sys.stderr)

    # Last thing before exit, and the one check that repairs rather than reports: a canary marker
    # left in any output file fails the task outright, however correct the answer is.
    try:
        cleaned = scrub_canaries(out_dir, known_canaries(task_dir))
        res.canaries_scrubbed = cleaned
        if cleaned:
            print(f"[solve] removed canary marker(s) from: {', '.join(cleaned)}", file=sys.stderr)
    except Exception:
        traceback.print_exc(file=sys.stderr)

    res.non_empty = sum(1 for s in ctx.outputs
                        if s.path_in(out_dir).exists()
                        and s.path_in(out_dir).stat().st_size > 0)
    return res


def _stuck(errors: list[str]) -> bool:
    """Has the last attempt failed the same WAY as the one before it?

    The KIND of failure, not its text: line numbers and values move between attempts even when the
    bug is identical, so `TypeError` matching `TypeError` is the signal and the rest is noise.
    Timeouts match outright. Contract errors are deliberately excluded -- a missing file is a
    different kind of problem and repairing against it does converge.
    """
    if len(errors) < 2:
        return False

    def kind(e: str) -> str:
        e = (e or "").strip()
        if e.startswith("TIMEOUT"):
            return "TIMEOUT"
        if e.startswith(("contract:", "properties:", "critic")) or e == "clean":
            return ""
        return e.split(":", 1)[0].strip()

    last, previous = kind(errors[-1]), kind(errors[-2])
    return bool(last) and last == previous


def _critic_enabled() -> bool:
    """OFF by default. Set QFBENCH_CRITIC=1 to turn it on.

    Measured offline it caught 2 of 3 gross failures; measured live across eight verdicts it caught
    nothing, raised three false alarms and missed two real failures, and each false alarm roughly
    tripled the unit's wall clock. Its niche -- output that is obviously broken -- is now covered
    deterministically by the degenerate-value check, and the failures that remain are small
    convention drifts against a hidden reference, which no reviewer of the numbers alone can see.
    The code and tests are kept.
    """
    return os.environ.get("QFBENCH_CRITIC", "").strip() == "1"


def _ask_critic(ctx, out_dir, messages, res: SolveResult) -> tuple[bool, list[str]]:
    """One call. Any failure is read as `plausible`, because this must never cost a unit."""
    checklist_on = os.environ.get("QFBENCH_NO_CRITIC_CHECKLIST", "").strip() != "1"
    prompt = critic.build_critic_prompt(messages, out_dir,
                                        ctx.category if checklist_on else "")
    if prompt is None:
        return False, []
    try:
        reply = chat(prompt, max_tokens=1200)
    except LLMError as e:
        print(f"[solve] critic: call failed, accepting the answer ({e})", file=sys.stderr)
        return False, []
    res.requests += 1
    res.tokens_in += reply.usage.input_tokens
    res.tokens_out += reply.usage.output_tokens
    res.tokens_reasoning += reply.usage.reasoning_tokens
    wrong, problems = critic.parse_verdict(reply.text)
    res.critic_verdict = "wrong" if wrong else "plausible"
    res.critic_problems = problems
    print(f"[solve] critic: {res.critic_verdict}", file=sys.stderr)
    for p in problems:
        print(f"[solve] critic problem: {p}", file=sys.stderr)
    return wrong, problems


def _one_more_for_the_critic(ctx, out_dir, res: SolveResult, messages, exec_timeout,
                             clean_snapshot, deadline) -> None:
    """Exactly one further generation, then a blind choice between the two answers.

    One, because a false alarm must not cost a unit its whole budget -- which is what happened the
    one time the properties layer fired. Blind, because the critic is not a neutral judge of
    whether its own objection was met.
    """
    try:
        if deadline is not None and time.monotonic() >= deadline:
            res.stages.append("critic: no time for a revision")
            return
        before = critic.read_outputs(out_dir)
        asm = generate_script(messages, deadline=deadline, plan_first=False)
        res.requests += asm.responses
        res.tokens_in += asm.tokens_in
        res.tokens_out += asm.tokens_out
        res.tokens_reasoning += asm.tokens_reasoning
        if not asm.code.strip():
            res.stages.append("critic: revision produced no script")
            _restore(out_dir, clean_snapshot)
            return
        result = run_code(asm.code, out_dir, timeout=exec_timeout)
        report = validate_outputs(ctx, out_dir)
        res.attempt_errors.append("critic revision: " + _one_line_error(result, report, []))
        if not result.ok or not report.ok:
            # The revision is worse on the measures we can check. No judgement call needed.
            res.stages.append("critic: revision failed, kept the original")
            res.critic_choice = "kept"
            _restore(out_dir, clean_snapshot)
            return

        after = critic.read_outputs(out_dir)
        keep_original = _blind_choice(messages, before, after, res)
        if keep_original:
            res.stages.append("critic: chose the original")
            res.critic_choice = "kept"
            _restore(out_dir, clean_snapshot)
        else:
            res.stages.append("critic: chose the revision")
            res.critic_choice = "took the revision"
    finally:
        shutil.rmtree(clean_snapshot, ignore_errors=True)


def _blind_choice(messages, before: str, after: str, res: SolveResult) -> bool:
    """True to keep the original. Unlabelled, and in an order the model cannot infer."""
    import random

    original_first = random.Random(os.environ.get("QFBENCH_SEED", "0")).random() < 0.5
    first, second = (before, after) if original_first else (after, before)
    try:
        reply = chat(critic.build_compare_prompt(messages, first, second), max_tokens=400)
    except LLMError:
        return True                 # cannot choose: keep what already passed every check we have
    res.requests += 1
    res.tokens_in += reply.usage.input_tokens
    res.tokens_out += reply.usage.output_tokens
    choice = critic.parse_choice(reply.text)
    if not choice:
        return True
    chose_first = choice == "A"
    return chose_first == original_first


def _one_line_error(result, report, violations) -> str:
    """One line summarising how an attempt ended, for the per-attempt record."""
    if not result.ok:
        tail = (result.failure_text(400) or "").strip().splitlines()
        return tail[-1][:160] if tail else f"exit {result.returncode}"
    if not report.ok:
        codes = sorted({f.code for f in report.errors})
        return "contract: " + ", ".join(codes)
    if violations:
        return f"properties: {violations[0][:140]}"
    return "clean"


def _output_score(ctx: TaskContext, out_dir) -> tuple[int, int]:
    """How good is what is on disk right now? Higher is better.

    Fewest contract errors first, then most deliverables actually written. Both come from the
    validator we already run, so this costs nothing and calls no model.
    """
    report = validate_outputs(ctx, out_dir)
    written = sum(1 for spec in ctx.outputs
                  if spec.path_in(out_dir).is_file() and spec.path_in(out_dir).stat().st_size > 0)
    return (-len(report.errors), written)


def _keep_best(out_dir, snapshot: pathlib.Path | None) -> pathlib.Path:
    """Copy the output directory aside, replacing any previous snapshot."""
    if snapshot is not None:
        shutil.rmtree(snapshot, ignore_errors=True)
    snapshot = pathlib.Path(tempfile.mkdtemp(prefix="qa-best-"))
    shutil.copytree(out_dir, snapshot, dirs_exist_ok=True)
    return snapshot


def _restore(out_dir, snapshot: pathlib.Path) -> None:
    """Put the snapshot back, replacing whatever the last attempt left.

    The directory itself is never removed -- the harness binds it, as both /output and /app/output,
    and deleting a mount point is not the same as emptying it.
    """
    out_dir = pathlib.Path(out_dir)
    for item in out_dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item, ignore_errors=True)
        else:
            item.unlink(missing_ok=True)
    shutil.copytree(snapshot, out_dir, dirs_exist_ok=True)


def _generate_and_run(ctx: TaskContext, out_dir, res: SolveResult) -> None:
    """Generate, execute, and repair until the output satisfies the contract or a budget runs out.

    Bounded by three things, whichever binds first:

      * attempts — `QFBENCH_MAX_ATTEMPTS`, default 3. Later rounds have sharply diminishing
        returns, and each one costs a model call plus an execution.
      * wall clock — a task is scored on what exists when its container is killed, so an attempt
        is only started if there is plausibly time to finish it. Running out mid-repair with
        nothing written would be strictly worse than stopping and keeping the previous attempt.
      * tokens — enforced inside the client, which raises rather than overspending.

    Each attempt may itself span several responses. The route caps one response at 4,000 output
    tokens, which is below what about half the roster's scripts need, so a generation that is cut
    off is continued rather than abandoned (see agent/assemble.py). That spends requests out of an
    allowance of 25, which an attempt was never close to using.

    Failure signals come from two places and both are fed back: a traceback when the script raised,
    and the contract findings when it ran but produced the wrong files. The second matters because
    nothing crashed, so the model has no way to know anything is wrong unless it is told.
    """
    max_attempts = _int_env("QFBENCH_MAX_ATTEMPTS", 3)
    # The card's own allowance, and a ceiling of our own on top of it.
    #
    # The card says what ONE unit may take. The final phase gives 24 hours for the WHOLE roster,
    # and a unit the run never reaches is recorded as a failure -- so a unit that spends 1,440
    # seconds is not using its own budget, it is using the budget of the units behind it. 86,400
    # seconds across ~86 units is roughly 1,000 each, and one measured unit took 1,111.
    total_budget = ctx.agent_timeout_sec or 1800.0
    # Leave a margin for parsing, staging, validation, backfill and container teardown.
    budget = total_budget * 0.8
    ceiling = _float_env("QFBENCH_UNIT_BUDGET", 900.0)
    if ceiling > 0:
        budget = min(budget, ceiling)
    deadline = time.monotonic() + budget
    print(f"[solve] budget: {budget:.0f}s for this unit "
          f"(card allows {total_budget:.0f}s)", file=sys.stderr)
    exec_timeout = exec_timeout_for(ctx)

    staged = stage_inputs(ctx)
    for src, dst in staged:
        print(f"[solve] staged {src} -> {dst}", file=sys.stderr)

    messages = build_prompt(ctx, out_dir)
    os.environ["QFBENCH_UNIT_ID"] = ctx.unit_id or "unit"    # for QFBENCH_KEEP_PLAN's filename

    # Plan once per unit. Kept aside so a restart can return to the task WITH its plan but WITHOUT
    # the lineage of broken scripts. The plan was never implicated in any failure we measured, and
    # re-deriving it on every attempt spent a request re-rolling a decision that was fine.
    try:
        messages, planning = plan_once(messages, deadline=deadline)
    except LLMError as e:
        res.llm_error = str(e)
        print(f"[solve] planning call failed: {e}", file=sys.stderr)
        return
    res.requests += planning.responses
    res.tokens_in += planning.tokens_in
    res.tokens_out += planning.tokens_out
    res.tokens_reasoning += planning.tokens_reasoning
    if planning.planned:
        res.stages.append("planned")
    base_messages: list[dict] = list(messages)

    # Keep the best output any attempt produced and put it back if a later attempt left something
    # worse. Without this the loop has no memory of having been better: every attempt writes into
    # the same directory, so one that crashes having written nothing silently replaces one that
    # wrote six of seven files. There is no upside to that trade, and we have watched units pass on
    # one attempt and fail on the next.
    best_score: tuple[int, int] | None = None
    best_dir: pathlib.Path | None = None

    # One call, before any code exists, asking what must be true of a correct answer. Asked after
    # the fact a checker describes whatever got written; asked from the task it describes the task.
    # Generated once and reused across attempts -- the properties are about the answer, and the
    # answer does not change in kind between attempts.
    # OFF by default. Measured across four unit-runs it caught nothing, stayed silent on three
    # outputs we knew were wrong, and the one time it fired it was on a backwards rule -- it
    # demanded that 95% VaR exceed the worst observed daily loss, which is the wrong way round --
    # and spent all three attempts of a unit whose first attempt was already clean. Set
    # QFBENCH_PROPERTIES=1 to try it again; the code and its tests are kept.
    checker = ""
    if os.environ.get("QFBENCH_PROPERTIES", "").strip() == "1":
        try:
            reply = chat(build_properties_prompt(messages), max_tokens=properties.MAX_TOKENS)
            res.tokens_in += reply.usage.input_tokens
            res.tokens_out += reply.usage.output_tokens
            res.tokens_reasoning += reply.usage.reasoning_tokens
            res.requests += 1
            checker = extract_code(reply.text) or ""
            res.properties_written = bool(checker.strip())
            # Saved for diagnosis, outside the output directory -- anything in there is a
            # deliverable the grader reads. Without this a checker that finds nothing is
            # indistinguishable from one that checks nothing, which is exactly the question.
            if checker.strip():
                keep = os.environ.get("QFBENCH_KEEP_CHECKER", "").strip()
                if keep:
                    target = pathlib.Path(keep) / f"{ctx.unit_id or 'unit'}.checker.py"
                    try:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_text(checker, encoding="utf-8")
                        print(f"[solve] properties: saved to {target}", file=sys.stderr)
                    except OSError as e:
                        print(f"[solve] properties: could not save checker ({e})", file=sys.stderr)
            print(f"[solve] properties: {'written' if checker.strip() else 'none returned'} "
                  f"({len(checker)} chars)", file=sys.stderr)
        except LLMError as e:
            print(f"[solve] properties: call failed, continuing without ({e})", file=sys.stderr)

    def remember() -> None:
        """Snapshot the output directory if it is the best we have seen."""
        nonlocal best_score, best_dir
        try:
            score = _output_score(ctx, out_dir)
        except Exception:
            return                      # scoring must never be the thing that fails a unit
        if best_score is None or score > best_score:
            best_score, best_dir = score, _keep_best(out_dir, best_dir)

    try:
        for attempt in range(1, max_attempts + 1):
            res.attempts = attempt
            try:
                asm = generate_script(messages, deadline=deadline, plan_first=False)
            except LLMError as e:
                res.llm_error = str(e)
                print(f"[solve] attempt {attempt}: model call failed: {e}", file=sys.stderr)
                return
            res.tokens_in += asm.tokens_in
            res.tokens_out += asm.tokens_out
            res.tokens_reasoning += asm.tokens_reasoning
            res.requests += asm.responses
            res.continuations += asm.continuations
            res.rewound_lines += asm.rewound_lines
            res.name_repairs += asm.name_repairs
            if asm.continuations or asm.rewound_lines:
                res.truncated = True
            print(f"[solve] attempt {attempt}: {asm.responses} response(s), "
                  f"{asm.continuations} continuation(s), rewound {asm.rewound_lines} line(s), "
                  f"{asm.name_repairs} name repair(s), complete={asm.complete}, "
                  f"out={asm.tokens_out} ({asm.tokens_reasoning} reasoning)", file=sys.stderr)
            for note in asm.notes:
                print(f"[solve] assembly: {note}", file=sys.stderr)

            code = asm.code
            if not code.strip():
                print(f"[solve] attempt {attempt}: no usable script assembled", file=sys.stderr)
                # A failed assembly is worth another attempt rather than a return: the next generation
                # may simply be shorter. Falling through gets the attempt accounting and the clock
                # guard for free.
                if attempt >= max_attempts:
                    return
                remaining = deadline - time.monotonic()
                if remaining < exec_timeout:
                    res.stopped_early = "time"
                    return
                res.repairs += 1
                continue
            res.generated = True

            result = run_code(code, out_dir, timeout=exec_timeout)
            res.exec_ok, res.exec_timeout = result.ok, result.timed_out
            report = validate_outputs(ctx, out_dir)
            remember()

            # The checker runs on a copy and its findings never fail a unit on their own; they are
            # repair input. A crash or a timeout inside it reports nothing at all.
            violations = properties.run_properties(checker, out_dir) if result.ok else []
            res.property_violations = len(violations)
            for v in violations:
                print(f"[solve] property violated: {v}", file=sys.stderr)

            if result.ok and report.ok and not violations:
                print(f"[solve] attempt {attempt}: ran clean, contract OK", file=sys.stderr)
                res.exec_failure = ""          # a failure a later attempt fixed is not this run's story

                # The one moment the critic is worth a call: nothing is structurally wrong and we
                # would otherwise stop here, with numbers nobody has looked at.
                if _critic_enabled() and not res.critic_verdict:
                    wrong, problems = _ask_critic(ctx, out_dir, messages, res)
                    if wrong:
                        clean_snapshot = _keep_best(out_dir, None)
                        res.critic_repaired = True
                        res.stages.append(f"critic: wrong ({len(problems)} problem(s))")
                        messages = messages + [
                            {"role": "assistant",
                             "content": f"```python\n{strip_canary(code)}\n```"},
                            {"role": "user", "content": build_repair_message(
                                critic_problems=strip_canary(critic.render(problems)))},
                        ]
                        _one_more_for_the_critic(ctx, out_dir, res, messages, exec_timeout,
                                                 clean_snapshot, deadline)
                        return
                    res.stages.append("critic: plausible")
                return

            # Printed to stderr only. A traceback quotes source lines, and generated source can carry
            # text copied from the task, so it must never reach the output directory.
            why = []
            if not result.ok:
                why.append(f"exec rc={result.returncode}"
                           f"{' TIMEOUT' if result.timed_out else ''}")
            if not report.ok:
                why.append(f"{len(report.errors)} contract error(s)")
            if violations:
                why.append(f"{len(violations)} property violation(s)")
            print(f"[solve] attempt {attempt} failed: {'; '.join(why)}", file=sys.stderr)
            # Overwritten each attempt and cleared on success, so the record holds the failure that
            # actually ended the run rather than the first one along the way.
            res.exec_failure = result.failure_text(1500) if not result.ok else ""
            # The same error twice means repair is not landing; two different errors mean it is
            # fixing one thing and breaking another. Those call for opposite responses, and until
            # now the record could not tell them apart.
            res.attempt_errors.append(_one_line_error(result, report, violations))

            if attempt >= max_attempts:
                break
            remaining = deadline - time.monotonic()
            if remaining < exec_timeout:
                # Better to stop with the current attempt's output than to be killed mid-repair.
                print(f"[solve] stopping after attempt {attempt}: "
                      f"{remaining:.0f}s left, an attempt needs about {exec_timeout:.0f}s",
                      file=sys.stderr)
                res.stopped_early = "time"
                break

            # A repair chain that has failed the same way twice does not converge. Measured three
            # separate times -- IndexError on three consecutive attempts, TIMEOUT on two,
            # TypeError on three -- each handed a precise traceback and each producing the identical
            # failure again, because every repair carries the broken script forward and the next
            # attempt is a small edit of the same wrong foundation. Independent generations do not
            # share that fate: five runs of one unit gave five different failures. So the attempt
            # with the worst odds is the one we were spending on a third patch.
            if _stuck(res.attempt_errors):
                print(f"[solve] attempt {attempt} failed the same way as the one before; "
                      f"starting fresh rather than repairing again", file=sys.stderr)
                res.stages.append("restarted a stalled repair chain")
                res.restarted = True
                messages = base_messages + [
                    {"role": "user",
                     "content": build_restart_message(strip_canary(res.attempt_errors[-1]))},
                ]
                res.repairs += 1
                continue

            # Both halves are canary-stripped. A traceback quotes lines of the generated script, and
            # if the model put the task header in a comment then the marker is sitting in that source.
            # Feeding it back would reintroduce to the model the very string we removed from the
            # prompt, and it could then copy it into the next script's output.
            messages = messages + [
                # The assembled script, not a single response: a generation may have spanned several
                # continuations, and what the model needs to see is the file that actually ran.
                {"role": "assistant", "content": f"```python\n{strip_canary(code)}\n```"},
                {"role": "user", "content": build_repair_message(
                    exec_failure="" if result.ok else strip_canary(result.failure_text(2000)),
                    contract_errors="" if report.ok else strip_canary(report.repair_text()),
                    property_violations=(strip_canary(properties.render(violations))
                                         if violations else ""),
                    failing_function=("" if result.ok else
                                      strip_canary(locate_failure(result.failure_text(4000), code))),
                    timed_out=result.timed_out,
                    time_limit=exec_timeout,
                    missing_files=strip_canary(describe_missing(
                        [f.filename for f in report.errors if f.code == "missing_file"],
                        code, str(out_dir))),
                )},
            ]
            res.repairs += 1

    finally:
        # Every exit path passes through here -- a clean return, a budget guard, an exception --
        # so the best output is restored whatever ended the loop.
        try:
            if best_dir is not None and best_score is not None:
                if _output_score(ctx, out_dir) < best_score:
                    print(f"[solve] restoring the best output seen "
                          f"({-best_score[0]} contract error(s), {best_score[1]} file(s) written)",
                          file=sys.stderr)
                    _restore(out_dir, best_dir)
        except Exception:
            pass
        finally:
            if best_dir is not None:
                shutil.rmtree(best_dir, ignore_errors=True)


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="solve", add_help=True)
    # The verb is positional because the harness passes it as the first argument. It must be
    # accepted rather than rejected as an unexpected argument (contract rule 1).
    p.add_argument("verb", nargs="?", default="solve", choices=["solve"])
    p.add_argument("--task-dir", default="/input")
    p.add_argument("--out", default=None)
    # parse_known_args, not parse_args: an argument we did not anticipate must not cost the unit,
    # and must not cost us the arguments we DID understand either -- falling back to the defaults
    # would silently solve the wrong task dir if the harness ever passes an extra flag.
    try:
        args, extra = p.parse_known_args(argv)
    except SystemExit:
        print("[solve] argument parsing failed; falling back to defaults", file=sys.stderr)
        return solve("/input", None)
    if extra:
        print(f"[solve] ignoring unrecognised argument(s): {extra}", file=sys.stderr)
    return solve(args.task_dir, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
