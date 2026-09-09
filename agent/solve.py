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
import json
import os
import sys
import time
import traceback

from .execute import exec_timeout_for, run_code, stage_inputs
from .llm import LLMError, chat
from .prompt import build_prompt, build_repair_message, extract_code, strip_canary
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
    deliverables: int = 0
    non_empty: int = 0
    attempts: int = 0                # generation attempts made, including the first
    repairs: int = 0                 # follow-up attempts driven by a failure signal
    stopped_early: str = ""          # "time" when the wall-clock guard cut repair short
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


def _generate_and_run(ctx: TaskContext, out_dir, res: SolveResult) -> None:
    """Generate, execute, and repair until the output satisfies the contract or a budget runs out.

    Bounded by three things, whichever binds first:

      * attempts — `QFBENCH_MAX_ATTEMPTS`, default 3. Later rounds have sharply diminishing
        returns, and each one costs a model call plus an execution.
      * wall clock — a task is scored on what exists when its container is killed, so an attempt
        is only started if there is plausibly time to finish it. Running out mid-repair with
        nothing written would be strictly worse than stopping and keeping the previous attempt.
      * tokens — enforced inside the client, which raises rather than overspending.

    Failure signals come from two places and both are fed back: a traceback when the script raised,
    and the contract findings when it ran but produced the wrong files. The second matters because
    nothing crashed, so the model has no way to know anything is wrong unless it is told.
    """
    max_attempts = _int_env("QFBENCH_MAX_ATTEMPTS", 3)
    total_budget = ctx.agent_timeout_sec or 1800.0
    # Leave a margin for parsing, staging, validation, backfill and container teardown.
    deadline = time.monotonic() + total_budget * 0.8
    exec_timeout = exec_timeout_for(ctx)

    staged = stage_inputs(ctx)
    for src, dst in staged:
        print(f"[solve] staged {src} -> {dst}", file=sys.stderr)

    messages = build_prompt(ctx, out_dir)

    for attempt in range(1, max_attempts + 1):
        res.attempts = attempt
        try:
            reply = chat(messages, max_tokens=8000)
        except LLMError as e:
            res.llm_error = str(e)
            print(f"[solve] attempt {attempt}: model call failed: {e}", file=sys.stderr)
            return
        res.tokens_in += reply.usage.input_tokens
        res.tokens_out += reply.usage.output_tokens
        print(f"[solve] attempt {attempt}: model={reply.model} mode={reply.mode} "
              f"in={reply.usage.input_tokens} out={reply.usage.output_tokens}", file=sys.stderr)

        code = extract_code(reply.text)
        if not code:
            print(f"[solve] attempt {attempt}: no code block in reply", file=sys.stderr)
            return
        res.generated = True

        result = run_code(code, out_dir, timeout=exec_timeout)
        res.exec_ok, res.exec_timeout = result.ok, result.timed_out
        report = validate_outputs(ctx, out_dir)

        if result.ok and report.ok:
            print(f"[solve] attempt {attempt}: ran clean, contract OK", file=sys.stderr)
            return

        # Printed to stderr only. A traceback quotes source lines, and generated source can carry
        # text copied from the task, so it must never reach the output directory.
        why = []
        if not result.ok:
            why.append(f"exec rc={result.returncode}"
                       f"{' TIMEOUT' if result.timed_out else ''}")
        if not report.ok:
            why.append(f"{len(report.errors)} contract error(s)")
        print(f"[solve] attempt {attempt} failed: {'; '.join(why)}", file=sys.stderr)

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

        # Both halves are canary-stripped. A traceback quotes lines of the generated script, and
        # if the model put the task header in a comment then the marker is sitting in that source.
        # Feeding it back would reintroduce to the model the very string we removed from the
        # prompt, and it could then copy it into the next script's output.
        messages = messages + [
            {"role": "assistant", "content": strip_canary(reply.text)},
            {"role": "user", "content": build_repair_message(
                exec_failure="" if result.ok else strip_canary(result.failure_text(2000)),
                contract_errors="" if report.ok else strip_canary(report.repair_text()),
            )},
        ]
        res.repairs += 1


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
