# SPDX-License-Identifier: MIT
"""Input staging and generated-code execution.  (Phase 0, Step 0.3)

Why staging exists
------------------
41 of the 87 practice tasks tell the agent to read its data from `/app/data/<file>`. That is where
the data sits when the organizers build *their* image for a task. At evaluation time it is our
image that runs, and the task folder is mounted read-only at `/input` instead, so `/app/data/...`
does not exist.

The prompt does tell the model the real, resolved path. But a model reading a task description
that says `/app/data/prices.csv` will often use that path anyway. Rather than depend on the model
ignoring its own instructions, staging copies each input to the path the instructions claim, so
both routes work.

Why the generated script never goes in the output directory
-----------------------------------------------------------
Each task's instructions carry a "canary" — a unique marker string the organizers plant to detect
benchmark text leaking into places it should not. One of the four admissibility checks (`g2`) scans
every text file in the output directory for these markers, and finding one fails the task outright,
regardless of whether the answer was correct.

Generated Python is a likely carrier, because models routinely copy task text into a header comment
at the top of a script. So the script is written to a scratch directory, and only the data files it
produces are allowed into the output directory.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

from .task_context import TaskContext

#: Leave the rest of the unit's wall clock for generation, staging and the repair loop (0.7).
EXEC_TIMEOUT_FRACTION = 0.5
EXEC_TIMEOUT_CAP = 900.0


@dataclasses.dataclass
class ExecResult:
    ok: bool
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    script_path: str = ""

    def failure_text(self, limit: int = 4000) -> str:
        """Diagnostic text for the repair loop (Step 0.7).

        Returns the tail rather than the head: a traceback's final lines carry the exception.
        """
        blob = (self.stderr or self.stdout or "").strip()
        return blob[-limit:]


def stage_inputs(ctx: TaskContext) -> list[tuple[str, str]]:
    """Materialise each resolved input at the path the instructions claim it is at.

    Returns the (from, to) pairs actually staged. Copies rather than symlinks: the generated code
    may open the file with any library, and a symlink introduces an avoidable failure mode.
    Never overwrites an existing file.
    """
    staged: list[tuple[str, str]] = []
    for spec in ctx.inputs:
        if not spec.resolved or not spec.declared_path:
            continue
        src = pathlib.Path(spec.resolved)
        dst = pathlib.Path(spec.declared_path)
        if not src.is_file() or dst.exists() or "/output/" in spec.declared_path:
            continue
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            staged.append((str(src), str(dst)))
        except OSError:
            # A read-only or otherwise unwritable location is not fatal: the prompt already gives
            # the resolved path, so the generated code has a working route to the data.
            continue
    return staged


def exec_timeout_for(ctx: TaskContext) -> float:
    """Execution budget, from the unit's own card.

    `[agent].timeout_sec` varies across the public set (1200/1800/2400/3600/5400) and the held-out
    units are authored separately, so the value is always read from the card rather than hardcoded.

    `QFBENCH_EXEC_TIMEOUT` overrides it. That exists for the evaluation sweep, where one script
    hanging for the full card allowance would stall a run over all 87 tasks; it is never set at
    scoring time, where each task gets its own container and its own clock.
    """
    override = os.environ.get("QFBENCH_EXEC_TIMEOUT")
    if override:
        try:
            return float(override)
        except ValueError:
            pass
    total = ctx.agent_timeout_sec or 1800.0
    return min(total * EXEC_TIMEOUT_FRACTION, EXEC_TIMEOUT_CAP)


def run_code(
    code: str,
    out_dir: str | os.PathLike[str],
    *,
    timeout: float = 900.0,
    scratch: str | os.PathLike[str] | None = None,
) -> ExecResult:
    """Run generated code in a scratch directory, with the output directory as its target.

    The child inherits the environment (so QFBENCH_SEED reaches it) plus OUTPUT_DIR, and runs with
    cwd set to the scratch dir so any stray relative write lands there rather than in the
    deliverables.
    """
    workdir = pathlib.Path(scratch) if scratch else pathlib.Path(tempfile.mkdtemp(prefix="qa-exec-"))
    workdir.mkdir(parents=True, exist_ok=True)
    script = workdir / "solution.py"
    script.write_text(code, encoding="utf-8")

    env = dict(os.environ)
    env["OUTPUT_DIR"] = str(out_dir)
    env.setdefault("QFBENCH_SEED", "0")
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    try:
        p = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(workdir), env=env, timeout=timeout,
            capture_output=True, text=True, errors="replace",
        )
        return ExecResult(p.returncode == 0, p.returncode, p.stdout, p.stderr,
                          script_path=str(script))
    except subprocess.TimeoutExpired as e:
        return ExecResult(
            False, -1,
            (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or ""),
            f"TIMEOUT after {timeout:.0f}s",
            timed_out=True, script_path=str(script),
        )
    except Exception as e:                       # the runner must not raise into the caller
        return ExecResult(False, -1, "", f"{type(e).__name__}: {e}", script_path=str(script))
