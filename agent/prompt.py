# SPDX-License-Identifier: MIT
"""Prompt construction and code extraction.  (Phase 0, Step 0.3)

Two parts of this prompt affect whether a task scores at all, rather than merely how well the
model answers. Both are worth understanding before changing anything here:

1.  **The canary marker is removed before the model ever sees it.**
    A "canary" is a unique random string the organizers plant in task text so they can detect that
    benchmark material has leaked somewhere it should not be. 66 of the 87 practice tasks carry one
    in the opening lines of `instruction.md`, and one of the four admissibility checks (`g2`) fails
    any submission whose output files contain it — regardless of whether the answer was right.

    The realistic way it leaks is that the model copies the task header into a comment at the top
    of the script it writes. A model that is never shown the marker cannot reproduce it, which
    makes stripping it here the cheapest of the three defences we have.

2.  **The model is given the RESOLVED input paths, not the ones in the task text.**
    41 of the 87 tasks describe their data as living at `/app/data/...`. That is true when the
    organizers build their own image for a task, but at evaluation time our image runs and the task
    folder is mounted at `/input` instead. Following the task text literally therefore sends the
    generated code looking in a directory that does not exist, so the paths the parser actually
    resolved are stated explicitly and marked as authoritative.
"""

from __future__ import annotations

import os
import re

from .task_context import TaskContext, render_output_contract

# The canary line itself, and the banner that always accompanies it.
_CANARY_LINE = re.compile(
    r"^.*(?:canary|BENCHMARK DATA SHOULD NEVER APPEAR).*$", re.IGNORECASE | re.MULTILINE
)
_GUID = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                   r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")

_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def strip_canary(text: str) -> str:
    """Remove canary markers and any bare GUID from text bound for the model.

    Belt and braces: the marker lines go first, then any surviving GUID-shaped token is redacted,
    so a canary written in an unexpected format still cannot reach the model.
    """
    text = _CANARY_LINE.sub("", text)
    return _GUID.sub("[redacted]", text)


SYSTEM = (
    "You are a quantitative-finance engineer. You write a single, complete, self-contained "
    "Python 3.13 script that solves the task and writes the required output files.\n"
    "\n"
    "Rules:\n"
    "- Output ONE ```python code block and nothing else. No explanation before or after.\n"
    "- The script runs offline with no network. Available libraries: numpy, pandas, scipy, "
    "pyarrow, statsmodels, scikit-learn, arch, polars, matplotlib, seaborn, plotly, openpyxl, "
    "numba, backtrader, TA-Lib, plus the standard library. Import nothing else — there is no "
    "package index at run time.\n"
    "- Read inputs ONLY from the exact paths given. Do not guess filenames or extensions.\n"
    "- Write every required deliverable to the output directory, with exactly the given "
    "filenames. Never write reward.json.\n"
    "- Never print, copy or embed the task text into any output file — write only computed "
    "values.\n"
    "- Seed every source of randomness from the QFBENCH_SEED environment variable "
    "(default 0) so reruns reproduce.\n"
    "- Handle the task's edge cases rather than crashing; a wrong number still scores, an "
    "exception scores nothing."
)


def build_prompt(ctx: TaskContext, out_dir: str | os.PathLike[str]) -> list[dict]:
    """Assemble the messages for one generation call."""
    parts: list[str] = []

    if ctx.title:
        parts.append(f"# Task: {ctx.title}")

    # --- inputs: resolved paths win over whatever the prose claims ---------------------------
    if ctx.inputs:
        lines = ["## Input files (READ FROM THESE EXACT PATHS)"]
        for i in ctx.inputs:
            where = i.resolved or i.declared_path or f"(unresolved: {i.filename})"
            note = ""
            if i.declared_path and i.resolved and i.declared_path != i.resolved:
                # Stated explicitly, because the instruction text below still names the old path.
                note = f"   (the instructions call this {i.declared_path})"
            lines.append(f"- `{where}`  [{i.fmt}]{note}")
        parts.append("\n".join(lines))

    # --- the output contract: the exact filenames to write ------------------------------------
    # Every task specifies its own deliverable names, and a file written under the wrong name
    # scores zero no matter how correct the numbers inside it are. This block is therefore the
    # single most important thing the model is told.
    parts.append(f"## Output directory\nWrite all deliverables into: `{out_dir}`\n\n"
                 + render_output_contract(ctx))

    for spec in ctx.outputs:
        if spec.schema_text:
            parts.append(f"## Shape of `{spec.filename}`\n{strip_canary(spec.schema_text)}")

    # --- the task itself, canary-stripped -----------------------------------------------------
    parts.append("## Task description (verbatim, for the finance — paths in it may be stale)\n"
                 + strip_canary(ctx.instruction).strip())

    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "\n\n".join(parts)}]


def build_repair_message(exec_failure: str = "", contract_errors: str = "") -> str:
    """The follow-up turn telling the model what went wrong with its previous script.

    Two kinds of failure reach here and they need different instructions:

      * the script raised or timed out — the traceback names the line, and the fix is usually local;
      * the script ran but its output does not satisfy the contract — nothing crashed, so the model
        has no reason to suspect a problem unless it is told explicitly.

    The script is asked for in full each time rather than as a patch. Partial edits invite the model
    to reference code it can no longer see, and a whole script is cheap at these sizes.
    """
    parts = ["Your script did not succeed. Fix it and return the COMPLETE corrected script."]

    if exec_failure:
        parts.append("It failed when executed. The error was:\n\n"
                     f"```\n{exec_failure}\n```")
    if contract_errors:
        parts.append(
            "The required output files were not produced correctly:\n\n"
            f"{contract_errors}\n\n"
            "These are checked mechanically before any grading of the numbers, so a task fails "
            "on them regardless of whether the finance is right.")

    parts.append(
        "Return one ```python block containing the entire corrected script. Keep the parts that "
        "worked, change what caused the failure, and do not explain the fix.")
    return "\n\n".join(parts)


def extract_code(reply: str) -> str | None:
    """Pull the Python out of a model reply.

    Prefers a fenced block, falling back to the raw reply when it appears to be a script. The
    fallback is deliberate: failing to extract forfeits the unit, whereas an incorrect extraction
    costs only one execution attempt.
    """
    blocks = _FENCE.findall(reply or "")
    if blocks:
        return max(blocks, key=len).strip()
    text = (reply or "").strip()
    if re.search(r"^\s*(import|from|def |#!)", text, re.MULTILINE):
        return text
    return None
