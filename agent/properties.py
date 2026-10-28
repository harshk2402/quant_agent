# SPDX-License-Identifier: MIT
"""Ask the model what must be true of a correct answer, then check it.

WHY. The dev run came back with 81 of 86 units labelled `domain_gate_failed`: the code ran, the
files were well-formed, and the numbers were wrong. Nothing we check catches that. The structural
layer is finished -- output contracts, canaries, truncation, undefined names -- and it was never
going to reach this, because being wrong is not a structural property.

The hand-written invariants in `invariants.py` were the first attempt and they have a ceiling that
one measured failure shows exactly. A unit reported an american put, american call, european put
and european call, with and without dividends, as EXACTLY 0.0 -- the script had reversed its time
axis and read the answer off the wrong end. Not one invariant fired, because they key on column
names like `price` and `delta` and the file's keys were `american_put` and `european_call`. A rule
written in advance cannot know what a task will call things.

A model asked "what must be true here?" does not need to recognise a name. It knows an
at-the-money option with a year to run is worth more than nothing.

THE ASYMMETRY THIS LEANS ON. Stating what must be true is a much easier task than being right. The
same model that produced those zeros wrote 534 lines of correct Crank-Nicolson, correct PSOR and
correct dividend handling to get there -- it failed on an index, not on the finance. Asking it for
properties uses it for the half it is good at.

ORDER MATTERS: the properties are requested BEFORE the implementation exists. Asked afterwards,
they describe whatever got written -- a checker generated from buggy code will faithfully assert
the bug. Asked from the task alone, they describe the task.

WHAT KEEPS IT SAFE. A generated checker is not trusted:

  * it runs against a COPY of the output, so it cannot write into the real deliverables;
  * it gets a short timeout, and a crash or a timeout is ignored completely rather than reported;
  * its findings are repair input, never a verdict -- nothing here fails a unit on its own;
  * and the best output seen is already snapshotted, so a repair driven by a false violation
    cannot leave us worse off than the attempt that triggered it.

That last point is what makes this worth trying at all. Every false alarm this project has shipped
cost attempts and sometimes a passing result; with rollback in place the downside is bounded to
wasted requests.
"""

from __future__ import annotations

import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

#: Output budget for the properties call. Smaller than the code budget: a dozen checks is short,
#: and the model has the plan in front of it already.
MAX_TOKENS = 2000

#: How long the generated checker may run. It reads a handful of files and compares numbers; a
#: checker that takes longer than this is doing something it was not asked to do.
TIMEOUT_SEC = 60.0

#: Violations beyond this are not reported. A checker that finds fifty problems has misunderstood
#: the task, and a wall of text would crowd the real failure out of the repair message.
MAX_VIOLATIONS = 8

_VIOLATION = re.compile(r"^\s*VIOLATION:\s*(.+?)\s*$", re.MULTILINE)


def build_properties_prompt(messages: list[dict]) -> list[dict]:
    """Ask for a checker, given the task and whatever plan has been made."""
    return list(messages) + [{"role": "user", "content": PROPERTIES_INSTRUCTION}]


PROPERTIES_INSTRUCTION = (
    "Before the implementation, write a CHECKER for the answer.\n\n"
    "This is a separate Python script whose only job is to read the output files and report "
    "anything that cannot be true of a correct answer. It does not solve the task.\n\n"
    "Requirements:\n"
    "- it takes the output directory as sys.argv[1] and reads the deliverables from there\n"
    "- for each property that FAILS, print one line: VIOLATION: <what is wrong, with the numbers>\n"
    "- print nothing at all if everything checks out\n"
    "- it must never write or delete a file, and must exit 0 whatever it finds\n"
    "- wrap each check in try/except so one failure cannot stop the others\n"
    "- if a file is missing, say so and move on\n\n"
    "Check the things that must hold for ANY correct answer to this task, not the things your "
    "implementation happens to do. Good properties are relationships and bounds that follow from "
    "the finance or from the task statement: a quantity that cannot be negative, an ordering that "
    "must hold between two outputs, a value that must lie between two others, a total that must "
    "match a sum, a result that cannot be zero because the inputs make it impossible.\n\n"
    "The general invariants for this category are stated earlier in this conversation, and they "
    "are ALREADY checked automatically -- restating them here adds nothing. Use them as the model "
    "for what a good property looks like (a relationship, a bound, an ordering, written with the "
    "actual quantities), and then write the ones they cannot cover: the properties specific to "
    "THIS task's numbers, conventions and inputs. Those are the ones only you can state, because "
    "a rule written in advance does not know that these options are at the money, or that this "
    "curve starts at a particular level, or how many rows the input has.\n\n"
    "Do not check that a file parses or that a column exists -- that is checked elsewhere and is "
    "not what you are for.\n\n"
    "Reply with ONE ```python block and nothing else."
)


def run_properties(checker: str, out_dir, timeout: float = TIMEOUT_SEC) -> list[str]:
    """Run a generated checker against a copy of the output. Returns the violations it reported.

    Returns [] for anything that goes wrong -- a crash, a timeout, a checker that will not import.
    An unverifiable answer is not a wrong one, and our own verification must never be the reason a
    unit fails.
    """
    if not checker.strip():
        return []
    work = pathlib.Path(tempfile.mkdtemp(prefix="qa-props-"))
    try:
        # A COPY, so a checker that ignores the "never write" instruction cannot touch the real
        # deliverables. It is generated code; the instruction is not a control.
        mirror = work / "output"
        shutil.copytree(out_dir, mirror, dirs_exist_ok=True)
        script = work / "checker.py"
        script.write_text(checker, encoding="utf-8")
        try:
            done = subprocess.run([sys.executable, str(script), str(mirror)],
                                  capture_output=True, text=True, timeout=timeout, cwd=str(work))
        except (subprocess.TimeoutExpired, OSError):
            return []
        found = _VIOLATION.findall(done.stdout or "")
        return [v[:300] for v in found[:MAX_VIOLATIONS]]
    except Exception:
        return []
    finally:
        shutil.rmtree(work, ignore_errors=True)


def render(violations: list[str]) -> str:
    """The violations, phrased for the repair turn."""
    lines = "\n".join(f"- {v}" for v in violations)
    return (
        "You also wrote a checker for this task, and it reports that the answer violates "
        f"properties you yourself said must hold:\n\n{lines}\n\n"
        "These are statements about the ANSWER, not about the code running. Work out which "
        "computation produces each wrong value and fix that, rather than adjusting the output to "
        "satisfy the check."
    )
