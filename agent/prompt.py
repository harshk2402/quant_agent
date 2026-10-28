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

2.  **The invariants of the task's own category are stated before the task is.**
    Every unit card declares one of ten categories, and the fourth grading gate applies
    category-specific sanity checks — put-call parity for an option pricer, survival probability
    falling with time for a credit curve, expected shortfall at least VaR for a risk report. Those
    relationships are also where a capable model goes wrong for reasons that have nothing to do
    with capability: a theta per year where the task wanted per day, a sigma as 20 rather than
    0.20, a Sharpe computed on total instead of excess returns. Saying which convention is expected
    costs nothing, whereas discovering it from a failed validation costs an attempt and part of a
    25-request budget.

3.  **The model is shown what is INSIDE each input, not just where it is.**
    Resolving the path is half the problem. Measured on seven units whose scripts assemble and run
    to hundreds of lines, five of six crashes were a KeyError on a key the model had guessed --
    `pair_conventions["EURUSD"]` where the file nests everything under `pairs`, `inst["issue_date"]`
    where no such field exists. One wrong guess discards the whole run, and the traceback names the
    missing key without revealing the real schema, so a repair attempt guesses again. The columns
    and key structure therefore go in the prompt, with sample values, because whether
    `maturity_date` holds "2033-05-15" or 3.5 decides whether the script works.

4.  **The model is given the RESOLVED input paths, not the ones in the task text.**
    41 of the 87 tasks describe their data as living at `/app/data/...`. That is true when the
    organizers build their own image for a task, but at evaluation time our image runs and the task
    folder is mounted at `/input` instead. Following the task text literally therefore sends the
    generated code looking in a directory that does not exist, so the paths the parser actually
    resolved are stated explicitly and marked as authoritative.
"""

from __future__ import annotations

import ast
import os
import re

from .invariants import guidance_for
from .preview import describe_inputs
from .task_context import TaskContext, render_output_contract

# The canary line itself, and the banner that always accompanies it.
_CANARY_LINE = re.compile(
    r"^.*(?:canary|BENCHMARK DATA SHOULD NEVER APPEAR).*$", re.IGNORECASE | re.MULTILINE
)
_GUID = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                   r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")

_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def _disabled(name: str) -> bool:
    """Switches for A/B measurement, not for production behaviour.

    Both additions below are hypotheses: they are meant to help, and on the representative set four
    units that passed in September stopped passing, three of them having made a single model call
    with no stitching involved -- so the prompt is the only thing that changed for them. A block
    that cannot be turned off cannot be measured, and "it should help" is not evidence.
    """
    return os.environ.get(name, "").strip() == "1"


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
    "exception scores nothing.\n"
    # Ordering, because of where the crashes actually landed. Several scripts computed every
    # number correctly and then died in the plotting tail -- a (9,13) against (13,9) mesh in
    # plot_surface, a matplotlib shape error -- with the results still only in memory. A failure
    # in the last ten lines threw away four hundred correct ones. Writing the data files BEFORE
    # anything optional means a crash or a timeout in the tail leaves the deliverables on disk,
    # and the output kept from an attempt is the best any attempt produced.
    "- ORDER OF WORK: compute the core results first, then IMMEDIATELY write every required "
    "data file (CSV, JSON, Parquet). Only after all of those are on disk, produce plots and "
    "anything optional. Wrap each plot in its own try/except so a plotting failure can never "
    "stop the script or lose a data file already written.\n"
    "- Keep the script lean: no argument parsing, no logging, no re-validation of inputs, no "
    "plots beyond those the task requires. Every extra line is another place to fail."
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

    # What is in those files. Placed directly after the paths, so a name and its contents are read
    # together rather than a page apart.
    previews = "" if _disabled("QFBENCH_NO_INPUT_PREVIEW") else describe_inputs(ctx)
    if previews:
        parts.append(
            "## What each input actually contains\n"
            "Read from these structures as shown -- do not assume column or key names that are "
            "not listed here. Values are real samples, abbreviated; row counts and nesting are "
            "accurate, but a long file is previewed only in part.\n\n" + previews)

    # --- the output contract: the exact filenames to write ------------------------------------
    # Every task specifies its own deliverable names, and a file written under the wrong name
    # scores zero no matter how correct the numbers inside it are. This block is therefore the
    # single most important thing the model is told.
    parts.append(f"## Output directory\nWrite all deliverables into: `{out_dir}`\n\n"
                 + render_output_contract(ctx))

    for spec in ctx.outputs:
        if spec.schema_text:
            parts.append(f"## Shape of `{spec.filename}`\n{strip_canary(spec.schema_text)}")

    # --- the domain invariants for this category ----------------------------------------------
    # Placed before the task text rather than after it, so these are read as constraints on the
    # answer rather than as an afterthought about it.
    if not _disabled("QFBENCH_NO_INVARIANT_GUIDANCE"):
        parts.append("## Financial invariants your answer must satisfy\n"
                     "The task description below takes precedence over everything in this section: "
                     "where it states a convention, a formula or a tolerance, follow it exactly and "
                     "ignore anything here that disagrees. These are defaults for what it leaves "
                     "unsaid, and the relationships hold for any correct result whatever method you "
                     "use.\n"
                     + guidance_for(ctx.category))

    # --- the task itself, canary-stripped -----------------------------------------------------
    parts.append("## Task description (verbatim, for the finance — paths in it may be stale)\n"
                 + strip_canary(ctx.instruction).strip())

    # The output contract, again, as the last thing read.
    #
    # It appears earlier too, but earlier is the middle of a long prompt, and this is the single
    # constraint that costs a unit outright: the starter pack's own warning is that the output
    # contract is per-unit and generalising from one unit will cost you a submission. A file
    # written under the wrong name scores zero however right the numbers are, and unlike the
    # finance it is free to get right. Stated positively and last, because a prohibition at the
    # top of a prompt is the easiest thing in it to forget.
    if ctx.outputs:
        names = "\n".join(f"- `{spec.filename}`" for spec in ctx.outputs)
        parts.append(f"## Before you finish\nWrite exactly these files, with exactly these "
                     f"names, into `{out_dir}`:\n{names}\n\n"
                     f"Write every one of them. Use these names character for character.")

    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "\n\n".join(parts)}]


_FRAME = re.compile(r'File "[^"]*solution\.py", line (\d+), in (\S+)')


def locate_failure(traceback_text: str, code: str) -> str:
    """Quote the function the traceback blames, so the repair instruction is local.

    A traceback is the least ambiguous signal we ever get -- exact file, line and function -- and
    until now we threw away everything but the text. Measured: a run was handed
    `IndexError ... line 148, in crank_nicolson_psor` and failed to fix it on two further attempts.
    The likely reason is that we ask for the ENTIRE corrected script, so each repair regenerates
    five hundred lines and re-rolls every decision in them while fixing the one that broke.

    Quoting the offending function does not change what we ask for -- a whole script, because
    splicing a patch into generated code is a fragile business we decided against -- but it changes
    the instruction from "fix this program" to "fix this function and reproduce the rest".
    """
    frames = _FRAME.findall(traceback_text or "")
    if not frames or not code.strip():
        return ""
    line_no, func = frames[-1]          # the innermost frame in our own file is the culprit
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return ""
    lines = code.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func:
            start, end = node.lineno, getattr(node, "end_lineno", node.lineno)
            body = "\n".join(lines[start - 1:end])
            return (f"The failure is inside `{func}`, at line {line_no}:\n\n"
                    f"```python\n{body}\n```\n\n"
                    f"Change what is wrong in THIS function. Reproduce the rest of the script "
                    f"unchanged -- the other parts ran, and rewriting them risks breaking what "
                    f"already works.")
    return (f"The failure is at line {line_no}, in `{func}`. Change that, and reproduce the rest "
            f"of the script unchanged.")


TIMEOUT_ADVICE = (
    "Your script was killed for running too long: it exceeded {limit:.0f} seconds and produced "
    "nothing usable.\n\n"
    "A timeout is not a small problem to tune around -- the run is lost entirely, and the whole "
    "roster shares one clock. The usual cause in this kind of task is a Python `for` loop over a "
    "large grid or over many paths, where each iteration does a little arithmetic. Interpreted "
    "loops over hundreds of thousands of points take minutes; the same arithmetic as numpy array "
    "operations takes a fraction of a second.\n\n"
    "Rewrite the slow part:\n"
    "- replace element-by-element loops with whole-array numpy operations\n"
    "- solve each tridiagonal system with `scipy.linalg.solve_banded` rather than looping, and "
    "iterate a PSOR/SOR sweep over arrays rather than over single elements\n"
    "- build grids with `np.linspace` and index them with slices, never by appending in a loop\n"
    "- keep every grid size and iteration count the task specifies -- make the work faster, do "
    "not make it smaller. A fast wrong answer scores the same as a slow one.\n\n"
    "If one part genuinely must loop, bound its iterations explicitly so it cannot run away."
)


def build_repair_message(exec_failure: str = "", contract_errors: str = "",
                         property_violations: str = "", failing_function: str = "",
                         timed_out: bool = False, time_limit: float = 0.0,
                         critic_problems: str = "", missing_files: str = "") -> str:
    """The follow-up turn telling the model what went wrong with its previous script.

    Two kinds of failure reach here and they need different instructions:

      * the script raised or timed out — the traceback names the line, and the fix is usually local;
      * the script ran but its output does not satisfy the contract — nothing crashed, so the model
        has no reason to suspect a problem unless it is told explicitly.

    The script is asked for in full each time rather than as a patch. Partial edits invite the model
    to reference code it can no longer see, and a whole script is cheap at these sizes.
    """
    parts = ["Your script did not succeed. Fix it and return the COMPLETE corrected script."]

    if timed_out:
        # A timeout carries no traceback, so every other signal we normally pass is empty. Measured:
        # a unit lost two consecutive attempts to 120-second timeouts, and the repair message told
        # the model only that execution "failed" -- so it regenerated something equally slow, twice.
        parts.append(TIMEOUT_ADVICE.format(limit=time_limit))
    elif exec_failure:
        parts.append("It failed when executed. The error was:\n\n"
                     f"```\n{exec_failure}\n```")
    if failing_function and not timed_out:
        parts.append(failing_function)
    if contract_errors:
        parts.append(
            "The required output files were not produced correctly:\n\n"
            f"{contract_errors}\n\n"
            "These are checked mechanically before any grading of the numbers, so a task fails "
            "on them regardless of whether the finance is right.")

    if missing_files:
        parts.append(missing_files)

    if critic_problems:
        # The only channel that is about the ANSWER rather than the run: nothing crashed, every
        # file is present, and the numbers still cannot be right.
        parts.append(critic_problems)

    if property_violations:
        # A third kind of failure, and the only one that is about the ANSWER rather than the run:
        # nothing crashed and every file is present, but a relationship the model itself said must
        # hold does not.
        parts.append(property_violations)

    parts.append(
        "Return one ```python block containing the entire corrected script. Keep the parts that "
        "worked, change what caused the failure, and do not explain the fix.")
    return "\n\n".join(parts)


def build_continuation_message(prefix: str, anchor_lines: int = 15,
                               missing: str = "") -> str:
    """Ask for the rest of a script that was cut off at the output ceiling.

    The model has the script so far in the preceding turn, so this only has to say where to resume
    and what not to do. Two of those instructions exist because models ignore the polite version:
    a continuation gets wrapped in a code fence, and the last line or two gets re-emitted. Both are
    also handled mechanically in `assemble.join`, since asking is not a control.

    The anchor is quoted verbatim rather than described, because "continue where you left off" is
    ambiguous once we have trimmed the partial tail away -- the model's idea of where it stopped and
    ours are different by however many lines did not parse.
    """
    lines = prefix.rstrip("\n").splitlines()
    tail = "\n".join(lines[-anchor_lines:])
    last = lines[-1] if lines else ""
    prompt = (
        "Your script was cut off because it reached the output limit. It currently ends with "
        "these lines:\n\n"
        f"```python\n{tail}\n```\n\n"
        "Continue from there. Output ONLY the remaining code:\n"
        f"- the final line above (`{last.strip()}`) may have been cut off mid-way. Begin your "
        "answer by writing that ONE line again, complete, and then carry on. Only that line.\n"
        "- do not repeat anything before it, and do not restate the imports\n"
        "- do not wrap your answer in a code fence and do not explain anything\n"
        "- match the indentation of the code above\n"
        "- keep it compact so the rest fits: no commentary, no defensive re-checks, no examples\n"
        "- finish the whole task, including writing every required output file"
    )
    if missing:
        # Used when a static check found a name read but never assigned: the script has been cut
        # back to the break, and the model needs to know what the lost code was supposed to build,
        # or it may simply reproduce the same gap.
        return (prompt + "\n\nNote: in the version you wrote, the code below this point used "
                f"`{missing}` without ever assigning it -- the line that created it was lost. "
                "Make sure your continuation defines it before using it.")
    return prompt


#: What to ask for in the planning call.
#:
#: This asks for DETAIL, and that is the whole point. An earlier version asked for "at most 12
#: numbered steps, one line each" and got exactly that -- 1,802 characters, finishing at 868 of
#: its 4,000 tokens. The implementation call then reasoned 3,907 tokens and produced no code,
#: because a twelve-line outline does not contain the decisions: the model still had to derive
#: them, and derived them right through its budget again.
#:
#: The version that worked was an accident. A 15,183-character prose write-up -- formulas,
#: conventions, discretisation, file schemas -- cut the implementation call's reasoning from
#: ~3,900 tokens to 286 and produced 11,659 characters of Python on a unit that had produced none.
#: The deliberation has to happen somewhere; this is a request for it to happen HERE, where the
#: output is kept, rather than in the call that needs its budget for code.
PLAN_INSTRUCTION = (
    "Before writing any code, work out the approach in full. This is the only time you will think "
    "about this problem: the implementation step will follow what you write here literally, so "
    "every decision must already be made by the end of it.\n\n"
    "FIRST, a conventions ledger. Go through the task text and quote, VERBATIM, every rule it "
    "states that changes a number: the annualisation factor, the compounding method (simple, "
    "log, geometric), the day count, every window length and whether its ends are inclusive, "
    "which rows or dates are included or excluded, the risk-free rate and how it enters, "
    "rounding, units, sign conventions, tolerances. One line each: the rule as the task wrote it, "
    "then how you will apply it. Where the task is silent on something that matters, say "
    "UNSTATED and name the default you will use. Do not substitute a familiar default for a "
    "stated one -- if the task says 250 and you would normally use 252, the answer is 250.\n\n"
    "THEN set out, in as much detail as you need:\n"
    "- the formula or method for each quantity the task asks for, written out explicitly\n"
    "- every convention from the ledger applied where it bites, and each UNSTATED one decided\n"
    "- any discretisation, grid, iteration or tolerance, with its parameters chosen\n"
    "- for EVERY window, lookback, slice or date range: one worked example with the actual "
    "indices written out -- e.g. 'for t=12 the formation window is indices 0 through 10, which "
    "is 11 values' -- and the count. Write the indices, not the formula. An off-by-one in a "
    "window is invisible in a formula and obvious in an example.\n"
    "- for each output file: its exact name, its columns in order, and what one row means\n"
    "- the edge cases you will have to handle, and what you will do about each\n"
    "- the order the steps run in, and which step produces which file\n\n"
    "Prose or numbered steps, whichever suits. Do NOT write code -- no snippets, no function "
    "bodies. Be specific: name the numbers and the formulas rather than describing them."
)


def build_plan_prompt(messages: list[dict]) -> list[dict]:
    """Turn the task prompt into a request for a plan.

    WHY PLAN AT ALL. Measured on this model: on a hard unit the first response spends its entire
    4,000-token budget deliberating and returns empty content, and a follow-up that forbids
    reasoning simply moves the same deliberation into `content` as 15,000 characters of prose. The
    thinking is not optional and it is not free -- the only question is whether we keep it.

    So the plan call is not extra work. It is the deliberation we were already paying for, bought
    once, in a form we can put into the NEXT call's input, which is uncapped. The code call then
    starts with the approach settled and the whole response budget available for code.
    """
    return list(messages) + [{"role": "user", "content": PLAN_INSTRUCTION}]


def build_implement_message(plan: str, max_chars: int = 4000) -> str:
    """Ask for the script, with the approach already decided.

    The plan is quoted back rather than summarised: paraphrasing it would be another judgement
    call, and the point is that the decisions are already made.
    """
    plan = plan.strip()
    if len(plan) > max_chars:
        plan = plan[:max_chars].rsplit("\n", 1)[0] + "\n(plan truncated)"
    return (
        "This is the approach you set out:\n\n"
        f"{plan}\n\n"
        "It is settled -- do not reconsider it, and do not explain it. Write the complete script "
        "now, following it.\n"
        "Reply with ONE ```python block and nothing else, starting with ```python as the very "
        "first characters of your reply."
    )


def build_properties_prompt(messages: list[dict]) -> list[dict]:
    """Ask for a checker for the answer. The instruction itself lives in `properties.py`, beside
    the code that runs what comes back."""
    from .properties import PROPERTIES_INSTRUCTION
    return list(messages) + [{"role": "user", "content": PROPERTIES_INSTRUCTION}]


def describe_missing(missing: list[str], code: str, out_dir: str = "", context: int = 3) -> str:
    """Tell the two kinds of missing deliverable apart.

    `missing_file` is our commonest contract error and today it says the same thing in both cases:
    "required deliverable was not written". But the two cases need opposite fixes.

      * The filename appears in the script. The write exists, so either something earlier raised
        before reaching it, or it is being written somewhere other than the output directory --
        which matters here, because many units' instructions name a path that is not where the
        harness actually binds the output.
      * The filename appears nowhere. The model forgot the deliverable entirely, and the fix is to
        add it, not to debug anything.
    """
    if not missing or not code.strip():
        return ""
    lines = code.splitlines()
    parts: list[str] = []
    for name in missing:
        hits = [i for i, line in enumerate(lines) if name in line]
        if not hits:
            parts.append(f"- `{name}`: this filename does not appear anywhere in your script. "
                         f"It was never written. Add the code that produces it.")
            continue
        i = hits[0]
        lo, hi = max(0, i - context), min(len(lines), i + context + 1)
        snippet = "\n".join(lines[lo:hi])
        extra = f" (and {len(hits) - 1} other place(s))" if len(hits) > 1 else ""
        parts.append(
            f"- `{name}`: your script DOES reference it, around line {i + 1}{extra}:\n\n"
            f"```python\n{snippet}\n```\n"
            f"  So either execution never reached this point, or it is being written somewhere "
            f"other than the output directory"
            + (f" — which must be `{out_dir}`." if out_dir else "."))
    return ("Some required files are missing, and the reason differs between them:\n\n"
            + "\n".join(parts))


def build_restart_message(last_error: str) -> str:
    """Start again from the task, having abandoned a repair chain that was not converging.

    Measured three separate times: IndexError on three consecutive attempts, TIMEOUT on
    two, TypeError on three -- each attempt handed a precise traceback and each producing the same
    failure again. A repair carries the broken script forward, so every attempt is a small edit of
    the last and works from the same wrong foundation.

    Independent generations do not share that fate: five runs of one unit produced five different
    failure modes. So this turn deliberately shows the model NO previous code -- only what went
    wrong, so it can avoid that approach rather than inherit it.
    """
    return (
        "Previous attempts at this task kept failing in the same way:\n\n"
        f"```\n{last_error}\n```\n\n"
        "Start again from scratch. Do not try to repair the earlier script -- write a new one, "
        "using a different approach for whatever produced that failure. A simpler method that "
        "runs and gives an approximate answer is worth far more than a sophisticated one that "
        "does not run.\n\n"
        "Reply with ONE ```python block containing the complete script and nothing else."
    )


def build_no_output_message(spent_thinking: bool) -> str:
    """Ask again when a response came back with no usable code.

    This exists because of a measured failure, and a specific one. Nemotron 3 is a reasoning model
    and its internal thinking is charged against the SAME per-request output budget as its answer.
    On eight of fifteen units it spent the entire 4,000 tokens reasoning and returned empty
    content -- three times each, because the loop retried the identical prompt and got the
    identical result. Twelve thousand tokens per unit, not one line of code.

    So this turn is not a continuation: there is no prefix to resume from. It is the same task with
    one instruction changed -- answer first, think less. The model's own reasoning is deliberately
    NOT fed back, even though it is available and often trails off part-way toward a solution: it
    is draft rather than answer, and returning it invites exactly the behaviour we are trying to
    interrupt.
    """
    why = ("Your previous response used its entire output budget on internal reasoning and "
           "returned no code."
           if spent_thinking else
           "Your previous response did not contain a Python script.")
    return (
        f"{why}\n\n"
        "Reply with the script and nothing else. Start your reply with ```python as the very "
        "first characters — no analysis, no plan, no explanation before or after it.\n"
        "Think as little as possible before writing: a straightforward, direct implementation "
        "that runs and writes the required files is worth far more than a considered one that "
        "never gets written. If you cannot fit the whole task, write the part that produces the "
        "output files first."
    )


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
