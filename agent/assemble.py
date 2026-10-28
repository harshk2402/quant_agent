# SPDX-License-Identifier: MIT
"""Building one script out of several bounded model responses.

WHY THIS EXISTS. The House route caps a single response at 4,000 output tokens. Measured on the
September baseline, the median generated script was already ~3,800 tokens per call and 40 of 87
units averaged above 4,000, with seven pinned within four tokens of the then-ceiling of 8,000 --
all seven of which crashed and wrote nothing. A script cut off mid-statement is not a degraded
answer, it is a SyntaxError, so roughly half the roster was at risk of scoring zero for a reason
that has nothing to do with finance.

The cap is not an obstacle the organizers overlooked. The FinanceZero baseline we are measured
against calls the model once with `max_tokens=4096` and runs what comes back; what separates a real
agent from it is meant to be the *25 requests per unit*, not a larger single response. Using one
call and hitting the ceiling is playing the baseline's game.

WHAT STITCHING MEANS HERE -- AND WHAT IT DOES NOT. There is no semantic merge in this module, and
that is deliberate. Only output is capped; input is not. So every continuation is generated with
the whole script-so-far in front of the model: it is not writing a fragment for us to cleverly
reconcile, it is writing the next part of a file it can already read. The intelligence sits in the
generation. This module only appends, and then checks.

Three checks, all mechanical, and each able to reject but never to rewrite:

  * `ast.parse` -- is the joined result a valid program? This is also a truncation detector that
    does not depend on `finish_reason`, which some routes omit.
  * duplicate top-level definitions and imports -- a model that redefines something has lost track
    of what it already wrote, which is a failed assembly rather than untidiness.
  * (left to the caller) does it run and produce the deliverables.

An earlier draft of this module also applied model-proposed edits to the assembled script by
anchored replacement. That is gone. Verifying a claim about code is mechanical; repairing it is
not, and dressing a model-authored patch in deterministic machinery made it look safer than it was.
When something is wrong, the fix is to regenerate a bounded piece with the model -- not surgery.
"""

from __future__ import annotations

import ast
import dataclasses
import os
import pathlib
import re
import time

#: How many times a single generation may be continued. Four continuations is ~20,000 output tokens
#: in total, which is far above anything the baseline's longest script needed; a generation still
#: unfinished by then is looping, not long.
MAX_CONTINUATIONS = 4

#: How many times a break found by the static check may be repaired. A repair rewinds to the break
#: and regenerates from there, so each one costs continuations of its own; two is enough for the
#: single lost line this is built for, and a script that keeps breaking needs a fresh generation
#: rather than more patching of the same one.
MAX_NAME_REPAIRS = 2

#: How many times to re-ask when a response comes back with no usable code at all. Two when the
#: response hit the output ceiling -- it was working and ran out of room, so another go is
#: reasonable -- and one when it stopped of its own accord, since a model that chose not to answer
#: is unlikely to change its mind. Each of these costs a full response for nothing, which is what
#: makes the bound matter: measured on nemotron, eight units spent 12,000 tokens apiece producing
#: nothing at all because the loop retried the same prompt three times.
MAX_EMPTY_RETRIES_AFTER_LENGTH = 2
MAX_EMPTY_RETRIES_AFTER_STOP = 1

#: Output budget for the planning call: a FULL response, not a fraction of one.
#:
#: The first version of this capped the plan at 700 tokens, reasoning that a dozen numbered steps
#: is cheap. Measured, that was wrong in a specific way: the model spent 635 of the 700 on
#: deliberation and emitted 45 characters of plan -- one truncated line. It does not produce a plan
#: and then stop; it reasons its way to one, and a small cap just buys a fragment of the result.
#:
#: Paying for a whole response here is what the measurement supports. With a real plan in its
#: input, the code call's reasoning fell from ~3,900 tokens to 286 and it produced 11,659
#: characters of Python where before it produced none. One response buys back two.
PLAN_MAX_TOKENS = 4000

#: How much of the plan to carry into the code call. Input is not capped, but an essay is not a
#: plan, and this model will write one if allowed.
PLAN_MAX_CHARS = 4000

#: Below this, a response that is not code is too short to be an approach worth feeding back --
#: a refusal, a one-line apology -- and the sterner "write code now" instruction is the better ask.
MIN_PLAN_CHARS = 400

#: How many lines of the prefix are quoted back as the anchor for a continuation. Enough for the
#: model to locate itself unambiguously, short enough not to invite it to re-emit them.
ANCHOR_LINES = 15

_FENCE_OPEN = re.compile(r"^\s*```(?:python|py)?\s*\n", re.IGNORECASE)
_FENCE_CLOSE = re.compile(r"(?:\n|^)\s*```\s*$")


class AssemblyError(Exception):
    """The pieces could not be made into one valid script."""


# ---------------------------------------------------------------------------
# trimming a truncated script back to something whole
# ---------------------------------------------------------------------------

#: Expression kinds that do nothing when they stand alone as a statement, so a trailing
#: `rankings_rows` or `df.to_csv` is the visible stump of a line cut off after its first token.
#:
#: A BACKSTOP, not the mechanism. Two better things run ahead of it: the continuation is asked to
#: restate the cut line in full, which is evidence from the only party that knows what it meant,
#: and `undefined_names` catches the harmful case afterwards whatever shape it took. This list only
#: covers the stumps those two miss -- a stump whose name does resolve, like `df.to_csv` where `df`
#: exists, which no analysis can call wrong and which quietly fails to write a file.
#:
#: A Call is excluded: `write_outputs()` on its own line is a real statement.
_EFFECTLESS = (ast.Name, ast.Attribute, ast.Subscript, ast.Tuple, ast.List, ast.BinOp,
               ast.Compare, ast.UnaryOp)


def _ends_in_a_stump(code: str) -> int:
    """The line number a trailing effect-free expression starts on, or 0 when there is none."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return 0
    if not tree.body:
        return 0
    last = tree.body[-1]
    if isinstance(last, ast.Expr) and isinstance(last.value, _EFFECTLESS):
        return last.lineno
    return 0


def longest_parsable_prefix(code: str) -> tuple[str, int]:
    """The largest leading run of whole lines that parses AND does not end on a stump.

    Rather than guessing where the model was cut off -- mid-statement, inside a string, inside a
    dict literal -- the boundary is computed: drop trailing lines until what remains is a valid
    program. There is no limit on how many that is. An earlier version refused to rewind more than
    80 lines or 30% of the script, on the reasoning that a truncation landing that deep left a ruin
    better regenerated than extended; measured, a response arrived with 229 lines of correct Python
    needing 121 dropped, the cap allowed 68, and the whole script was discarded and asked for again
    from nothing. How much had to be cut is a fact about where the response ran out of room, and
    says nothing about what survives. If truncation landed inside a function, this discards that whole partial function,
    which is right, because we are about to ask for it again.

    Parsing alone is not enough, which took a measured run to discover: a truncation can leave a
    fragment that is syntactically valid and semantically empty, and keeping it loses the statement
    it came from. So a candidate ending in an effect-free expression is rejected and the search
    continues past it.

    Raises AssemblyError when nothing usable survives, or when so much had to go that continuing
    from the remainder is worse than starting over.
    """
    lines = code.splitlines()
    if not lines:
        raise AssemblyError("empty generation")

    for dropped in range(0, len(lines)):
        candidate = lines[:len(lines) - dropped]
        if not candidate:
            break
        text = "\n".join(candidate)
        try:
            ast.parse(text)
        except SyntaxError:
            continue
        stump = _ends_in_a_stump(text)
        if stump:
            # Cut from where the stump begins rather than one line at a time: the expression may
            # span several lines, and stepping would accept a shorter stump on the next pass.
            candidate = candidate[:stump - 1]
            if not candidate:
                break
            text = "\n".join(candidate)
            try:
                ast.parse(text)
            except SyntaxError:
                continue
            if _ends_in_a_stump(text):
                continue
            return text, len(lines) - len(candidate)
        return text, dropped

    raise AssemblyError(
        f"nothing parsable in {len(lines)} line(s): this response was not a script")


def _strip_fences(text: str) -> str:
    """Remove a code fence the model opened around a continuation.

    A continuation is a fragment of a file, but models reliably wrap anything code-shaped in a
    fence anyway. Left in place it would be a SyntaxError in the middle of the assembled script.
    """
    text = _FENCE_OPEN.sub("", text)
    return _FENCE_CLOSE.sub("", text)


#: Does a line look like Python rather than conversation? Used only to find where the code starts
#: in a reply, never to judge code further in. Deliberately generous: wrongly keeping a line of
#: prose produces a SyntaxError we will catch, whereas wrongly dropping a line of code produces a
#: script that runs and is quietly wrong.
_CODEISH = re.compile(
    r"""^(
        (import|from|def|class|async|await|lambda|if|elif|else|for|while|with|try|except|
         finally|return|raise|yield|pass|break|continue|assert|del|global|nonlocal|print)\b
      | [@#]                                    # decorator or comment
      | ["'\[\]{}(),]                           # a literal, or a bracket being continued
      | [\w.]+\s*[-+*/%|&^]?=                    # assignment, including augmented
      | [\w.]+\s*[(\[.]                         # a call, an index or an attribute
    )""",
    re.VERBOSE,
)


def _strip_preamble(text: str) -> str:
    """Drop a conversational opener before the code resumes.

    Only leading lines are considered, and only until something that looks like code appears.
    Anything after that is left alone: a line inside the script that happens to read like prose is
    almost certainly a string or a comment that belongs there.

    If every line looks like prose the original text is returned untouched. Having misjudged the
    whole thing, deleting it would be the worse of the two mistakes -- the join then fails to parse
    and the caller retries, which is recoverable.
    """
    lines = text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        # A continuation legitimately resumes mid-block, so indentation counts as code.
        if line[:1] in (" ", "\t") or _CODEISH.match(stripped):
            return "\n".join(lines[i:])
    return text


def _drop_overlap(prefix: str, continuation: str, max_lines: int = 40) -> str:
    """Remove from `continuation` any opening lines that merely repeat the end of `prefix`.

    Models re-emit the last line or two of what they are continuing however firmly they are asked
    not to, and a duplicated line is not always a syntax error -- a repeated assignment parses
    cleanly and silently changes the program. So the overlap is removed by matching text, longest
    run first, rather than trusted not to occur.
    """
    pre = prefix.rstrip("\n").splitlines()
    cont = continuation.lstrip("\n").splitlines()
    if not pre or not cont:
        return continuation
    for n in range(min(max_lines, len(pre), len(cont)), 0, -1):
        if [ln.rstrip() for ln in pre[-n:]] == [ln.rstrip() for ln in cont[:n]]:
            return "\n".join(cont[n:])
    return "\n".join(cont)


def _resolve_cut_line(prefix: str, continuation: str) -> tuple[str, str]:
    """When the continuation restates our final line in fuller form, prefer theirs.

    THE POINT OF THIS FUNCTION. Where a response was cut off mid-statement is a question about what
    the model intended to write, and only the model knows that. So the continuation prompt asks it
    to write that one line again in full, and this reads the answer: if its first line begins with
    our last line, ours was a fragment and is dropped.

    That single rule covers every shape of truncation -- mid-call, mid-subscript, mid-comprehension,
    mid-f-string -- because it rests on evidence rather than on recognising shapes. The list of
    "effect-free expression" node types elsewhere in this module is a backstop for when the model
    does not comply, not the mechanism.

    Only a strict extension counts. An identical line is ordinary repetition and `_drop_overlap`
    handles it; a merely similar line is not evidence of anything.
    """
    pre = prefix.rstrip("\n").splitlines()
    cont = continuation.lstrip("\n").splitlines()
    if not pre or not cont:
        return prefix, continuation
    ours, theirs = pre[-1].strip(), cont[0].strip()
    if ours and theirs != ours and theirs.startswith(ours):
        return "\n".join(pre[:-1]), "\n".join(cont)
    return prefix, continuation


def join(prefix: str, continuation: str) -> str:
    """Append a continuation to the script so far.

    In order: strip the fence the model wraps a fragment in, drop any conversational opener, let a
    restated line replace the fragment it completes, then remove plain repetition of the tail.
    """
    piece = _strip_preamble(_strip_fences(continuation))
    prefix, piece = _resolve_cut_line(prefix, piece)
    piece = _drop_overlap(prefix, piece)
    if not piece.strip():
        return prefix
    if not prefix.strip():
        return piece.lstrip("\n")
    return prefix.rstrip("\n") + "\n" + piece.lstrip("\n")


# ---------------------------------------------------------------------------
# checks on the assembled script
# ---------------------------------------------------------------------------

def extract_partial(reply: str) -> str:
    """Pull code out of a reply that may have been cut off mid-stream.

    Three cases, in order of confidence: a properly closed fenced block; an opening fence whose
    close never arrived, in which case everything after it is the code; and a bare reply, which is
    returned as-is. The last is not a guess we have to be right about -- `parse_error` is the
    arbiter, and prose that is not code simply fails to parse and is handled as a failed assembly.
    """
    text = reply or ""
    closed = re.search(r"```(?:python|py)?\s*\n(.*?)\n?```", text, re.DOTALL | re.IGNORECASE)
    if closed:
        return closed.group(1).strip("\n")
    opened = re.search(r"```(?:python|py)?\s*\n", text, re.IGNORECASE)
    if opened:
        return text[opened.end():]
    return text.strip("\n")


def parse_error(code: str) -> str | None:
    """None when the script is a valid program, else a short description of why it is not."""
    try:
        ast.parse(code)
    except SyntaxError as e:
        return f"{e.msg} at line {e.lineno}"
    except ValueError as e:                 # e.g. a null byte in the source
        return str(e)
    return None


def undefined_names(code: str) -> list[str]:
    """Names the script uses but never assigns, as "<name> (line N)".

    This is the check that matters most after a join, and it is delegated rather than written here.
    Scope analysis has to be right about comprehensions, class bodies, `global`, the walrus
    operator, star-imports and except-clause names, and pyflakes is the mature version of exactly
    that. An earlier draft of this module hand-rolled a list of AST node shapes that "looked like"
    a truncation stump; it was a worse approximation of this, and it missed shapes it had not
    enumerated.

    Why it earns its place: a script assembled from several responses can lose a line at a seam,
    and the result is a name that is read but never built. That is not a style problem -- it is a
    `NameError` partway through a 300-line script, after which nothing further is written. Across
    seven measured units it was the largest single failure once assembly was working, and pyflakes
    found every instance of it by name and line.

    Returns [] when pyflakes is unavailable, so a development machine without it behaves no worse
    than before: an unverifiable script is not a broken one.
    """
    try:
        from pyflakes import checker as pyflakes_checker
        from pyflakes import messages as pyflakes_messages
    except ImportError:
        return []
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return []                            # parse_error already covers this
    try:
        result = pyflakes_checker.Checker(tree, filename="<assembled>")
    except Exception:
        return []
    # ImportShadowedByLoopVar is here because of a measured failure: a script died with
    # `AttributeError: 'numpy.float64' object has no attribute 'DataFrame'`, which is what it looks
    # like when `for pd in ...` rebinds the pandas alias to a number. No analysis of `pd` itself
    # would catch that -- the name is defined, just not what the code assumes -- and a loop variable
    # is essentially never meant to shadow an import.
    wanted = (pyflakes_messages.UndefinedName,
              pyflakes_messages.UndefinedLocal,
              pyflakes_messages.UndefinedExport,
              pyflakes_messages.ImportShadowedByLoopVar)
    out: list[str] = []
    for message in result.messages:
        if isinstance(message, wanted):
            name = message.message_args[0] if message.message_args else "?"
            entry = f"{name} (line {message.lineno})"
            if entry not in out:
                out.append(entry)
    return out


def duplicate_definitions(code: str) -> list[str]:
    """Top-level functions or classes defined more than once, and repeated imports.

    A redefinition is the signature failure of generating a file in pieces: the model has lost
    track of what it already wrote. Python accepts it silently -- the later definition simply wins
    -- so nothing downstream would notice. Reported, never repaired: which of the two was intended
    is not knowable from the text, and only whoever regenerates the piece can decide.
    """
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return []                            # parse_error already covers this
    seen: dict[str, int] = {}
    imports: dict[str, int] = {}
    out: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            seen[node.name] = seen.get(node.name, 0) + 1
            if seen[node.name] == 2:
                out.append(f"{node.name} is defined more than once")
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                key = f"{getattr(node, 'module', '') or ''}.{alias.name}"
                imports[key] = imports.get(key, 0) + 1
                if imports[key] == 2:
                    out.append(f"{alias.name} is imported more than once")
    return out


# ---------------------------------------------------------------------------
# the driver
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Assembly:
    """One assembled script and how it came to be.

    A single record rather than a handful of return values, because the pipeline that produces it
    is about to grow (planned sections, per-deliverable repair) and threading five more parameters
    through it would be worse. It is a record, not a state machine: nothing dispatches on it.
    """
    code: str = ""
    responses: int = 0              # model calls spent on this assembly
    planned: bool = False           # a plan was obtained and fed into the code call
    continuations: int = 0          # how many of those were continuations
    empty_retries: int = 0          # responses that came back with no usable code at all
    name_repairs: int = 0           # rewinds driven by a name used before it was assigned
    tokens_in: int = 0              # summed from each response, NOT read off the module-level
    tokens_out: int = 0             # SESSION_USAGE, which `Usage.add` mutates in place
    tokens_reasoning: int = 0       # of tokens_out, the share spent thinking rather than answering
    rewound_lines: int = 0          # lines discarded to reach a parsable prefix
    complete: bool = False          # the model signalled a natural stop and the result parses
    notes: list[str] = dataclasses.field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.code.strip()) and parse_error(self.code) is None


def _make_plan(result: "Assembly", messages: list[dict], chat_fn, deadline: float | None,
               **chat_kwargs) -> str:
    """Spend one short response deciding the approach, and return it.

    Takes whatever deliberation comes back: the plan text if the model emitted one, and failing
    that its reasoning, which on this model is often the only thing a response produces. Both are
    the same thing for our purposes -- an approach, already paid for, that the code call can start
    from instead of deriving again.
    """
    from .prompt import build_plan_prompt

    if deadline is not None and time.monotonic() >= deadline:
        return ""
    kwargs = dict(chat_kwargs)
    kwargs["max_tokens"] = min(kwargs.get("max_tokens", PLAN_MAX_TOKENS), PLAN_MAX_TOKENS)
    try:
        reply = chat_fn(build_plan_prompt(messages), **kwargs)
    except TypeError:
        # A chat_fn that does not take max_tokens (the scripted ones in tests).
        reply = chat_fn(build_plan_prompt(messages), **chat_kwargs)
    result.responses += 1
    _account(result, reply)

    # Take the LONGER of the two, not content-first. Both are deliberation and either may hold it:
    # on one measured call the model emitted 45 characters of plan alongside 635 tokens of
    # reasoning, and preferring content would have thrown away almost everything we paid for.
    content = (reply.text or "").strip()
    reasoning = (getattr(reply, "reasoning", "") or "").strip()
    plan = max((content, reasoning), key=len)
    # Saved for diagnosis when asked, outside the output directory. Without this a plan that
    # steers the implementation wrong is indistinguishable from an implementation that went wrong
    # on its own -- measured, a unit's signal count moved from a correct 105 to 106 the moment the
    # plan was asked to decide window boundaries explicitly, and nothing recorded what it decided.
    keep = os.environ.get("QFBENCH_KEEP_PLAN", "").strip()
    if plan and keep:
        try:
            unit = os.environ.get("QFBENCH_UNIT_ID", "unit")
            target = pathlib.Path(keep) / f"{unit}.plan.md"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(plan, encoding="utf-8")
            result.notes.append(f"plan saved to {target}")
        except OSError:
            pass
    if not plan:
        result.notes.append("the planning call produced nothing")
    elif plan is reasoning and content:
        result.notes.append("the planning call reasoned more than it wrote; using the reasoning")
    elif plan is reasoning:
        result.notes.append("the planning call returned only reasoning; using it as the plan")
    return plan


def _is_usable(code: str) -> bool:
    """Is there any Python here at all?

    Not "does it parse" -- a truncated script does not, and is perfectly usable as a prefix to
    continue from. The test is whether ANY leading run of it is a valid program. That separates a
    half-written script from a refusal or an answer given in prose, which the extractor hands back
    verbatim because it cannot know the difference.
    """
    if not code.strip():
        return False
    try:
        longest_parsable_prefix(code)
    except AssemblyError:
        return False
    return True


def _ask_until_code(result: "Assembly", messages: list[dict], chat_fn, extract,
                    deadline: float | None, **chat_kwargs):
    """Send the prompt, and re-ask with a sterner instruction while nothing usable comes back.

    A response carries two signals and both matter: whether any code was extracted, and why the
    model stopped. A reasoning model that hits the output ceiling having emitted nothing has not
    refused -- it has spent the whole budget thinking -- and the remedy is a different instruction,
    not a different attempt at the same one. Retrying the identical prompt reproduces the identical
    failure, which is exactly what the measured runs did three times per unit.

    Returns the last reply and whatever code came with it ("" when none ever did).
    """
    from .prompt import build_no_output_message

    reply = chat_fn(messages, **chat_kwargs)
    result.responses += 1
    _account(result, reply)
    # `_strip_fences` after extraction, not instead of it: a response cut off at the ceiling never
    # emits its closing fence, so the fenced-block extractor misses it and falls back to the raw
    # reply -- with the opening ```python still attached as the first line. That line is not Python,
    # so every prefix of the script would fail to parse and a perfectly recoverable truncation
    # would look unsalvageable. This is the single most important line in the module.
    code = _strip_fences(extract(reply.text) or "")
    if _is_usable(code):
        return reply, code

    spent_thinking = bool(getattr(reply, "truncated", False))
    budget = (MAX_EMPTY_RETRIES_AFTER_LENGTH if spent_thinking
              else MAX_EMPTY_RETRIES_AFTER_STOP)
    result.notes.append(
        "no usable code in the response; it "
        + ("hit the output ceiling having produced only reasoning" if spent_thinking
           else "stopped of its own accord"))

    for _ in range(budget):
        if deadline is not None and time.monotonic() >= deadline:
            result.notes.append("ran out of wall clock before any code was produced")
            break

        # Whatever the model DID produce -- a prose write-up of the approach, or the reasoning
        # behind one -- is the expensive part of solving the task, and we have already bought it.
        # Measured: feeding such a write-up back cut the next call's reasoning from ~3,900 tokens
        # to 286 and produced 11,659 characters of Python on a unit that had produced none. So the
        # response is recycled as a plan rather than discarded, and only a truly empty one gets the
        # bare "write code" instruction.
        from .prompt import build_implement_message
        deliberation = max((_strip_fences(reply.text or ""),
                            getattr(reply, "reasoning", "") or ""), key=len).strip()
        if len(deliberation) >= MIN_PLAN_CHARS:
            result.notes.append("the response was an approach rather than code; using it as a plan")
            turns = list(messages) + [
                {"role": "assistant", "content": deliberation[:PLAN_MAX_CHARS]},
                {"role": "user", "content": build_implement_message(deliberation, PLAN_MAX_CHARS)},
            ]
        else:
            turns = list(messages) + [{"role": "user",
                                       "content": build_no_output_message(spent_thinking)}]
        reply = chat_fn(turns, **chat_kwargs)
        result.responses += 1
        result.empty_retries += 1
        _account(result, reply)
        code = _strip_fences(extract(reply.text) or "")
        if _is_usable(code):
            result.notes.append("a sterner re-ask produced code")
            return reply, code
        spent_thinking = bool(getattr(reply, "truncated", False))

    result.notes.append("no usable code after re-asking")
    return reply, ""


def _account(result: "Assembly", reply) -> None:
    """Add one response's token usage to the assembly.

    Counted per response rather than by comparing the module-level SESSION_USAGE before and after:
    `Usage.add` mutates in place, so a "before" snapshot is the same object as the "after" one and
    every difference reads zero. That is not hypothetical -- a whole sweep once reported attempts=0
    for the same reason.
    """
    usage = getattr(reply, "usage", None)
    if usage is not None:
        result.tokens_in += getattr(usage, "input_tokens", 0)
        result.tokens_out += getattr(usage, "output_tokens", 0)
        result.tokens_reasoning += getattr(usage, "reasoning_tokens", 0)


def _extend(result: "Assembly", messages: list[dict], chat_fn, continuation_prompt,
            max_continuations: int, deadline: float | None, **chat_kwargs) -> None:
    """Keep asking for more until the script is whole, the cap is reached, or time runs out.

    Separate from `generate_script` because it is entered twice: once because a response hit the
    output ceiling, and again when the static check finds a break and the script is cut back to it.
    Re-using one loop means a repair cannot behave differently from a continuation.
    """
    for _ in range(max_continuations):
        if deadline is not None and time.monotonic() >= deadline:
            result.notes.append("ran out of wall clock mid-assembly")
            return

        # The previous turns are REPLACED, not appended to: the model needs the task and the script
        # so far, and nothing is gained by carrying every earlier continuation instruction along.
        # This keeps input proportional to the script rather than to the number of rounds.
        turns = list(messages) + [
            {"role": "assistant", "content": f"```python\n{result.code}\n```"},
            {"role": "user", "content": continuation_prompt(result.code)},
        ]
        reply = chat_fn(turns, **chat_kwargs)
        result.responses += 1
        result.continuations += 1
        _account(result, reply)

        piece = reply.text or ""
        joined = join(result.code, piece)
        if joined == result.code:
            # The identical failure as on a first response, and it needs the identical remedy: a
            # continuation that hit the ceiling having emitted only reasoning has not run out of
            # things to say, it has run out of budget to say them in. Giving up here would discard
            # a script that is already most of the way written.
            if getattr(reply, "truncated", False) and result.empty_retries < MAX_EMPTY_RETRIES_AFTER_LENGTH:
                from .prompt import build_no_output_message
                result.empty_retries += 1
                result.notes.append("a continuation produced only reasoning; re-asking for code")
                turns = list(messages) + [
                    {"role": "assistant", "content": f"```python\n{result.code}\n```"},
                    {"role": "user", "content": build_no_output_message(True)},
                ]
                reply = chat_fn(turns, **chat_kwargs)
                result.responses += 1
                _account(result, reply)
                joined = join(result.code, reply.text or "")
                if joined == result.code:
                    result.notes.append("still nothing after re-asking; stopping")
                    return
            else:
                result.notes.append("a continuation added nothing; stopping")
                return

        why = parse_error(joined)
        if why is None:
            result.code = joined
            if not getattr(reply, "truncated", False):
                result.complete = True
                return
            continue

        # The continuation itself was cut off, which is expected on a long script: keep the part
        # that parses and go round again.
        try:
            result.code, dropped = longest_parsable_prefix(joined)
            result.rewound_lines += dropped
        except AssemblyError:
            result.notes.append(f"a continuation could not be joined cleanly: {why}")
            return
    result.notes.append(f"still unfinished after {max_continuations} continuation(s)")


def plan_first_enabled() -> bool:
    """Plan first, then implement. The deliberation happens on any non-trivial task whether we ask
    for it or not; asking for it explicitly is what stops it eating the response the code needs.
    Switched off with QFBENCH_NO_PLAN_FIRST=1 so the two can be compared."""
    return os.environ.get("QFBENCH_NO_PLAN_FIRST", "").strip() != "1"


def _plan_into(result: "Assembly", messages: list[dict], chat_fn, deadline: float | None,
               **chat_kwargs) -> list[dict]:
    """Make the plan and return `messages` extended with the plan turns (unchanged if no plan)."""
    from .prompt import build_implement_message

    plan = _make_plan(result, messages, chat_fn, deadline, **chat_kwargs)
    if not plan:
        return messages
    result.planned = True
    return list(messages) + [
        {"role": "assistant", "content": plan},
        {"role": "user", "content": build_implement_message(plan, PLAN_MAX_CHARS)},
    ]


def plan_once(messages: list[dict], *, chat_fn=None, deadline: float | None = None,
              **chat_kwargs) -> tuple[list[dict], "Assembly"]:
    """Plan a unit exactly once, before any generation.

    Returns the messages with the plan turns appended and an `Assembly` carrying only the cost of
    the planning call, so the caller can account for it. Repairs and restarts then build on the
    planned messages rather than spending a request re-deriving a decision that was fine.
    """
    from . import llm

    result = Assembly()
    if plan_first_enabled():
        messages = _plan_into(result, messages, chat_fn or llm.chat, deadline, **chat_kwargs)
    return messages, result


def generate_script(
    messages: list[dict],
    *,
    chat_fn=None,
    extract=None,
    continuation_prompt=None,
    max_continuations: int = MAX_CONTINUATIONS,
    deadline: float | None = None,
    plan_first: bool | None = None,
    **chat_kwargs,
) -> Assembly:
    """Generate a complete script, continuing across responses when one is not enough.

    `plan_first=None` follows `plan_first_enabled()`. A caller that has already planned (see
    `plan_once`) passes False so repairs and restarts reuse the plan instead of re-rolling it.

    The loop ends when the model stops naturally AND the accumulated code parses. Those are two
    separate conditions on purpose: `finish_reason` is absent on some routes, and a response can
    also claim to have stopped while leaving a block unclosed, so neither signal alone is trusted.

    Stops early, with whatever is usable, on three bounds: the continuation cap, the wall clock, and
    the request budget (a `BudgetExceeded` from the caller's `chat_fn` propagates -- running out of
    requests mid-assembly is the caller's problem to report, not something to swallow here).

    `chat_fn` and `extract` are injectable so the whole loop can be driven offline in tests with
    scripted truncation, rather than only against a live route.
    """
    from . import llm
    from .prompt import build_continuation_message

    chat_fn = chat_fn or llm.chat
    extract = extract or extract_partial
    continuation_prompt = continuation_prompt or build_continuation_message

    result = Assembly()

    if plan_first is None:
        plan_first = plan_first_enabled()
    if plan_first:
        messages = _plan_into(result, messages, chat_fn, deadline, **chat_kwargs)

    reply, code = _ask_until_code(result, messages, chat_fn, extract, deadline, **chat_kwargs)
    if not code.strip():
        return result

    truncated = getattr(reply, "truncated", False) or parse_error(code) is not None
    if truncated:
        # Trim the partial tail before asking for more. If this fails the truncation landed too
        # deep to continue from, and the honest answer is to hand back nothing and let the caller
        # retry the whole generation rather than extend a ruin.
        try:
            code, dropped = longest_parsable_prefix(code)
        except AssemblyError as e:
            result.code = ""
            # Worth distinguishing in the log, because the two call for different responses and
            # stderr is the only window we have at scoring time: a response that hit the ceiling
            # and left nothing usable is a length problem, while one that stopped naturally and
            # still does not parse was never a script -- a refusal, or an answer in prose.
            if getattr(reply, "truncated", False):
                result.notes.append(str(e))
            else:
                result.notes.append("the response stopped naturally but contains no parsable code")
            return result
        result.rewound_lines += dropped
        result.code = code
        _extend(result, messages, chat_fn, continuation_prompt, max_continuations, deadline,
                **chat_kwargs)
    else:
        result.code, result.complete = code, True

    # Everything below runs whether or not the script needed continuing. A single response can lose
    # a line too -- the model is writing to a budget and may skip what it ran out of room for -- and
    # an early return for the "complete" case meant the one check that catches it never ran.

    # The static check, and the repair it drives. A script assembled from several responses can
    # lose a line at a seam, and the result is a name read but never built -- a NameError partway
    # through, after which nothing further is written. Rather than patching the text, the script is
    # cut back to the break and the continuation machinery runs again from there: the same
    # mechanism, triggered by a different signal, so there is no second way for this to go wrong.
    for _ in range(MAX_NAME_REPAIRS):
        if parse_error(result.code) is not None:
            break
        broken = undefined_names(result.code)
        if not broken:
            break
        name, _, where = broken[0].partition(" (line ")
        try:
            line_no = int(where.rstrip(")"))
        except ValueError:
            break
        lines = result.code.splitlines()
        if line_no <= 1 or line_no > len(lines):
            break
        result.code = "\n".join(lines[:line_no - 1])
        result.rewound_lines += len(lines) - (line_no - 1)
        result.name_repairs += 1
        result.complete = False
        result.notes.append(f"'{name}' was used before it was assigned (line {line_no}); "
                            f"regenerating from there")
        _extend(result, messages, chat_fn,
                lambda code, _n=name: continuation_prompt(code, missing=_n),
                max_continuations, deadline, **chat_kwargs)

    still_broken = undefined_names(result.code) if parse_error(result.code) is None else []
    if still_broken:
        # Reported, not hidden: the caller's repair loop sees it, and the sweep records it. A script
        # with a dangling name will raise where it is used, so this is a prediction of the failure
        # rather than a style note.
        result.notes.append("name(s) still used before assignment: " + ", ".join(still_broken[:4]))

    if parse_error(result.code) is not None:
        # Never hand back something that cannot run; the caller's retry is cheaper than an
        # execution that is certain to fail.
        result.notes.append(f"assembled script does not parse: {parse_error(result.code)}")
        result.code = ""
    else:
        result.notes.extend(duplicate_definitions(result.code))
    return result
