# SPDX-License-Identifier: MIT
"""Show the finished answer to the model and ask whether the numbers can be right.

WHY THIS AND NOT THE CHECKS WE ALREADY HAVE. The dev run reported 81 of 86 units as
`domain_gate_failed`: the code ran, the files were well-formed, the numbers were wrong. Three
layers were built for that and measured against saved failures:

    layer                     caught   missed   false alarms
    hand-written invariants      0        2          --
    generated property tests     0        2           1  (a backwards VaR rule that cost a
                                                          unit its entire repair budget)
    this critic                  2        1           0

The difference is information. An invariant is written in advance and keys on column names; a
generated checker is written before the output exists and has to anticipate. The critic sees the
actual numbers. Given eight option values of exactly 0.0 it answered "wrong" and reasoned from the
task's own parameters -- S0=K=100, sigma=0.3, T=1 -- that the value must be positive. Given a
subtler failure it identified both broken invariants, including an exercise boundary pinned at the
grid maximum and an American call differing from a European by 0.0507 where the task demands 1e-3.

WHAT KEEPS IT CHEAP AND SAFE.

  * It runs ONCE per unit, at the only moment it is useful: when the script has run cleanly and
    the contract is satisfied, which is where we would otherwise stop with wrong numbers.
  * A `wrong` verdict buys exactly ONE more generation. The properties layer showed what happens
    without that bound -- a false alarm consumed three attempts on a unit whose first attempt was
    already clean.
  * It cannot fail a unit, edit output, or override anything. A failed call, a missing verdict or
    an unparseable reply are all treated as `plausible`.
  * `PROBLEM` lines are read only when the verdict is `wrong`. Measured, the model sometimes argues
    with itself mid-reply -- "so this is not a problem. Wait -- rechecking" -- and the verdict line
    is the only part worth trusting.

AND THE COMPARISON, WHICH IS THE PART TO BE CAREFUL ABOUT. After a critic-driven repair there are
two clean candidates and our deterministic score cannot separate them: same contract errors, same
files. Asking the model to choose is reasonable, but the critic is NOT an independent judge of its
own complaint -- it will favour the answer that addressed the objection it raised, including when
the objection was wrong. So the comparison is BLIND: two candidates, unlabelled, in a fixed order
decided by a coin we have already flipped, with no mention that one is a revision and no
restatement of the complaint. It is asked which is more likely correct, not whether its point was
taken.
"""

from __future__ import annotations

import pathlib
import re

#: Characters of each deliverable shown. Enough for a JSON summary in full and the head of a CSV.
MAX_PER_FILE = 2500

#: Characters of output shown in total.
MAX_TOTAL = 12000

#: Problems carried into the repair turn. More than this and the critic has misread the task.
MAX_PROBLEMS = 5

#: Files that belong to the grader, never to us -- never shown, never judged.
_NOT_OURS = {"reward.json", "reward.txt", "pytest_report.json"}

_VERDICT = re.compile(r"^\s*VERDICT:\s*(plausible|wrong)\b", re.IGNORECASE | re.MULTILINE)
#: A problem must name what the value SHOULD be and what it IS. That shape cannot express a
#: confirmation, which is the point: measured, the critic once returned `VERDICT: wrong` followed
#: by five "problems" that each concluded the value was correct -- "which is correct", "which
#: matches", "is plausible" -- and all five were passed to the repair loop as defects. A line that
#: does not contradict itself is not a defect, and now cannot be written as one.
_PROBLEM = re.compile(r"^\s*PROBLEM:\s*(.+?\bshould be\b.+?\bbut is\b.+?)\s*$",
                      re.IGNORECASE | re.MULTILINE)
_CHOICE = re.compile(r"^\s*ANSWER:\s*([AB])\b", re.IGNORECASE | re.MULTILINE)

CRITIC_INSTRUCTION = (
    "You are reviewing the OUTPUT of a program written for the task above. You cannot see the "
    "code -- only the numbers it produced.\n\n"
    "{body}\n\n"
    "{checklist}"
    "Decide whether these could be a correct answer to the task.\n\n"
    "Judge only what the numbers themselves show: values that are impossible, relationships that "
    "must hold and do not, quantities that cannot be zero or constant given these inputs, "
    "orderings that are the wrong way round. Do not comment on formatting, precision or anything "
    "you would need to see the code to know.\n\n"
    "List any problems FIRST, one line each, in exactly this form:\n"
    "PROBLEM: <the value> should be <what it must be> but is <what it actually is>\n\n"
    "Every problem must fill in all three parts. If you check something and it turns out to be "
    "correct, write nothing about it -- this list is only for things that are wrong.\n\n"
    "Then, on the LAST line, state your conclusion:\n"
    "VERDICT: plausible | wrong\n\n"
    "Say wrong only if you listed at least one problem above. If you listed none, say plausible."
)

COMPARE_INSTRUCTION = (
    "Two different programs were written for the task above, and each produced an answer. "
    "You cannot see either program -- only what they wrote.\n\n"
    "## Answer A\n{a}\n\n## Answer B\n{b}\n\n"
    "Which is more likely to be correct?\n\n"
    "Reply in exactly this form:\n"
    "ANSWER: A | B\n"
    "REASON: <one line, naming the value that decides it>\n\n"
    "Judge the numbers alone. If they are equally plausible, pick the one whose values are more "
    "consistent with each other and with the task's stated inputs."
)


def read_outputs(out_dir, max_per_file: int = MAX_PER_FILE, max_total: int = MAX_TOTAL) -> str:
    """The deliverables, as text, for the model to look at.

    Binary files are named rather than quoted -- a plot is not something it can judge, and the
    bytes would swamp everything that is.
    """
    out = pathlib.Path(out_dir)
    if not out.is_dir():
        return ""
    parts: list[str] = []
    spent = 0
    for f in sorted(out.iterdir()):
        if f.is_dir() or f.name in _NOT_OURS or f.name.startswith("."):
            continue
        try:
            text = f.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            parts.append(f"### {f.name}\n(not text; {f.stat().st_size} bytes)")
            continue
        head = text[:max_per_file]
        if len(text) > max_per_file:
            head += f"\n... ({len(text)} characters in total)"
        block = f"### {f.name}\n{head}"
        if spent + len(block) > max_total:
            parts.append("(further files not shown)")
            break
        parts.append(block)
        spent += len(block)
    return "\n\n".join(parts)


def build_critic_prompt(messages: list[dict], out_dir, category: str = "") -> list[dict] | None:
    """The review turn, or None when there is nothing to review.

    The category's invariants are repeated here as a CHECKLIST, even though they already appear
    earlier in the conversation. Placement is the point. In the original prompt they are advice to
    someone about to write code -- "put-call parity must hold" is a thing to nod at before there
    are any numbers. Here there are numbers, and the same sentence is a test that can be run. The
    critic had them in its context all along and never referred to them, because the question it
    was asked was the open-ended "is this plausible" rather than "does this hold".
    """
    body = read_outputs(out_dir)
    if not body.strip():
        return None
    checklist = ""
    if category:
        from .invariants import guidance_for
        checklist = (
            "Work through these first. They hold for any correct answer to a task of this kind, "
            "so each one is a test you can run against the numbers above:\n\n"
            f"{guidance_for(category)}\n\n"
            "Where the task states its own convention, the task wins. Then look for anything else "
            "impossible that these do not cover.\n\n")
    return list(messages) + [{"role": "user",
                              "content": CRITIC_INSTRUCTION.format(body=body,
                                                                   checklist=checklist)}]


#: Pulls the two numbers out of "<name> should be X but is Y".
_SHOULD_BE = re.compile(r"should be\s+(-?[\d.eE+]+)\s+but is\s+(-?[\d.eE+]+)", re.IGNORECASE)


def _is_a_real_contradiction(problem: str) -> bool:
    """Reject a "problem" whose expected and actual values are the same number.

    Measured, with the required wording in place: `annualized_vol should be 0.266599766168436 but
    is 0.266599766168436`. The template asks for a contradiction and the model filled both halves
    with the same value -- a confirmation that satisfies the format. Non-numeric problems are kept;
    the test is only applied where there are two numbers to compare.
    """
    match = _SHOULD_BE.search(problem)
    if not match:
        return True
    try:
        expected, actual = float(match.group(1)), float(match.group(2))
    except ValueError:
        return True
    return expected != actual


def parse_verdict(reply: str) -> tuple[bool, list[str]]:
    """(is_wrong, problems). Anything unparseable is read as `plausible`.

    The verdict line gates the problems deliberately. The model sometimes reasons its way across
    a reply and ends up somewhere other than where it started, and only its conclusion is worth
    acting on.
    """
    match = _VERDICT.search(reply or "")
    if not match or match.group(1).lower() != "wrong":
        return False, []
    problems = [p[:300] for p in _PROBLEM.findall(reply or "")
                if _is_a_real_contradiction(p)][:MAX_PROBLEMS]
    # A `wrong` verdict with nothing that states a contradiction is not actionable, and measured,
    # it is usually not a real objection either.
    return (True, problems) if problems else (False, [])


def render(problems: list[str]) -> str:
    """The problems, phrased for the repair turn."""
    lines = "\n".join(f"- {p}" for p in problems)
    return (
        "The script ran and wrote every required file, so nothing is structurally wrong. But "
        f"reviewing the NUMBERS against the task, these cannot be right:\n\n{lines}\n\n"
        "Find the computation that produces each one and fix it. Do not adjust the values on the "
        "way out to satisfy this -- the arithmetic that produced them is what is wrong."
    )


def build_compare_prompt(messages: list[dict], first: str, second: str) -> list[dict]:
    """A blind comparison of two candidate answers.

    Blind on purpose. The critic is not a neutral judge of whether its own objection was answered:
    told which output is the revision, it will favour the revision. Unlabelled, in an order it
    cannot infer, it is answering a different and much cleaner question -- which of these two sets
    of numbers is more likely to be right.
    """
    return list(messages) + [{"role": "user",
                              "content": COMPARE_INSTRUCTION.format(a=first, b=second)}]


def parse_choice(reply: str) -> str:
    """"A", "B", or "" when the reply does not say."""
    match = _CHOICE.search(reply or "")
    return match.group(1).upper() if match else ""
