# SPDX-License-Identifier: MIT
"""Score the task-context parser against ground truth, across every public unit.

Ground truth = the filenames each unit's own `checks/*.py` actually opens under OUTPUT_DIR.

This grades the PROSE parse specifically (`use_checks=False`). The agent may also read the
checker at run time when it is mounted, but that path cannot be graded here -- it IS the ground
truth, so it would agree with itself. What needs measuring is the fallback: how well prose alone
recovers the contract when the checker is absent.

The practice kit is a SIBLING of this repo, not part of it -- their task data must never
be vendored into our image (licence: it is QF-Bench v1 material, not ours to relicense).
Point $QFBENCH_KIT elsewhere if you keep the kit somewhere else.

Usage:  python3 tools/validate_task_context.py [units_dir] [-v]
"""

from __future__ import annotations

import os
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from agent.task_context import DATA_EXTS, NEVER_OUTPUT, parse_task  # noqa: E402

_EXT_ALT = "|".join(DATA_EXTS)

# A quoted filename OR a quoted path ending in one. The path form is the one the previous
# extractor missed entirely: checkers overwhelmingly write f"{OUTPUT_DIR}/results.csv" or
# Path("/app/output/x.json"), not a bare "results.csv" -- so 19 of 87 units yielded no ground
# truth at all and were silently skipped rather than graded.
_QUOTED = re.compile(rf"""['"]([A-Za-z0-9_./{{}}-]+\.(?:{_EXT_ALT}))['"]""")

# Position, not spelling, separates a deliverable from an input (the organizers' own wording in
# conformance.sh). A checker reaches an INPUT through a data/reference root and a LOG artefact
# through a log root; those files already exist, so crediting them would pass an agent that wrote
# nothing of its own. Veto the whole line when it reaches into one of those roots.
_ROOT_VETO = re.compile(
    r"(data|input|ref|reference|log)_(dir|path)"
    r"|input_(json|csv|parquet|text)"
    r"|/app/data|/input|/tests/reference|/logs"
    r"|[^a-z0-9_](data|ref) */",
    re.I,
)
# What survives must be either a bare name (a helper joins it to the output dir) or explicitly
# output-rooted.
_OUTPUT_ROOTED = re.compile(r"^(/app)?/output/|^\{OUTPUT_DIR\}/")


def ground_truth(unit: pathlib.Path) -> set[str]:
    """Filenames the unit's checker resolves against the OUTPUT dir.

    Reads every `checks/*.py`, not just test_outputs.py: on some units test_outputs.py is a thin
    shim that delegates to checks/verifier.py and names no file itself.
    """
    names: set[str] = set()
    for f in sorted((unit / "checks").glob("*.py")):
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            if _ROOT_VETO.search(line):
                continue
            for tok in _QUOTED.findall(line):
                if "/" in tok and not _OUTPUT_ROOTED.match(tok):
                    continue
                base = tok.rsplit("/", 1)[-1]
                if "{" in base or "}" in base:      # an f-string hole, not a name
                    continue
                names.add(base)
    return names - NEVER_OUTPUT


def main() -> int:
    positional = [a for a in sys.argv[1:] if not a.startswith("-")]
    default_kit = os.environ.get(
        "QFBENCH_KIT", str(pathlib.Path(__file__).resolve().parents[2] / "track1-coding-public")
    )
    root = pathlib.Path(positional[0] if positional else f"{default_kit}/units").resolve()
    if not root.is_dir():
        print(f"kit not found at {root}\n"
              f"clone it beside this repo, or set $QFBENCH_KIT to its path", file=sys.stderr)
        return 2
    verbose = "-v" in sys.argv
    units = sorted(d for d in root.iterdir() if (d / "instruction.md").is_file())

    exact = graded = tp = fp = fn = superset = 0
    extras = []
    zero_out = []
    problems = []

    for u in units:
        # use_checks=False is essential: this script grades the PROSE parse against the
        # checker. If the parser were also allowed to read the checker, it would be scored
        # against its own input and would agree by construction.
        ctx = parse_task(u, probe_filesystem=False, use_checks=False)
        pred = set(ctx.output_filenames)
        if not pred:
            zero_out.append(u.name)
        gt = ground_truth(u)
        if not gt:
            continue                      # no usable ground truth for this unit
        graded += 1
        tp += len(pred & gt); fp += len(pred - gt); fn += len(gt - pred)
        if pred == gt:
            exact += 1
        elif gt <= pred:
            superset += 1   # every graded file covered, plus extras the checks ignore
            extras.append((u.name, sorted(pred - gt)))
        else:
            problems.append((u.name, sorted(gt - pred), sorted(pred - gt)))

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    print(f"units parsed          : {len(units)}")
    print(f"units with ground truth: {graded}")
    if graded:
        print(f"exact filename-set match : {exact}/{graded} ({exact / graded * 100:.1f}%)")
        print(f"superset (no file missed): {superset}  -> covered "
              f"{(exact + superset)}/{graded} ({(exact + superset) / graded * 100:.1f}%)")
    print(f"file-level precision  : {prec:.3f}  ({tp} tp, {fp} fp)")
    print(f"file-level recall     : {rec:.3f}  ({tp} tp, {fn} fn)")
    print(f"file-level F1         : {f1:.3f}")
    print(f"units yielding NO output filename: {len(zero_out)} {zero_out if zero_out else ''}")

    if extras and verbose:
        print("\n--- supersets: extra files instruction.md requires but checks ignore ---")
        for name, ex in extras:
            print(f"{name}: {ex}")
    if problems and verbose:
        print("\n--- mismatches (unit :: MISSED | SPURIOUS) ---")
        for name, missed, spurious in problems:
            print(f"{name}\n    missed  : {missed}\n    spurious: {spurious}")
    elif problems:
        print(f"\n{len(problems)} units mismatched; rerun with -v for detail")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())