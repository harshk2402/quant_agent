"""Score the task-context parser against ground truth, across every public unit.

Ground truth = the filenames each unit's own `checks/test_outputs.py` actually opens
under OUTPUT_DIR.  That file is stripped from the submission mount, so the agent can
never see it -- which is exactly why it makes an honest held-out test set here.

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

from agent.task_context import (  # noqa: E402
    DATA_EXTS, NEVER_OUTPUT, _dockerfile_dests, _manifest_inputs, parse_task,
)

_EXT_ALT = "|".join(DATA_EXTS)
_LIT = re.compile(rf"""['"]([A-Za-z0-9_][A-Za-z0-9_.\-]*\.(?:{_EXT_ALT}))['"]""")


def ground_truth(unit: pathlib.Path) -> set[str]:
    """Filenames the checks read from the OUTPUT dir."""
    f = unit / "checks" / "test_outputs.py"
    if not f.is_file():
        return set()
    text = f.read_text(encoding="utf-8", errors="replace")

    # Names the checks join to the INPUT dir, or to the organizer's reference-data dir
    # (REF_DIR -> checks/reference_data), are not deliverables.
    input_side = set()
    for line in text.splitlines():
        if re.search(r"INPUT_DIR|/input|REF_DIR|reference_data", line):
            input_side.update(_LIT.findall(line))

    names = set(_LIT.findall(text)) - input_side - NEVER_OUTPUT
    names -= {pathlib.PurePosixPath(p).name for p in _manifest_inputs(unit)}
    names -= set(_dockerfile_dests(unit))
    return {n for n in names if not n.endswith(".py")}


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
        ctx = parse_task(u, probe_filesystem=False)
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
