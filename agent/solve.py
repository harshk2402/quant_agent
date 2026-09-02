# SPDX-License-Identifier: MIT
"""The `solve` verb — the CLI contract the harness invokes.  (Phase 0, Step 0.3 skeleton)

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

This skeleton writes correctly-named, well-formed PLACEHOLDER deliverables. That is deliberate:
it makes the whole contract — parser, verb, packaging, architecture, mounts — verifiable across
all 87 units with no model calls at all. The generation step replaces `_placeholder_payload`.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback

from .task_context import OutputSpec, TaskContext, parse_task, resolve_output_dir


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


def solve(task_dir: str, out: str | None) -> int:
    """Solve one unit. Returns a process exit code -- and it is always 0 (see rule 2)."""
    out_dir = resolve_output_dir(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        ctx = parse_task(task_dir)
    except Exception:
        # Even the parser failing must not crash the run. Nothing is known about the contract
        # here, so there is nothing safe to write -- exit 0 and let the gates record it.
        traceback.print_exc(file=sys.stderr)
        return 0

    print(f"[solve] {ctx.unit_id or task_dir}: "
          f"{len(ctx.inputs)} input(s), {len(ctx.outputs)} deliverable(s) -> {out_dir}",
          file=sys.stderr)
    for w in ctx.warnings:
        print(f"[solve] warning: {w}", file=sys.stderr)

    for spec in ctx.outputs:
        try:
            _write_placeholder(spec, ctx, out_dir)
        except Exception:
            traceback.print_exc(file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="solve", add_help=True)
    # The verb is positional because the harness passes it as the first argument. Accepting it
    # (rather than rejecting an "unexpected" argument) is the whole of contract rule 1.
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
