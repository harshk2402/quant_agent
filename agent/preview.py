# SPDX-License-Identifier: MIT
"""What is actually inside each input file, for the prompt.

WHY. The parser resolves input *paths* and the prompt states them authoritatively, but until now
nothing looked *inside* them. So the model was told to read `pair_conventions.json` and had to
invent what it contained -- and it invented wrong. Measured on seven units whose scripts now
assemble and run, five of six crashes were a KeyError on a guessed key:

    pair_conventions["EURUSD"]   the file is {"_comment": ..., "pairs": {...}}
    inst["issue_date"]           records carry id, type, coupon_rate, frequency, notional,
                                 maturity_date, day_count -- and no issue date at all
    df["CIK"], merge on "asof_date", tickers read as columns when they are row values

One wrong guess at line 146 of a 400-line script throws away the whole run, and no amount of
repair budget recovers it efficiently: the traceback names the key but not the real schema, so the
next attempt guesses again. Showing the schema up front costs a few hundred tokens of input, which
is the one resource that is not capped.

WHAT IS SHOWN. Enough to bind a name to a shape, and no more: for a table its columns and a row or
two of real values, for a JSON document its key structure to a few levels. Sample values matter as
much as names -- whether `maturity_date` holds "2033-05-15" or 3.5 decides whether the script works.

FOUR RULES THIS MODULE KEEPS.

  * **Bounded.** Per-file and overall caps, because an input can be a hundred megabytes and the
    prompt is not free. A preview is a sketch; the script reads the real file.
  * **Cheap.** Headers and metadata, a bounded read, never a full parse of a large file.
  * **Canary-stripped.** Several units ship a `fetch_data.py` beside their data with the marker in
    plain sight, and the whole point of stripping it from the instruction is undone by quoting it
    back out of a data directory.
  * **Silent on failure.** A file that cannot be described is simply not described. A preview is a
    convenience; raising here would cost the task.
"""

from __future__ import annotations

import csv
import io
import json
import pathlib

#: Characters any one file's preview may use. Generous enough for a wide table's columns plus two
#: sample rows, tight enough that ten inputs cannot crowd out the task description.
MAX_PER_FILE = 900

#: Characters for all previews together. Input is the one resource the route does not cap, and a
#: unit with fourteen data files needs every one of them sketched: a first-come budget meant the
#: last few got nothing, and `pair_conventions` was still indexed wrongly on a run where the files
#: ahead of it had used the allowance up.
MAX_TOTAL = 14000

#: No file gets less than this, however many there are. Below it a sketch is not worth having.
MIN_PER_FILE = 260

#: Rows of real data shown under a table's header.
SAMPLE_ROWS = 2

#: How deep into a JSON document to describe, and how many keys at each level.
JSON_DEPTH = 3
JSON_KEYS = 12

#: Bytes read from a file for a preview. A header and a couple of rows are always at the front.
READ_BYTES = 256 * 1024

#: Suffixes previewed as data. A `.py` input is named but never quoted: it is usually a fetch script
#: carrying a canary, and when it IS the deliverable (the Polars migration unit) the instruction
#: already describes it.
_TABULAR = {".csv", ".tsv"}
_JSONISH = {".json", ".jsonl", ".ndjson"}
#: `.pqt` is Parquet under a shortened extension. Three units in the public roster use it, and
#: matching only `.parquet` silently skipped all of them.
_PARQUET = {".parquet", ".pqt"}


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _cell(value: object, limit: int = 28) -> str:
    """One sample value, shortened. Keeps the type legible: dates stay dates, floats stay floats."""
    text = "" if value is None else str(value)
    text = text.replace("\n", " ").replace("\r", " ").strip()
    return _truncate(text, limit) if text else "<empty>"


# ---------------------------------------------------------------------------
# per-format descriptions
# ---------------------------------------------------------------------------

def _describe_table(path: pathlib.Path, delimiter: str) -> str:
    raw = path.read_bytes()[:READ_BYTES].decode("utf-8", errors="replace")
    rows = list(csv.reader(io.StringIO(raw), delimiter=delimiter))
    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows:
        return ""
    header = rows[0]
    out = [f"columns ({len(header)}): " + ", ".join(header)]
    for row in rows[1:1 + SAMPLE_ROWS]:
        # Paired with the column name rather than printed as a bare tuple: on a 40-column table a
        # positional row is unreadable, and the point is to bind each NAME to a kind of value.
        pairs = [f"{h}={_cell(v)}" for h, v in zip(header, row)]
        out.append("  row: " + ", ".join(pairs))
    return "\n".join(out)


def _json_shape(node: object, depth: int = 0) -> str:
    """A compact sketch of a JSON value's structure.

    Describes shape, not contents: keys and types, with a short scalar shown inline because the
    value is often what disambiguates (a rate as 0.0543 rather than 5.43 changes the arithmetic).
    """
    pad = "  " * depth
    if isinstance(node, dict):
        if depth >= JSON_DEPTH:
            return f"{{{len(node)} keys: " + ", ".join(list(node)[:JSON_KEYS]) + "}"
        lines = []
        for key in list(node)[:JSON_KEYS]:
            child = _json_shape(node[key], depth + 1)
            sep = "\n" if "\n" in child else " "
            lines.append(f"{pad}{key}:{sep}{child}")
        if len(node) > JSON_KEYS:
            lines.append(f"{pad}... {len(node) - JSON_KEYS} more key(s)")
        return "\n".join(lines)
    if isinstance(node, list):
        if not node:
            return "[] (empty list)"
        first = _json_shape(node[0], depth + 1)
        sep = "\n" if "\n" in first else " "
        return f"[{len(node)} item(s)], each like:{sep}{first}"
    if isinstance(node, bool) or node is None:
        return f"<{json.dumps(node)}>"
    if isinstance(node, (int, float)):
        return f"<number, e.g. {node}>"
    return f'<string, e.g. "{_cell(node, 40)}">'


def _describe_json(path: pathlib.Path) -> str:
    text = path.read_bytes()[:READ_BYTES].decode("utf-8", errors="replace")
    try:
        return _json_shape(json.loads(text))
    except ValueError:
        # Either a JSONL file, or a document too big for the bounded read. The first line of a
        # JSONL file is a whole record, which is exactly the shape we want to show.
        first = next((ln for ln in text.splitlines() if ln.strip()), "")
        try:
            return "one record per line, each like:\n" + _json_shape(json.loads(first), 1)
        except ValueError:
            return ""


def _describe_parquet(path: pathlib.Path) -> str:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return ""
    schema = pq.read_schema(path)
    cols = [f"{name} ({dtype})" for name, dtype in zip(schema.names, schema.types)]
    return f"columns ({len(cols)}): " + ", ".join(cols)


def _describe_archive(path: pathlib.Path) -> str:
    """The entries in a zip, which is what the script has to navigate.

    The SEC filing units ship XBRL archives, and a script cannot open a member it does not know
    the name of. Names only, never contents: the files inside may be very large, and one level of
    naming is what the guessing problem needs.
    """
    import zipfile
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if not n.endswith("/")]
    shown = names[:20]
    more = f"\n... {len(names) - len(shown)} more entr(ies)" if len(names) > len(shown) else ""
    return f"zip archive, {len(names)} entr(ies):\n  " + "\n  ".join(shown) + more


def _describe_excel(path: pathlib.Path) -> str:
    try:
        import openpyxl
    except ImportError:
        return ""
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        out = [f"sheets: {', '.join(book.sheetnames)}"]
        sheet = book[book.sheetnames[0]]
        for i, row in enumerate(sheet.iter_rows(max_row=1 + SAMPLE_ROWS, values_only=True)):
            label = "columns" if i == 0 else "  row"
            out.append(f"{label}: " + ", ".join(_cell(v) for v in row))
        return "\n".join(out)
    finally:
        book.close()


def describe_file(path: str | pathlib.Path, limit: int = MAX_PER_FILE) -> str:
    """A bounded, canary-free sketch of one input file, or "" when it cannot be described."""
    from .prompt import strip_canary

    path = pathlib.Path(path)
    try:
        if not path.is_file():
            return ""
        suffix = path.suffix.lower()
        if suffix in _TABULAR:
            body = _describe_table(path, "\t" if suffix == ".tsv" else ",")
        elif suffix in _JSONISH:
            body = _describe_json(path)
        elif suffix in _PARQUET:
            body = _describe_parquet(path)
        elif suffix in (".xlsx", ".xls"):
            body = _describe_excel(path)
        elif suffix == ".zip":
            body = _describe_archive(path)
        else:
            return ""
    except Exception:
        return ""                     # a preview is a convenience and must never cost the task
    return _truncate(strip_canary(body).strip(), limit)


def describe_inputs(ctx) -> str:
    """The preview block for every resolved input, or "" when there is nothing to show.

    The budget is shared out in advance rather than spent first-come. With fourteen inputs, a
    running total means the files at the end of the list are described as "not previewed" -- and
    which file a script needs most is not the order they happen to be listed in.
    """
    resolved = [spec for spec in ctx.inputs if spec.resolved]
    if not resolved:
        return ""
    share = max(MIN_PER_FILE, min(MAX_PER_FILE, MAX_TOTAL // len(resolved)))

    blocks: list[str] = []
    for spec in resolved:
        body = describe_file(spec.resolved, limit=share)
        if body:
            blocks.append(f"### `{spec.filename}`\n{body}")
    return "\n\n".join(blocks)
