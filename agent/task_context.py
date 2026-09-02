"""Task-context parser for QFBench 2.0 Track 1 units.  (Phase 0, Step 0.1)

Why this module exists
----------------------
The single biggest scoring trap in this competition (SUBMISSION_CLI.md invariant 9,
NVIDIA issue #17) is that the deliverable filename and shape are **per-unit**.  An agent
that solves the exemplar and generalises writes `results.parquet` everywhere; the next
unit wanted `results.json` and the run scored zero for a filename despite real work.
Across the 87 public units only four deliverables are parquet, while `results.json` (38)
and `solution.json` (19) dominate.

The unit's own `checks/test_outputs.py` would answer this exactly -- but it is STRIPPED
from the submission-facing mount (scripts/build_dev_dataset.py: T1_ANSWER_DIRS =
("reference", "checks")), because it carries the graded answers.  So at solve time the
only ground truth is prose.  This module turns that prose into a structured TaskContext.

Three facts measured across all 87 public units that shape the design:

1.  Deliverables are frequently MULTIPLE files (up to 15 in one unit), so the parser
    returns a list, never a single filename.
2.  Section headings are wildly inconsistent (`## Output`, `## Output Files`,
    `## Deliverables`, `## Required outputs`, `## Step 11: Output Files`,
    `## Output: /app/output/results.json`, ...).  Heading-name matching alone is not
    enough; the parser layers section scoping with explicit-path scanning.
3.  Inputs are RELOCATED by each unit's own environment/Dockerfile (`COPY data/ /app/`,
    `COPY data/params.json /app/data/params.json`, ...).  The exemplar reading
    `/input/environment/data/options.parquet` is the exception, not the pattern -- most
    units read `/app/...`.  So input paths are resolved by probing the real filesystem.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import re
from typing import Iterable

try:  # Python 3.11+; the sandbox runs 3.13
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]


# Extensions that can plausibly be a deliverable or an input dataset.
DATA_EXTS = (
    "json", "jsonl", "csv", "tsv", "parquet", "pqt",
    "html", "png", "txt", "md", "xlsx", "xml", "zip", "py",
)
_EXT_ALT = "|".join(DATA_EXTS)

# The harness writes these itself; instruction.md explicitly says "do not write this file".
NEVER_OUTPUT = {"reward.json", "pytest_report.json", "reward.txt"}

FORMAT_BY_EXT = {
    "json": "json", "jsonl": "jsonl", "csv": "csv", "tsv": "tsv",
    "parquet": "parquet", "pqt": "parquet", "html": "html", "png": "png",
    "txt": "text", "md": "markdown", "xlsx": "xlsx", "xml": "xml",
    "zip": "zip", "py": "python",
}

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
# A filename, optionally prefixed by an explicit output directory.
_FILE_RE = re.compile(
    rf"(?P<dir>/(?:app/)?output/)?(?P<name>[A-Za-z0-9_][A-Za-z0-9_.\-]*\.(?:{_EXT_ALT}))\b"
)
_OUTPATH_RE = re.compile(
    rf"/(?:app/)?output/(?P<name>[A-Za-z0-9_][A-Za-z0-9_.\-]*\.(?:{_EXT_ALT}))\b"
)
# Any absolute path that is NOT under an output dir -- i.e. an input the prose names.
_INPATH_RE = re.compile(
    rf"(?P<path>/(?:input|app)/[A-Za-z0-9_./\-]*[A-Za-z0-9_\-]\.(?:{_EXT_ALT}))\b"
)


def _is_output_heading(title: str) -> bool:
    """True for headings that introduce deliverables.

    Checked against every distinct heading in the 87 public units.  `input` is vetoed
    first so `## Input Files` never matches on the word "file".
    """
    if re.search(r"\binputs?\b", title, re.I):
        return False
    return bool(re.search(r"\b(outputs?|deliverabl\w*|artifacts?)\b", title, re.I))


@dataclasses.dataclass
class OutputSpec:
    """One deliverable the unit expects."""
    filename: str                     # e.g. "results.json"
    fmt: str                          # e.g. "json"
    declared_path: str | None = None  # exact path as written, e.g. "/app/output/results.json"
    schema_text: str = ""             # the markdown describing this file, verbatim, for the LLM
    columns: list[str] = dataclasses.field(default_factory=list)

    def path_in(self, output_dir: str | os.PathLike[str]) -> pathlib.Path:
        return pathlib.Path(output_dir) / self.filename


@dataclasses.dataclass
class InputSpec:
    """One input file, with every place it might actually live at runtime."""
    filename: str
    fmt: str
    declared_path: str | None = None     # path as instruction.md writes it
    candidates: list[str] = dataclasses.field(default_factory=list)
    resolved: str | None = None          # first candidate that exists on disk

    def path(self) -> pathlib.Path | None:
        return pathlib.Path(self.resolved) if self.resolved else None


@dataclasses.dataclass
class TaskContext:
    """Everything the solve loop needs to know about one unit."""
    task_dir: pathlib.Path
    unit_id: str = ""
    title: str = ""
    category: str = ""
    difficulty: str = ""
    instruction: str = ""
    outputs: list[OutputSpec] = dataclasses.field(default_factory=list)
    inputs: list[InputSpec] = dataclasses.field(default_factory=list)
    agent_timeout_sec: float | None = None
    cpus: int | None = None
    memory: str = ""
    network: str = ""
    warnings: list[str] = dataclasses.field(default_factory=list)

    # -- convenience -------------------------------------------------------
    @property
    def output_filenames(self) -> list[str]:
        return [o.filename for o in self.outputs]

    @property
    def primary_output(self) -> OutputSpec | None:
        return self.outputs[0] if self.outputs else None

    def resolved_inputs(self) -> list[InputSpec]:
        return [i for i in self.inputs if i.resolved]

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["task_dir"] = str(self.task_dir)
        return d


# ---------------------------------------------------------------------------
# markdown sectioning
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class _Section:
    level: int
    title: str
    body: str


def _split_sections(md: str) -> list[_Section]:
    """Split markdown into sections; a section's body includes its subsections.

    Fenced code blocks are skipped so a `#` comment inside python doesn't become a
    heading.
    """
    lines = md.splitlines()
    heads: list[tuple[int, int, str]] = []  # (line_idx, level, title)
    in_fence = False
    for i, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = _HEADING_RE.match(line)
        if m:
            heads.append((i, len(m.group(1)), m.group(2)))

    out: list[_Section] = []
    for n, (idx, level, title) in enumerate(heads):
        end = len(lines)
        for j in range(n + 1, len(heads)):
            if heads[j][1] <= level:      # next heading of same or higher rank
                end = heads[j][0]
                break
        out.append(_Section(level, title, "\n".join(lines[idx:end])))
    return out


def fmt_is_tabular(filename: str) -> bool:
    """Only tabular deliverables have columns; a JSON's shape lives in schema_text."""
    return filename.rsplit(".", 1)[-1].lower() in {"csv", "tsv", "parquet", "pqt"}


def _columns_from(body: str) -> list[str]:
    """Best-effort column extraction: markdown table first, then backticked CSV header."""
    cols: list[str] = []
    for line in body.splitlines():
        s = line.strip()
        if s.startswith("|") and s.count("|") >= 3:
            cells = [c.strip().strip("`*") for c in s.strip("|").split("|")]
            if not cells:
                continue
            head = cells[0]
            # skip separator rows and the header row itself
            if set(head) <= set("-: ") or head.lower() in {"column", "field", "key", "name"}:
                continue
            if re.fullmatch(r"[A-Za-z0-9_.\-]+", head):
                cols.append(head)
    if cols:
        return list(dict.fromkeys(cols))
    # `a, b, c` style column list
    for m in re.finditer(r"`([A-Za-z0-9_.\-]+(?:\s*,\s*[A-Za-z0-9_.\-]+){2,})`", body):
        return [c.strip() for c in m.group(1).split(",")]
    # Bare CSV header inside an indented or fenced code block, e.g.
    #     ticker,type,signal_date,exec_date,price,shares,pnl
    in_fence = False
    for raw in body.splitlines():
        if raw.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if not (in_fence or raw.startswith("    ") or raw.startswith("\t")):
            continue
        line = raw.strip()
        if not re.fullmatch(r"[A-Za-z0-9_.\-]+(?:\s*,\s*[A-Za-z0-9_.\-]+){2,}", line):
            continue
        parts = [c.strip() for c in line.split(",")]
        # Reject numeric data rows; a header has real field names.
        if all(re.fullmatch(r"-?[\d.]+", c) for c in parts):
            continue
        return parts
    return []


# ---------------------------------------------------------------------------
# Dockerfile / manifest -> where inputs really live
# ---------------------------------------------------------------------------

def _dockerfile_dests(task_dir: pathlib.Path) -> dict[str, str]:
    """Map input basename -> runtime path, from the unit's environment/Dockerfile.

    The build context is `environment/`, so `COPY data/x.csv /app/x.csv` refers to
    `environment/data/x.csv` in the repo and lands at `/app/x.csv` in the container.
    """
    dockerfile = task_dir / "environment" / "Dockerfile"
    env_dir = task_dir / "environment"
    dests: dict[str, str] = {}
    if not dockerfile.is_file():
        return dests
    try:
        text = dockerfile.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return dests

    text = re.sub(r"\\\s*\n", " ", text)  # join line continuations
    for line in text.splitlines():
        s = line.strip()
        if not re.match(r"^(COPY|ADD)\b", s, re.I):
            continue
        toks = [t for t in s.split()[1:] if not t.startswith("--")]
        if len(toks) < 2:
            continue
        srcs, dst = toks[:-1], toks[-1]
        for src in srcs:
            src_path = (env_dir / src.rstrip("/")).resolve()
            dst_is_dir = dst.endswith("/") or "." not in pathlib.PurePosixPath(dst).name
            if src_path.is_dir():
                base = dst if dst_is_dir else dst + "/"
                for f in src_path.rglob("*"):
                    if f.is_file():
                        rel = f.relative_to(src_path).as_posix()
                        dests[f.name] = base.rstrip("/") + "/" + rel
            else:
                name = src_path.name
                dests[name] = dst.rstrip("/") + "/" + name if dst_is_dir else dst
    return dests


def _manifest_inputs(task_dir: pathlib.Path) -> list[str]:
    """Repo-relative paths of files manifest.json declares with role == 'input'."""
    mf = task_dir / "manifest.json"
    if not mf.is_file():
        return []
    try:
        data = json.loads(mf.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [
        f["path"] for f in data.get("files", [])
        if isinstance(f, dict) and f.get("path") and f.get("role", "input") == "input"
    ]


# ---------------------------------------------------------------------------
# card.toml
# ---------------------------------------------------------------------------

def _read_card(task_dir: pathlib.Path) -> dict:
    card = task_dir / "card.toml"
    if not card.is_file() or tomllib is None:
        return {}
    try:
        with card.open("rb") as fh:
            return tomllib.load(fh)
    except (OSError, ValueError):
        return {}


# ---------------------------------------------------------------------------
# output-dir resolution (mirrors checks/test.sh exactly)
# ---------------------------------------------------------------------------

def resolve_output_dir(explicit: str | os.PathLike[str] | None = None) -> pathlib.Path:
    """Pick the deliverable directory the way the unit's own checks/test.sh does.

    Precedence: explicit --out, then $OUTPUT_DIR, then /app/output if it exists,
    else /output.  The harness binds the same host dir at both paths, so either is
    safe -- but matching test.sh removes any doubt.
    """
    if explicit:
        return pathlib.Path(explicit)
    env = os.environ.get("OUTPUT_DIR")
    if env:
        return pathlib.Path(env)
    if pathlib.Path("/app/output").is_dir():
        return pathlib.Path("/app/output")
    return pathlib.Path("/output")


# ---------------------------------------------------------------------------
# the parser
# ---------------------------------------------------------------------------

def _gather_output_candidates(
    md: str, input_basenames: set[str]
) -> list[tuple[str, str | None, str]]:
    """Return ordered (filename, declared_path, schema_text) triples.

    Layered, because heading names are not reliable across the corpus:
      1. every explicit `/app/output/X` or `/output/X` path anywhere in the doc
         -- the strongest possible signal, and unambiguous;
      2. filenames named in output-flagged sections (heading title included, so
         `## Output: \\`/app/output/results.json\\`` is caught);
      3. if still nothing, bare backticked filenames in those sections.
    Anything the unit reads as input is subtracted so "load prices.csv, write
    results.json" cannot mistake the input for a deliverable.
    """
    found: dict[str, tuple[str | None, str]] = {}
    sections = _split_sections(md)

    def add(name: str, declared: str | None, schema: str) -> None:
        if name in NEVER_OUTPUT or name in input_basenames:
            return
        prev = found.get(name)
        if prev is None:
            found[name] = (declared, schema)
        else:
            # Sections are visited deepest-first, so the FIRST non-empty schema we see is
            # the tightest one (the file's own subsection). A later, shallower section is
            # its parent and describes every sibling file too -- never let it overwrite.
            found[name] = (prev[0] or declared, prev[1] or schema)

    # Layer 1 -- explicit output paths, in document order.
    for m in _OUTPATH_RE.finditer(md):
        add(m.group("name"), m.group(0), "")

    # Layer 2 -- output-flagged sections. Deepest sections first so a per-file
    # subsection supplies a tighter schema_text than its parent.
    out_sections = [s for s in sections if _is_output_heading(s.title)]
    covered: list[_Section] = []
    for s in out_sections:
        covered.append(s)
        covered.extend(
            sub for sub in sections
            if sub is not s and sub.body and sub.body in s.body and sub.level > s.level
        )
    for s in sorted(covered, key=lambda x: -x.level):
        for m in _FILE_RE.finditer(s.body):
            add(m.group("name"), m.group("dir") and m.group(0), s.body)

    # Layer 3 -- last resort: a section-less doc that still names files.
    if not found:
        for s in sections:
            if re.search(r"\bwrite|save|produce|emit\b", s.body, re.I):
                for m in _FILE_RE.finditer(s.body):
                    add(m.group("name"), m.group("dir") and m.group(0), s.body)

    return [(n, d, t) for n, (d, t) in found.items()]


def parse_task(
    task_dir: str | os.PathLike[str] = "/input",
    *,
    probe_filesystem: bool = True,
) -> TaskContext:
    """Parse one unit directory into a TaskContext.

    `probe_filesystem=False` keeps it pure (no disk probing for input resolution),
    which is what the offline validator uses.
    """
    task_dir = pathlib.Path(task_dir)
    ctx = TaskContext(task_dir=task_dir)

    instr = task_dir / "instruction.md"
    if instr.is_file():
        ctx.instruction = instr.read_text(encoding="utf-8", errors="replace")
    else:
        ctx.warnings.append("instruction.md missing -- cannot determine deliverables")

    card = _read_card(task_dir)
    task = card.get("task", {})
    meta = card.get("metadata", {})
    env = card.get("environment", {})
    ctx.unit_id = task.get("id", "") or task_dir.name
    ctx.title = task.get("title", "")
    ctx.category = meta.get("category", "")
    ctx.difficulty = meta.get("difficulty", "")
    ctx.agent_timeout_sec = card.get("agent", {}).get("timeout_sec")
    ctx.cpus = env.get("cpus")
    ctx.memory = env.get("memory", "")
    ctx.network = env.get("network", "")
    if not card:
        ctx.warnings.append("card.toml absent or unreadable -- proceeding on instruction.md alone")

    # --- inputs -----------------------------------------------------------
    docker_dests = _dockerfile_dests(task_dir)
    manifest_paths = _manifest_inputs(task_dir)
    declared_in: dict[str, str] = {}
    for m in _INPATH_RE.finditer(ctx.instruction):
        p = m.group("path")
        name = pathlib.PurePosixPath(p).name
        if "/output/" in p or name in NEVER_OUTPUT:
            continue
        declared_in.setdefault(name, p)

    names: list[str] = []
    names += [pathlib.PurePosixPath(p).name for p in manifest_paths]
    names += [n for n in docker_dests if n not in names]
    names += [n for n in declared_in if n not in names]

    for name in dict.fromkeys(names):
        cands: list[str] = []
        if name in declared_in:
            cands.append(declared_in[name])
        if name in docker_dests:
            cands.append(docker_dests[name])
        for rel in manifest_paths:
            if pathlib.PurePosixPath(rel).name == name:
                cands.append(f"/input/{rel}")
                cands.append(str(task_dir / rel))
        cands += [f"/app/data/{name}", f"/app/{name}", f"/input/environment/data/{name}"]
        cands = list(dict.fromkeys(cands))

        spec = InputSpec(
            filename=name,
            fmt=FORMAT_BY_EXT.get(name.rsplit(".", 1)[-1].lower(), "unknown"),
            declared_path=declared_in.get(name),
            candidates=cands,
        )
        if probe_filesystem:
            for c in cands:
                if pathlib.Path(c).is_file():
                    spec.resolved = c
                    break
            if spec.resolved is None:
                for root in ("/app", "/input", str(task_dir)):
                    rp = pathlib.Path(root)
                    if not rp.is_dir():
                        continue
                    hit = next((p for p in rp.rglob(name) if p.is_file()), None)
                    if hit:
                        spec.resolved = str(hit)
                        break
            if spec.resolved is None:
                ctx.warnings.append(f"input {name!r} not found on disk; tried {cands}")
        ctx.inputs.append(spec)

    # --- outputs ----------------------------------------------------------
    input_basenames = {i.filename for i in ctx.inputs}
    for name, declared, schema in _gather_output_candidates(ctx.instruction, input_basenames):
        ctx.outputs.append(
            OutputSpec(
                filename=name,
                fmt=FORMAT_BY_EXT.get(name.rsplit(".", 1)[-1].lower(), "unknown"),
                declared_path=declared,
                schema_text=schema.strip(),
                columns=_columns_from(schema) if fmt_is_tabular(name) else [],
            )
        )
    if not ctx.outputs:
        ctx.warnings.append(
            "NO deliverable filename found in instruction.md -- do NOT guess "
            "results.parquet from the exemplar (issue #17); escalate to the LLM."
        )
    return ctx


def render_output_contract(ctx: TaskContext) -> str:
    """A compact, unambiguous statement of the output contract, for the LLM prompt."""
    if not ctx.outputs:
        return "OUTPUT CONTRACT: could not be parsed. Re-read instruction.md yourself."
    lines = ["OUTPUT CONTRACT (exact filenames -- do not rename, do not add, do not omit):"]
    for o in ctx.outputs:
        cols = f"  columns: {', '.join(o.columns)}" if o.columns else ""
        lines.append(f"  - {o.filename}  [{o.fmt}]{cols}")
    lines.append("Write these into the --out directory. Never write reward.json.")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    import sys
    c = parse_task(sys.argv[1] if len(sys.argv) > 1 else "/input")
    print(json.dumps(c.to_dict(), indent=2, default=str))
