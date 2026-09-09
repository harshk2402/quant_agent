# SPDX-License-Identifier: MIT
"""Output-contract validation.  (Phase 0, Step 0.4)

This module answers one question before the container exits: *is what we are about to hand in
actually admissible?* It cannot tell us whether the numbers are right — the grading code that
decides that is the organizers' and may not be present — but almost every way a task scores zero
without being about the finance is detectable here:

  * a required file is missing, or is empty
  * a file does not parse as the format its name claims (a `.json` that is not JSON)
  * we wrote `reward.json`, which is the grader's file and not ours
  * a canary marker leaked into an output file

That last one is the reason this module can *act* rather than only report. A canary is a unique
marker string the organizers plant in task text so they can detect benchmark material leaking
where it should not be; one of the four admissibility checks scans every output text file for
them, and a single hit fails the task outright however correct the answer was. Detecting that and
doing nothing would still score zero, so `scrub_canaries` removes them.

Everything else is reported rather than fixed. The findings are structured so the repair loop
(Step 0.7) can feed them back to the model, which is why each carries a machine-readable `code`
alongside its human-readable message.
"""

from __future__ import annotations

import csv
import dataclasses
import io
import json
import os
import pathlib
import re

from .task_context import TaskContext, fmt_is_tabular

#: Written by the grader's own `test.sh`. Writing them ourselves corrupts the reward signal.
GRADER_FILES = {"reward.json", "reward.txt", "pytest_report.json"}

#: Text formats the admissibility scan reads, and therefore the ones a canary can hide in.
_SCANNED_SUFFIXES = {".py", ".json", ".txt", ".md", ".csv", ".log"}

_GUID = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                   r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")

ERROR = "error"
WARNING = "warning"


@dataclasses.dataclass
class Finding:
    level: str          # ERROR (would fail admissibility) | WARNING (worth knowing, not fatal)
    code: str           # stable identifier, for the repair loop to branch on
    message: str
    filename: str = ""

    def __str__(self) -> str:
        where = f" [{self.filename}]" if self.filename else ""
        return f"{self.level.upper()}: {self.code}{where} — {self.message}"


@dataclasses.dataclass
class ValidationReport:
    findings: list[Finding] = dataclasses.field(default_factory=list)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.level == ERROR]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.level == WARNING]

    @property
    def ok(self) -> bool:
        """True when nothing found would cost us the task."""
        return not self.errors

    def add(self, level: str, code: str, message: str, filename: str = "") -> None:
        self.findings.append(Finding(level, code, message, filename))

    def summary(self) -> str:
        if not self.findings:
            return "contract OK"
        return f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)"

    def repair_text(self) -> str:
        """The errors, phrased for the model. Consumed by the repair loop in Step 0.7."""
        return "\n".join(f"- {f.code}{f' ({f.filename})' if f.filename else ''}: {f.message}"
                         for f in self.errors)


# ---------------------------------------------------------------------------
# format checks
# ---------------------------------------------------------------------------

def _parses_as(path: pathlib.Path, fmt: str) -> str | None:
    """Return None when the file parses as `fmt`, else a short reason why it does not.

    The point is not to validate the finance but to catch a file that is structurally broken —
    truncated JSON, a CSV that is one unterminated line, a Parquet file that is actually text.
    Formats we cannot cheaply verify return None rather than guessing.
    """
    try:
        if fmt == "json":
            json.loads(path.read_text(encoding="utf-8", errors="replace"))
        elif fmt == "jsonl":
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.strip():
                    json.loads(line)
        elif fmt in ("csv", "tsv"):
            text = path.read_text(encoding="utf-8", errors="replace")
            rows = list(csv.reader(io.StringIO(text),
                                   delimiter="\t" if fmt == "tsv" else ","))
            if not rows or not any(any(c.strip() for c in r) for r in rows):
                return "no rows"
        elif fmt == "parquet":
            import pyarrow.parquet as pq
            pq.read_metadata(path)
        elif fmt == "png":
            if path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
                return "not a PNG (bad magic bytes)"
    except ImportError:
        return None                       # cannot check without the library; do not invent a fault
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    return None


# ---------------------------------------------------------------------------
# value-level checks ("tier 2")
# ---------------------------------------------------------------------------
#
# The two most common assertions across all 87 public graders are `columns` (48) and `row_count`
# (42), ahead of anything comparing against a reference answer. A large share of wrong answers is
# therefore structurally wrong rather than numerically wrong — and structure can be checked without
# knowing the correct answer, which is the only kind of check available to us at scoring time.
#
# Every rule below is deliberately conservative. A false alarm is worse than a missed problem: it
# sends the repair loop chasing something that was never broken, spending attempts and budget. So
# these fire only where the constraint is essentially always true of the quantity named.

#: column name pattern -> (low, high, level). Bounds are inclusive; None means unbounded.
_VALUE_RULES: list[tuple[re.Pattern, float | None, float | None, str]] = [
    # A price or premium is never negative. Restricted to names that clearly mean a price:
    # "value" and "pnl" are excluded because a position's value or P&L legitimately goes negative.
    (re.compile(r"^(price|premium|.*_price|price_.*)$", re.I), 0.0, None, ERROR),
    # A volatility is positive. The upper bound is loose on purpose — 1000% vol is implausible but
    # not impossible in a stressed synthetic task, so it warns rather than errors.
    (re.compile(r"^(sigma|vol|volatility|implied_vol.*|iv)$", re.I), 0.0, 10.0, WARNING),
    # A probability is a probability.
    (re.compile(r"^(prob|probability|.*_prob|prob_.*|pd)$", re.I), 0.0, 1.0, ERROR),
    # Discount factors live in (0, 1] for non-negative rates.
    (re.compile(r"^(df|discount_factor|discount)$", re.I), 0.0, 1.0, ERROR),
    # Correlations are bounded by construction.
    (re.compile(r"^(corr|correlation|rho)$", re.I), -1.0, 1.0, ERROR),
    # Delta of a vanilla option; a spread or portfolio delta can exceed this, so it warns.
    (re.compile(r"^delta$", re.I), -1.0, 1.0, WARNING),
]


def _frame_findings(path: pathlib.Path, spec, rep: ValidationReport) -> None:
    """Structural and value checks on a tabular deliverable.

    Skipped silently when pandas is unavailable: an unverifiable file is not a broken one.
    """
    try:
        import numpy as np
        import pandas as pd
    except ImportError:
        return
    try:
        df = (pd.read_parquet(path) if spec.fmt == "parquet"
              else pd.read_csv(path, sep="\t" if spec.fmt == "tsv" else ","))
    except Exception:
        return                                  # _parses_as already reported anything fatal

    if df.shape[0] == 0:
        rep.add(ERROR, "no_rows",
                "file has a header but no data rows", spec.filename)
        return

    num = df.select_dtypes(include="number")

    # NaN and infinity are almost always a computation that silently went wrong — a division by
    # zero, an unfilled join, a log of a non-positive number — rather than a deliberate answer.
    for col in num.columns:
        n_nan = int(num[col].isna().sum())
        if n_nan:
            rep.add(ERROR, "nan_values",
                    f"column '{col}' has {n_nan} NaN of {len(num)} rows", spec.filename)
        n_inf = int(np.isinf(num[col].to_numpy(dtype="float64", na_value=0.0)).sum())
        if n_inf:
            rep.add(ERROR, "inf_values",
                    f"column '{col}' has {n_inf} infinite value(s)", spec.filename)

    for col in num.columns:
        for pattern, low, high, level in _VALUE_RULES:
            if not pattern.match(str(col)):
                continue
            series = num[col].dropna()
            if series.empty:
                continue
            if low is not None and float(series.min()) < low:
                rep.add(level, "value_out_of_range",
                        f"column '{col}' has a minimum of {series.min():.6g}, below {low}",
                        spec.filename)
            if high is not None and float(series.max()) > high:
                rep.add(level, "value_out_of_range",
                        f"column '{col}' has a maximum of {series.max():.6g}, above {high}",
                        spec.filename)
            break


def _json_findings(path: pathlib.Path, spec, rep: ValidationReport) -> None:
    """NaN and Infinity in a JSON deliverable.

    `json.dumps` emits these as bare `NaN` / `Infinity`, which is valid to Python's parser but not
    to strict JSON. Either way it signals a computation that failed quietly.
    """
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    if re.search(r"\b(NaN|-?Infinity)\b", raw):
        rep.add(ERROR, "nan_values",
                "contains NaN or Infinity, which is neither valid JSON nor a usable answer",
                spec.filename)


def _header_of(path: pathlib.Path, fmt: str) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        first = next((ln for ln in text.splitlines() if ln.strip()), "")
        return [c.strip() for c in next(csv.reader(
            io.StringIO(first), delimiter="\t" if fmt == "tsv" else ","), [])]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# canaries
# ---------------------------------------------------------------------------

def known_canaries(task_dir: str | os.PathLike[str]) -> set[str]:
    """The canary markers belonging to THIS task, read from its own instruction and card.

    Deliberately narrow. Scrubbing every GUID-shaped token from the output would risk destroying a
    legitimate identifier that a task genuinely asked for; scrubbing only the markers that appear
    in the task's own text cannot.
    """
    task_dir = pathlib.Path(task_dir)
    found: set[str] = set()
    for name in ("instruction.md", "card.toml"):
        f = task_dir / name
        if f.is_file():
            try:
                found.update(_GUID.findall(f.read_text(encoding="utf-8", errors="replace")))
            except OSError:
                continue
    return found


def scrub_canaries(out_dir: str | os.PathLike[str], canaries: set[str]) -> list[str]:
    """Remove this task's canary markers from any output text file. Returns files changed.

    Active remediation rather than reporting, because a detected-but-unremoved canary still fails
    the task. Only the marker itself is replaced, so the surrounding data survives.
    """
    if not canaries:
        return []
    cleaned: list[str] = []
    for path in pathlib.Path(out_dir).rglob("*"):
        if not path.is_file() or path.suffix.lower() not in _SCANNED_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        hits = [c for c in canaries if c in text]
        if not hits:
            continue
        for c in hits:
            text = text.replace(c, "")
        try:
            path.write_text(text, encoding="utf-8")
            cleaned.append(path.name)
        except OSError:
            continue
    return cleaned


# ---------------------------------------------------------------------------
# the validator
# ---------------------------------------------------------------------------

def validate_outputs(ctx: TaskContext, out_dir: str | os.PathLike[str]) -> ValidationReport:
    """Inspect the output directory against the parsed contract. Read-only."""
    out = pathlib.Path(out_dir)
    rep = ValidationReport()

    if not ctx.outputs:
        rep.add(WARNING, "no_contract",
                "no deliverable filename was parsed, so nothing can be verified")

    for spec in ctx.outputs:
        path = spec.path_in(out)
        if not path.exists():
            rep.add(ERROR, "missing_file",
                    f"required deliverable was not written", spec.filename)
            continue
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        if size == 0:
            rep.add(ERROR, "empty_file", "file exists but is empty", spec.filename)
            continue

        why = _parses_as(path, spec.fmt)
        if why:
            rep.add(ERROR, "unparseable",
                    f"does not parse as {spec.fmt}: {why}", spec.filename)
            continue

        # Soft, and it must stay soft while column names come from prose. `spec.source` records
        # where the FILENAME was found, not where the columns were: column extraction always reads
        # the instruction text, and covers only about half of tabular deliverables. Escalating on
        # a prose guess would reject files that are very likely correct.
        #
        # This becomes an error only once columns are read from the unit's own checker, which is
        # what the grader actually asserts on. Not implemented yet.
        if fmt_is_tabular(spec.filename) and spec.columns:
            header = _header_of(path, spec.fmt)
            missing = [c for c in spec.columns if c not in header]
            if header and missing:
                rep.add(WARNING, "columns_missing",
                        f"header lacks {missing} (expected {spec.columns})", spec.filename)

        # Structure and values: the checks that catch an answer which is the right shape of file
        # but could not possibly be a correct result.
        try:
            if fmt_is_tabular(spec.filename):
                _frame_findings(path, spec, rep)
            elif spec.fmt == "json":
                _json_findings(path, spec, rep)
        except Exception:
            pass                                 # a validator must never be the thing that fails

    # Files the grader writes. Ours would be overwritten at best and misleading at worst.
    for name in sorted(GRADER_FILES):
        if (out / name).exists():
            rep.add(ERROR, "wrote_grader_file",
                    "this file belongs to the grader and must not be written by the agent", name)

    # Anything GUID-shaped left in a scanned file. scrub_canaries should already have removed the
    # markers we know about; this catches one arriving by a route we did not anticipate.
    for path in out.rglob("*"):
        if path.is_file() and path.suffix.lower() in _SCANNED_SUFFIXES:
            try:
                if _GUID.search(path.read_text(encoding="utf-8", errors="replace")):
                    rep.add(ERROR, "canary_suspect",
                            "contains a GUID-shaped token, which the admissibility scan may read "
                            "as a canary marker", path.name)
            except OSError:
                continue

    return rep
