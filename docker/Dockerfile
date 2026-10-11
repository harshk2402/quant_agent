# SPDX-License-Identifier: MIT
# Agenthon 2026 Track 1 submission image.
#
#   scripts/build.sh            the submission image, linux/amd64     -> quant-agent:dev
#   scripts/build.sh --arm64    a native image for local development  -> quant-agent:dev-arm64
#
# ARCHITECTURE IS NOT OPTIONAL. The fleet is x86-64; an arm64 image does not run, and building on
# Apple silicon produces one by default.
#
# The platform is a build argument so a native arm64 image can exist for local work: under amd64
# emulation on Apple silicon, anything that generates machine code at run time is extremely slow —
# numba's first import alone took over eight minutes. The argument DEFAULTS to amd64, so a plain
# `docker build` still produces a submittable image; arm64 has to be asked for explicitly.
#
# The two are separately tagged images, never one multi-arch manifest: the starter pack warns that
# a manifest resolving differently between runs could make a reproducibility rerun differ from
# itself. The arm64 image is for iteration speed only — never pushed, never submitted, and a result
# from it is not a result about the submission.
#
# WHY MULTI-STAGE. The evaluation worker pulls this image on every host that has not seen it, and
# that pull comes out of the unit's wall clock — measured by the organizers at 90–187s cold against
# ~15s warm, up to about 10% of an 1800s unit. Nothing is downloaded at run time, so image size is
# the only lever on that cost. The compiler toolchain exists solely to build TA-Lib, so it is
# confined to a builder stage and never reaches the image that gets pulled.
#
# The image must also be self-contained: PyPI is unreachable at run time, so every dependency the
# GENERATED code might import has to be baked in here. We cannot inherit finance-bench-sandbox,
# which exists only on the author's machine.

ARG TARGET_PLATFORM=linux/amd64

# =============================================================================================
# Stage 1 — builder. Compiles TA-Lib and installs every Python package into a virtualenv.
# Nothing from this stage ships except the venv and TA-Lib's shared libraries.
# =============================================================================================
FROM --platform=${TARGET_PLATFORM} python:3.13-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc g++ wget \
    && rm -rf /var/lib/apt/lists/*

# TA-Lib is a C library with a thin Python wrapper, and it is not in the Debian repositories, so it
# has to be compiled. No public task imports it — but the organizers' own sandbox carries it, the
# held-out tasks are authored separately from the ones we can read, and a missing library is not a
# degraded answer, it is a total loss on that task from an ImportError we would never see.
# Steps mirror docker/sandbox.Dockerfile in the practice kit.
RUN cd /tmp \
    && wget -q https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz \
    && tar xzf ta-lib-0.6.4-src.tar.gz \
    && cd ta-lib-0.6.4 \
    && ./configure --prefix=/usr \
    && make -j"$(nproc)" \
    && make install \
    && cd / && rm -rf /tmp/ta-lib*

# A virtualenv rather than the system site-packages, so the runtime stage can take the whole tree
# with one COPY and nothing else comes with it.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# The full sandbox stack. Version floors are copied from the kit's requirements-sandbox.txt rather
# than chosen: the tasks were authored against that environment, and diverging from it is the risk
# worth avoiding. Measured usage across the public instructions is arch (6 tasks), statsmodels (4),
# plotly (3), numba/polars/scikit-learn (1 each) — the rest are carried because the held-out tasks
# are unknown, and an import we cannot serve costs the whole task.
#
# yfinance is included for parity with the sandbox even though the network is disabled at run time;
# a script that imports it then fails on the fetch rather than on the import.
#
# numba is NOT in requirements-sandbox.txt, but the exemplar unit adds it in its own Dockerfile —
# a task author expected it to be importable, and at evaluation time OUR image runs, not the unit's,
# so a per-unit dependency exists only if we ship it. It was expected to force a numpy downgrade
# (numba 0.61 pinned numpy<2.3); measured here, pip resolves numba 0.67 alongside numpy 2.5, so it
# costs nothing.
RUN pip install --no-cache-dir \
        "numpy>=1.26" \
        "pandas>=2.1" \
        "scipy>=1.12" \
        "pyarrow>=15.0" \
        "ta-lib>=0.6" \
        "backtrader>=1.9" \
        "yfinance>=0.2" \
        "statsmodels>=0.14" \
        "scikit-learn>=1.4" \
        "arch>=6.3" \
        "polars>=1.0" \
        "matplotlib>=3.8" \
        "seaborn>=0.13" \
        "plotly>=5.18" \
        "openpyxl>=3.1" \
        "tqdm>=4.66" \
        "requests>=2.31" \
        "numba>=0.61.2"

# Bytecode and test trees are dead weight in a pulled image. Removing them is safe: Python
# regenerates bytecode on demand, and PYTHONDONTWRITEBYTECODE below stops it writing any back.
RUN find /opt/venv -type d -name "__pycache__" -prune -exec rm -rf {} + \
    && find /opt/venv -type d -name "tests" -prune -exec rm -rf {} + \
    && find /opt/venv -name "*.pyc" -delete

# =============================================================================================
# Stage 2 — the image that is actually pulled. No compiler, no headers, no apt lists.
# =============================================================================================
FROM --platform=${TARGET_PLATFORM} python:3.13-slim

# Required by SUBMISSION_CLI.md and genuinely enforced, and it must match submission.json. Without
# it no run record is produced at all: the submission comes back as an infrastructure refusal
# rather than a score.
LABEL qfbench2.interface_version="2.0"

# TA-Lib's runtime shared objects. The Python wrapper was compiled against them in the builder and
# loads them with dlopen, so the headers and static archives stay behind but these must come along.
COPY --from=builder /usr/lib/libta-lib.so* /usr/lib/

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/agent \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# The agent is stdlib-only; no install of our own package is needed, just an importable path.
# Last layer on purpose — this is the only one that changes between builds, so an agent edit
# rebuilds in seconds and leaves the expensive layers cached.
COPY agent/ /agent/agent/

# QFBench/Harbor convention: deliverables go to /app/output, which the harness binds (along with
# /output) to the same host directory.
RUN mkdir -p /app/output
WORKDIR /app

# `solve` arrives as the first argument after the image reference. A verbless ENTRYPOINT with the
# verb supplied by CMD, which the harness overrides with the same thing — the FinanceZero pattern.
# An image that does not accept the verb fails every unit before reading any input.
ENTRYPOINT ["python", "-m", "agent.solve"]
CMD ["solve", "--task-dir", "/input", "--out", "/app/output"]
