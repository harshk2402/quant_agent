# SPDX-License-Identifier: MIT
# Agenthon 2026 Track 1 submission image.
#
#   scripts/build.sh            the submission image, linux/amd64     -> quant-agent:dev
#   scripts/build.sh --arm64    a native image for local development  -> quant-agent:dev-arm64
#
# ARCHITECTURE IS NOT OPTIONAL. The fleet is x86-64; an arm64 image does not run, and building on
# Apple silicon produces one by default (RUNTIME-ENVIRONMENT.md).
#
# The platform is a build argument so that a native arm64 image can exist for local work: under
# amd64 emulation on Apple silicon, anything that generates machine code at run time is extremely
# slow — numba's first import alone took over eight minutes. The argument DEFAULTS to amd64, so a
# plain `docker build` with no arguments still produces a submittable image; arm64 has to be asked
# for explicitly.
#
# The two are kept as separately tagged images, never combined into one multi-arch manifest. The
# starter pack warns against assuming a multi-arch manifest is safe: one that resolves differently
# between runs could make the organizers' reproducibility rerun differ from itself.
#
# The arm64 image is for iteration speed only. It is never pushed and never submitted, and a result
# from it is not a result about the submission — architectures genuinely differ, both in available
# wheels and in floating-point output. Anything that gates a decision must be checked on amd64.
#
# We cannot inherit finance-bench-sandbox:latest -- that image exists only on the author's machine.
# The image must be self-contained: PyPI is unreachable at run time, so every dependency the
# GENERATED code might import has to be baked in here.
ARG TARGET_PLATFORM=linux/amd64
FROM --platform=${TARGET_PLATFORM} python:3.13-slim

# Required by SUBMISSION_CLI.md:28 and genuinely enforced; must match submission.json.
LABEL qfbench2.interface_version="2.0"

# ---------------------------------------------------------------------------------------------
# Layers below are ordered slowest-and-most-stable first, so that an edit to the agent rebuilds in
# seconds while the C compile and the package install stay cached.
# ---------------------------------------------------------------------------------------------

# Toolchain, needed only to compile TA-Lib below.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc g++ wget \
    && rm -rf /var/lib/apt/lists/*

# TA-Lib is a C library with a thin Python wrapper, and it is not in the Debian repositories, so it
# has to be compiled. No public task imports it -- but the organizers' own sandbox carries it, the
# ~30 held-out tasks are authored separately from the 87 we can read, and a missing library is not a
# degraded answer, it is a total loss on that task from an ImportError we would never see. The build
# is slow under amd64 emulation and is paid once; caching keeps it out of the iteration loop.
# Steps mirror docker/sandbox.Dockerfile in the practice kit.
RUN cd /tmp \
    && wget -q https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz \
    && tar xzf ta-lib-0.6.4-src.tar.gz \
    && cd ta-lib-0.6.4 \
    && ./configure --prefix=/usr \
    && make -j"$(nproc)" \
    && make install \
    && cd / && rm -rf /tmp/ta-lib*

# The full sandbox stack. Version floors are copied from the kit's requirements-sandbox.txt rather
# than chosen: the tasks were authored against that environment, and diverging from it is the risk
# worth avoiding. Measured usage across the 87 public instructions is arch (6 tasks), statsmodels
# (4), plotly (3), numba/polars/scikit-learn (1 each) -- the rest are carried because the held-out
# tasks are unknown.
#
# yfinance is included for parity with the sandbox even though the network is disabled at run time;
# a script that imports it will at least fail on the fetch rather than on the import.
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

# numba is deliberately listed even though it is NOT in the kit's requirements-sandbox.txt. The
# exemplar unit adds it in its own Dockerfile, which tells us a task author expected it to be
# importable -- and at evaluation time OUR image runs, not the unit's, so a per-unit dependency
# exists only if we ship it.
#
# It was expected to cost a numpy downgrade: numba 0.61 pinned numpy<2.3, which is why the kit
# notes that the exemplar unit's image drops to numpy 2.2.6. That no longer applies. Measured on
# this build, pip resolved numba 0.67.0 alongside numpy 2.5.3, so the dependency comes free and
# the image keeps the same numpy the bare sandbox carries.

# The agent is stdlib-only; no pip install of our own package is needed, just an importable path.
# Last layer on purpose -- this is the only one that changes between builds.
COPY agent/ /agent/agent/
ENV PYTHONPATH=/agent \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# QFBench/Harbor convention: deliverables go to /app/output, which the harness binds (along with
# /output) to the same host directory.
RUN mkdir -p /app/output
WORKDIR /app

# The verb arrives as the first argument after the image reference. This is the FinanceZero
# pattern: a verbless ENTRYPOINT, with the verb supplied by CMD and overridden by the harness.
ENTRYPOINT ["python", "-m", "agent.solve"]
CMD ["solve", "--task-dir", "/input", "--out", "/app/output"]
