# SPDX-License-Identifier: MIT
# Agenthon 2026 Track 1 submission image.
#
#   docker buildx build --platform linux/amd64 -f docker/Dockerfile -t quant-agent:dev --load .
#
# ARCHITECTURE IS NOT OPTIONAL. The fleet is x86-64; an arm64 image does not run, and building on
# Apple silicon produces one by default (RUNTIME-ENVIRONMENT.md). Hence the explicit --platform on
# the FROM line as well, so a plain `docker build` cannot silently produce the wrong arch.
#
# We cannot inherit finance-bench-sandbox:latest -- that image exists only on the author's machine.
# The image must be self-contained: PyPI is unreachable at run time, so every dependency the
# GENERATED code might import has to be baked in here.
FROM --platform=linux/amd64 python:3.13-slim

# Required by SUBMISSION_CLI.md:28 and genuinely enforced; must match submission.json.
LABEL qfbench2.interface_version="2.0"

# STAGE 1 (current): the minimum needed to prove the CLI contract end to end.
# STAGE 2 adds the rest of the sandbox stack (statsmodels, scikit-learn, arch, polars, plotly,
# openpyxl, matplotlib, seaborn, numba, TA-Lib from C source). Deferred deliberately: TA-Lib
# compiled under amd64 emulation is slow, and none of it is needed to verify packaging.
RUN pip install --no-cache-dir \
        numpy>=1.26 \
        pandas>=2.1 \
        scipy>=1.12 \
        pyarrow>=15.0

# The agent is stdlib-only; no pip install of our own package is needed, just an importable path.
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
