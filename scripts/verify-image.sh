#!/usr/bin/env bash
# Check that a built image is actually submittable, before spending model calls on it.
#
#   scripts/verify-image.sh                       the submission image (quant-agent:dev)
#   scripts/verify-image.sh quant-agent:dev-arm64 the native local-dev image
#
# Everything here is free: no network, no API key, no model calls. It checks the properties that
# fail a submission silently and completely, plus the one that motivated the Stage 2 rebuild —
# a generated script importing a library the image does not carry dies with an ImportError that
# looks, from a score, exactly like bad finance.
#
# Only an amd64 image can PASS as submittable. An arm64 image is checked on everything else and
# reported as a dev image, because passing here on arm64 says nothing about whether the amd64
# artifact works: wheel availability and floating-point output both differ by architecture.
set -uo pipefail
cd "$(dirname "$0")/.."
IMAGE="${1:-quant-agent:dev}"
FAIL=0
ok()   { printf '  ok    %s\n' "$1"; }
bad()  { printf '  FAIL  %s\n' "$1"; FAIL=1; }

echo "verifying $IMAGE"

# --- 1. architecture -------------------------------------------------------------------------
# The fleet is x86-64. An arm64 image does not run at all, on any task, and the error is not
# shown to us — so this is the single most expensive thing to get wrong.
arch=$(docker image inspect "$IMAGE" --format '{{.Os}}/{{.Architecture}}' 2>/dev/null)
DEV_IMAGE=0
if [ "$arch" = "linux/amd64" ]; then
    ok "architecture $arch"
elif [ "$arch" = "linux/arm64" ]; then
    printf '  dev   architecture %s — local development image, NOT submittable\n' "$arch"
    DEV_IMAGE=1
else
    bad "architecture is '$arch', need linux/amd64"
fi

# --- 2. interface label ----------------------------------------------------------------------
# Required by SUBMISSION_CLI.md and genuinely enforced; must match submission.json.
label=$(docker image inspect "$IMAGE" --format '{{index .Config.Labels "qfbench2.interface_version"}}' 2>/dev/null)
[ "$label" = "2.0" ] && ok "label qfbench2.interface_version=$label" || bad "label is '$label', need 2.0"

# --- 3. the verb resolves and the run exits 0 --------------------------------------------------
# The harness passes `solve` as the first argument after the image reference. An image that does
# not accept it exits 127 or 126 on every task, recorded as our failure rather than an organizer
# fault. Run against a real unit, offline, so the model call fails and the placeholder path is
# exercised — this checks the contract, not the agent.
unit="$(pwd)/../track1-coding-public/units/t1-zero-coupon-bootstrapping"
out=$(mktemp -d)
if [ -d "$unit" ]; then
    docker run --rm --network=none \
        -v "$unit":/input:ro -v "$out":/output -v "$out":/app/output \
        "$IMAGE" solve --task-dir /input --out /app/output >/dev/null 2>&1
    rc=$?
    [ "$rc" -eq 0 ] && ok "verb resolves, exit 0" || bad "exit $rc (127=verb not on PATH, 126=not executable)"
    [ -f "$out/results.json" ] && ok "wrote the expected deliverable" || bad "no results.json written"
else
    echo "  skip  verb check (practice kit not found beside repo)"
fi
rm -rf "$out"

# --- 4. the sandbox stack imports ---------------------------------------------------------------
# The reason Stage 2 exists. During the first sweep a generated script did `import statsmodels`
# and died, because the image carried four packages while the local environment carried the full
# stack — meaning the numbers we had measured were not measurements of the artifact.
echo "  --- library imports inside the container ---"
# --entrypoint is essential. The image's ENTRYPOINT is `python -m agent.solve`, so without the
# override the trailing `python -c ...` is handed to the AGENT as arguments: it rejects them as an
# unknown verb, falls back to defaults, solves nothing, and exits 0 by design. The check then
# reports success having imported nothing — which is exactly what an earlier version of this
# script did.
ALL_MODS="numpy pandas scipy pyarrow statsmodels sklearn arch polars matplotlib seaborn plotly openpyxl numba backtrader yfinance talib"

# Import verification does not run under emulation (an amd64 image on an Apple-silicon host).
#
# It was tried, and it does not work. Several libraries do heavy native work at import time —
# arch pulls in numba, which JIT-compiles machine code; polars loads a large Rust binary and probes
# CPU features — and under binary translation those imports do not finish in useful time. `import
# arch` ran for over 52 minutes; with numba and arch skipped, `import polars` then ran for another
# 52. Skipping libraries one at a time only moves the hang to the next one, and what would survive
# is precisely the heavy numerical stack the tasks depend on.
#
# So the imports are not attempted, and the verdict is PARTIAL rather than a pass. The same image
# imports all sixteen natively; that is what .github/workflows/verify-image.yml checks, on an
# x86-64 runner with no emulation involved. The arm64 dev image also imports all sixteen locally,
# in about four seconds.
HOST_ARCH=$(uname -m)
EMULATED=0
case "$HOST_ARCH:$arch" in
    arm64:linux/amd64|aarch64:linux/amd64) EMULATED=1 ;;
esac
SKIPPED=""

if [ "$EMULATED" -eq 1 ]; then
    SKIPPED="$ALL_MODS"
    echo "  skip  all imports — amd64 image under emulation on $HOST_ARCH; heavy native imports do"
    echo "        not complete under binary translation. Verified instead by CI on native x86-64."
else
    EXPECTED=$(echo $ALL_MODS | wc -w | tr -d ' ')
    imports=$(docker run --rm --network=none --entrypoint python "$IMAGE" -c "
import importlib, sys
mods = '$ALL_MODS'.split()
bad = []
for m in mods:
    try:
        importlib.import_module(m)
        print(f'ok    import {m}')
    except Exception as e:
        print(f'FAIL  import {m}: {type(e).__name__}: {e}')
        bad.append(m)
import numpy, pandas
print(f'info  numpy {numpy.__version__}, pandas {pandas.__version__}')
print(f'IMPORTED {len(mods) - len(bad)}/{len(mods)}')
sys.exit(1 if bad else 0)
" 2>&1)
    rc=$?
    echo "$imports" | grep -vE '^(WARNING: The requested image|IMPORTED )' | sed 's/^/  /'

    # Guard against a check that silently tests nothing. Exit status alone cannot be trusted: if the
    # command never reaches the import script, whatever runs instead may still exit 0. Require the
    # script's own completion marker, and require that it saw every module.
    if ! grep -q "^IMPORTED $EXPECTED/$EXPECTED\$" <<<"$imports"; then
        bad "import check did not complete, or not every module imported (rc=$rc)"
    elif [ "$rc" -ne 0 ]; then
        bad "import check exited $rc"
    fi
fi

echo
if [ "$FAIL" -ne 0 ]; then
    echo "FAIL — see above"
elif [ "$DEV_IMAGE" -eq 1 ]; then
    echo "PASS (dev image) — all checks pass on arm64. This says nothing about the amd64 submission"
    echo "image; verify quant-agent:dev before relying on it."
elif [ -n "$SKIPPED" ]; then
    # Not a pass. The contract checks passed, but no library was imported, and those libraries are
    # the whole point of the image. Calling this submittable would repeat the mistake of reporting
    # success for a check that did not run.
    echo "PARTIAL — architecture, label and CLI contract verified; library imports NOT verified"
    echo "under emulation. Rely on the verify-image CI workflow (native x86-64) for those."
else
    echo "PASS — image is submittable"
fi
exit "$FAIL"
