#!/usr/bin/env bash
# Run ONE unit's real checker against output we already produced.
#   scripts/selfgrade.sh t1-zero-coupon-bootstrapping [outdir]
#
# Self-grading needs MORE mounts than running does: 36 of 87 units' checkers open paths the run
# recipe never provides -- 17 read /tests/reference_data, 13 read /app/data, and a few read bare
# /app/<file>. Without these you get failures that are about mounts, not about your finance.
set -euo pipefail
cd "$(dirname "$0")/.."
UNIT="${1:?usage: selfgrade.sh <unit-id> [outdir]}"
OUT="${2:-/tmp/qa-run/$UNIT}"
KIT="${QFBENCH_KIT:-$(cd .. && pwd)/track1-coding-public}"
UNIT_DIR="$KIT/units/$UNIT"
[ -d "$OUT" ] || { echo "no output at $OUT -- run scripts/run-unit.sh $UNIT first" >&2; exit 2; }

MOUNTS=(-v "$UNIT_DIR":/input:ro -v "$OUT":/output -v "$OUT":/app/output
        -v "$UNIT_DIR/checks":/checks:ro)
[ -d "$UNIT_DIR/environment/data" ]  && MOUNTS+=(-v "$UNIT_DIR/environment/data":/app/data:ro)
[ -d "$UNIT_DIR/checks/reference_data" ] && \
    MOUNTS+=(-v "$UNIT_DIR/checks/reference_data":/tests/reference_data:ro)
# Units whose checker reads a bare /app/<file> rather than /app/data/<file>.
for f in "$UNIT_DIR"/environment/data/*; do
    [ -f "$f" ] && MOUNTS+=(-v "$f":"/app/$(basename "$f")":ro)
done

docker run --rm --network=none "${MOUNTS[@]}" \
  finance-bench-sandbox:latest bash /checks/test.sh
echo "--- reward ---"; cat "$OUT/reward.json" 2>/dev/null || echo "(no reward.json)"
