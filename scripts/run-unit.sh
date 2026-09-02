#!/usr/bin/env bash
# Run the agent against ONE unit, in the exact shape the harness uses.
#   scripts/run-unit.sh t1-zero-coupon-bootstrapping [image] [outdir]
#
# --network=none is deliberate: stricter than scoring's `restricted`, so anything that passes
# here will not surprise you later. Add -e MODEL_ENDPOINT -e MODEL_NAME and drop the flag only
# when the agent actually needs the model.
#
# Both output paths are mounted to ONE host dir because the units' checkers disagree about which
# they read (bs-greeks-pde hardcodes /output, zero-coupon uses /app/output); the real harness
# binds both, and mounting only one produces a "missing file" you can plainly see.
set -euo pipefail
cd "$(dirname "$0")/.."
UNIT="${1:?usage: run-unit.sh <unit-id> [image] [outdir]}"
IMAGE="${2:-quant-agent:dev}"
OUT="${3:-/tmp/qa-run/$UNIT}"
KIT="${QFBENCH_KIT:-$(cd .. && pwd)/track1-coding-public}"
UNIT_DIR="$KIT/units/$UNIT"
[ -d "$UNIT_DIR" ] || { echo "no such unit: $UNIT_DIR" >&2; exit 2; }

# Resources come from the unit's own card -- timeout_sec alone varies 1200/1800/2400/3600/5400
# across the public set, and held-out units are authored separately.
CPUS=$(grep -E '^cpus' "$UNIT_DIR/card.toml" | tr -dc '0-9' || echo 16)
MEM=$(grep -E '^memory' "$UNIT_DIR/card.toml" | sed -E 's/.*"([^"]+)".*/\1/' || echo 128G)

rm -rf "$OUT"; mkdir -p "$OUT"
docker run --rm --network=none \
  --cpus="${CPUS:-16}" --memory="${MEM:-128G}" \
  -v "$UNIT_DIR":/input:ro \
  -v "$OUT":/output -v "$OUT":/app/output \
  "$IMAGE" solve --task-dir /input --out /app/output
echo "exit=$?  ->  $OUT"; ls -la "$OUT"
