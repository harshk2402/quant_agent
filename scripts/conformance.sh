#!/usr/bin/env bash
# The organizers' own sweep over all 87 public units. Checks the two failures that score zero
# without ever being about the finance: did it exit 0, and did it write a filename the unit's
# checker expects. A FLOOR check -- it says nothing about whether the numbers are right.
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE="${1:-quant-agent:dev}"
KIT="${QFBENCH_KIT:-$(cd .. && pwd)/track1-coding-public}"
PACK="${QFBENCH_PACK:-$(cd .. && pwd)/Agenthon2026-public}/starter-packs/track1/conformance.sh"
[ -f "$PACK" ] || { echo "conformance.sh not found at $PACK" >&2; exit 2; }
exec bash "$PACK" "$IMAGE" "$KIT"
