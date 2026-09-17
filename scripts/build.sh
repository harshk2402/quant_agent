#!/usr/bin/env bash
# Build the agent image.
#
#   scripts/build.sh              submission image, linux/amd64   -> quant-agent:dev
#   scripts/build.sh --arm64      native local-dev image          -> quant-agent:dev-arm64
#   scripts/build.sh [--arm64] <tag>   override the tag
#
# amd64 is the default because it is the only architecture the fleet runs. An arm64 image does not
# run there at all, so arm64 must be requested explicitly and is tagged differently, making it hard
# to submit by accident.
#
# arm64 exists purely for iteration speed on Apple silicon, where emulating amd64 makes anything
# that compiles at run time (numba in particular) extremely slow. Never push it, and never treat a
# result from it as a result about the submission.
set -euo pipefail
cd "$(dirname "$0")/.."

PLATFORM="linux/amd64"
TAG="quant-agent:dev"
if [ "${1:-}" = "--arm64" ]; then
    PLATFORM="linux/arm64"
    TAG="quant-agent:dev-arm64"
    shift
fi
TAG="${1:-$TAG}"

echo "building $TAG for $PLATFORM"
docker buildx build \
    --platform "$PLATFORM" \
    --build-arg TARGET_PLATFORM="$PLATFORM" \
    -f docker/Dockerfile -t "$TAG" --load .

echo
got=$(docker image inspect "$TAG" --format '{{.Os}}/{{.Architecture}}')
echo "architecture: $got"
echo "label:        $(docker image inspect "$TAG" --format '{{index .Config.Labels "qfbench2.interface_version"}}')"
if [ "$got" != "$PLATFORM" ]; then
    echo "ERROR: asked for $PLATFORM but the image is $got" >&2
    exit 1
fi
