#!/usr/bin/env bash
# Build the submission image. ALWAYS linux/amd64 -- the fleet is x86-64 and an arm64 image does
# not run, which on Apple silicon is what a plain `docker build` produces.
set -euo pipefail
cd "$(dirname "$0")/.."
TAG="${1:-quant-agent:dev}"
docker buildx build --platform linux/amd64 -f docker/Dockerfile -t "$TAG" --load .
echo
echo -n "architecture check: "
docker image inspect "$TAG" --format '{{.Os}}/{{.Architecture}}'
docker image inspect "$TAG" --format 'label: {{index .Config.Labels "qfbench2.interface_version"}}'
