#!/usr/bin/env bash
# Push the submission image to a registry and prove it can be pulled with no credentials.
#
#   scripts/push-image.sh [repo] [tag]          default: ghcr.io/harshk2402/quant-agent submission
#
# Two things make this worth a script rather than two docker commands.
#
# 1. A PRIVATE image fails in the worst possible way. Ingestion pulls the image from inside the
#    program container, which holds no registry credential. If the pull fails the run still reports
#    "Finished" and lands on the same score floor as a submission that ran and was wrong — and it
#    fails INCONSISTENTLY, because a host that happens to have the image cached runs it fine. On
#    GHCR a freshly pushed package is private by default, and making it public is a UI-only action.
#
# 2. `docker pull`, `docker manifest inspect` and `buildx imagetools inspect` all use YOUR
#    credential, so they succeed on a private package and tell you nothing. The only meaningful
#    check is an anonymous one, which is what this script ends with.
#
# It also refuses to push a non-amd64 image: the local arm64 development image would be useless on
# the fleet and is easy to tag by mistake.
set -uo pipefail
cd "$(dirname "$0")/.."

REPO="${1:-ghcr.io/harshk2402/quant-agent}"
TAG="${2:-submission}"
LOCAL="${LOCAL_IMAGE:-quant-agent:dev}"

echo "local image : $LOCAL"
echo "target      : $REPO:$TAG"
echo

# --- refuse to push the wrong architecture -----------------------------------------------------
arch=$(docker image inspect "$LOCAL" --format '{{.Os}}/{{.Architecture}}' 2>/dev/null)
if [ -z "$arch" ]; then
    # Distinguish "wrong architecture" from "could not look at all" — an empty result means the
    # daemon is down or the tag does not exist, and blaming the architecture would mislead.
    echo "cannot inspect $LOCAL — is Docker running, and does that image exist?" >&2
    exit 1
fi
if [ "$arch" != "linux/amd64" ]; then
    echo "REFUSING to push: $LOCAL is '$arch', and the fleet runs linux/amd64." >&2
    echo "Build it with scripts/build.sh (no --arm64)." >&2
    exit 1
fi
echo "architecture: $arch  (ok)"

label=$(docker image inspect "$LOCAL" --format '{{index .Config.Labels "qfbench2.interface_version"}}' 2>/dev/null)
if [ "$label" != "2.0" ]; then
    echo "REFUSING to push: qfbench2.interface_version label is '$label', need 2.0." >&2
    exit 1
fi
echo "label       : $label  (ok)"

# --- push --------------------------------------------------------------------------------------
# Requires `docker login ghcr.io` first, with a token carrying write:packages.
echo
docker tag "$LOCAL" "$REPO:$TAG" || exit 1
if ! docker push "$REPO:$TAG"; then
    echo >&2
    echo "push failed. If this is an authentication error, log in first:" >&2
    echo "  echo \$GITHUB_TOKEN | docker login ghcr.io -u <github-username> --password-stdin" >&2
    echo "using a token with the write:packages scope." >&2
    exit 1
fi

# --- the digest the descriptor must name --------------------------------------------------------
# This identifies the manifest in the REGISTRY, not the local image id, and only exists post-push.
DIGEST=$(docker image inspect "$REPO:$TAG" --format '{{join .RepoDigests "\n"}}' 2>/dev/null \
         | grep -F "$REPO@" | head -1 | cut -d@ -f2)
if [ -z "$DIGEST" ]; then
    echo "pushed, but no registry digest was recorded — cannot continue" >&2
    exit 1
fi
echo
echo "digest      : $DIGEST"

# --- the only check that means anything ---------------------------------------------------------
# A token requested with no credentials. If this returns 200, an unauthenticated client — which is
# what ingestion is — can fetch the manifest.
PATH_ONLY="${REPO#*/}"          # strip the registry host; the API wants owner/name
echo
echo "anonymous pullability check:"
TOKEN=$(curl -s "https://ghcr.io/token?scope=repository:${PATH_ONLY}:pull&service=ghcr.io" \
        | python3 -c 'import json,sys; print(json.load(sys.stdin).get("token",""))' 2>/dev/null)
if [ -z "$TOKEN" ]; then
    echo "  could not obtain an anonymous token — check the repository path" >&2
    exit 1
fi
CODE=$(curl -s -o /dev/null -w '%{http_code}' \
    -H "Authorization: Bearer $TOKEN" \
    -H 'Accept: application/vnd.oci.image.index.v1+json' \
    -H 'Accept: application/vnd.docker.distribution.manifest.v2+json' \
    "https://ghcr.io/v2/${PATH_ONLY}/manifests/${DIGEST}")

echo "  HTTP $CODE"
if [ "$CODE" = "200" ]; then
    echo "  PASS — an unauthenticated client can fetch this manifest."
    echo
    echo "descriptor digest to use:"
    echo "  python3 tools/make_submission.py --team-id <id> --image $REPO --digest $DIGEST"
    exit 0
fi

echo "  FAIL — the image is NOT anonymously pullable (a fresh GHCR package is private by default)."
echo "  Make it public: github.com/users/<owner>/packages -> this package -> Package settings ->"
echo "  Change visibility -> Public. There is no API for it. Then re-run this script, or just the"
echo "  check, to confirm."
exit 1
