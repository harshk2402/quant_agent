# SPDX-License-Identifier: MIT
"""Build and validate `submission.json`, the descriptor that identifies our agent image.

The organizers call the descriptor "the most common way a submission fails before it ever runs":
twelve required fields, a closed vocabulary that rejects unknown keys, and a self-referential digest
with a specific canonicalisation. None of that is caught by eyeballing the file, so this script
builds it from the published fixture, seals the digest with the toolkit's own function, and then
validates it through the toolkit's parser.

Validating with `jsonschema` alone is NOT sufficient: the toolkit ships an invalid fixture
(`wrong_descriptor_digest.json`) that passes schema validation and is caught only by the parser.
This script therefore runs `SubmissionDescriptor.from_mapping`, which is what the organizers use.

Three traps this avoids by construction:

  * `house_endpoint_only` is recommended by the track kit's own SUBMISSION_CLI.md and is a HARD
    validation error — the vocabulary is closed and that key is not in it. We never write it.
  * `image` is an object of registry/repository/digest. The pre-freeze spellings `image_digest` and
    `model_disclosure` ship as NEGATIVE fixtures; writing them fails twice over.
  * `descriptor_digest` is the sha256 of the descriptor WITH that field removed, canonicalised
    under RFC 8785 (JCS) — not `json.dumps(sort_keys=True)`. `seal_descriptor_digest` does it.

Usage:
    python3 tools/make_submission.py \
        --team-id <from the competition platform> \
        --image ghcr.io/<owner>/quant-agent \
        --digest sha256:<64 hex>            # or --from-pushed-image <tag> to read it from Docker

The archive is one file, `submission.json`, at the root and nothing else; `--zip` writes it.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import zipfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from qfbench2_common.contracts.descriptor import (  # noqa: E402
    SubmissionDescriptor,
    seal_descriptor_digest,
)

# --- values that are ours to choose ------------------------------------------------------------

#: Our own code's licence, NOT the task data's. The tasks are CC BY-NC and are never vendored here.
LICENSE = "MIT"

#: `api` means the contribution is the harness and prompts, and the model is the organizer-hosted
#: house endpoint. We bundle no weights and no adapter.
CATEGORY = "api"

# --- the model disclosure, and what is honestly unknown ----------------------------------------
#
# The rules require every model used to be declared with a pinned version and a training cutoff,
# and reject floating aliases at verification. For an `api` submission calling the house endpoint,
# two of those five fields are not knowable from anything published:
#
#   * `version` — SUBMISSION_CLI.md says only that the pinned id "is published with the model pin".
#     The id is supplied to the container at run time as $MODEL_NAME; we never choose it.
#   * `training_cutoff` — not published anywhere for the house model.
#
# The honest response is to say so in the field rather than invent a plausible-looking pin, and to
# raise it with the organizers. A wrong-but-plausible value would be a false disclosure; a visibly
# unknown one is an accurate statement of what we were given.
#
# The NAME is known: the Track 1 starter pack names the model behind $MODEL_ENDPOINT while
# describing the BYO path.
HOUSE_MODEL_NAME = "nvidia/nemotron-3-super-120b-a12b"
HOUSE_MODEL_VERSION = "pinned-by-organizer-supplied-as-MODEL_NAME-at-runtime"
HOUSE_MODEL_CUTOFF = "not-published-by-organizer"


def digest_of_pushed_image(tag: str) -> str:
    """Read the registry digest Docker recorded when the image was pushed.

    This is the digest the descriptor must name, and it exists only after a push: it identifies the
    manifest in the registry, not the local image id. Reading it back beats transcribing 64 hex
    characters by hand.
    """
    out = subprocess.run(
        ["docker", "image", "inspect", tag, "--format", "{{join .RepoDigests \"\\n\"}}"],
        capture_output=True, text=True,
    )
    digests = [d.strip() for d in out.stdout.splitlines() if "@sha256:" in d]
    if not digests:
        raise SystemExit(
            f"no registry digest recorded for {tag!r}. The digest only exists once the image has "
            f"been pushed — run `docker push {tag}` first."
        )
    if len({d.split("@", 1)[1] for d in digests}) > 1:
        raise SystemExit(f"{tag!r} has several differing digests: {digests}")
    return digests[0].split("@", 1)[1]


def build(team_id: str, registry: str, repository: str, digest: str, phase: str) -> dict:
    body = {
        "schema_version": "1.0.0",
        "interface_version": "2.0",          # must match the image's qfbench2.interface_version
        "competition_id": f"agenthon2026-coding-{phase}",
        "team_id": team_id,
        "track": "coding",
        "phase": phase,
        "category": CATEGORY,
        "image": {
            "registry": registry,
            "repository": repository,
            "digest": digest,
        },
        "image_access": "public",
        "models": [
            {
                "name": HOUSE_MODEL_NAME,
                "version": HOUSE_MODEL_VERSION,
                "training_cutoff": HOUSE_MODEL_CUTOFF,
                "access": "api",
                "revision": HOUSE_MODEL_VERSION,
            }
        ],
        "license": LICENSE,
    }
    # Seal last: the digest covers the descriptor with the digest field removed, so every other
    # value must be final before this runs.
    return seal_descriptor_digest(body)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team-id", required=True,
                    help="your team id, from the competition platform — nothing can derive it")
    ap.add_argument("--image", required=True,
                    help="registry/repository, e.g. ghcr.io/owner/quant-agent")
    ap.add_argument("--digest", default="",
                    help="sha256:<64 hex> of the PUSHED image")
    ap.add_argument("--from-pushed-image", default="",
                    help="read the digest from a local tag that has been pushed")
    ap.add_argument("--phase", default="final", choices=["dev", "final", "verification"])
    ap.add_argument("--out", default="submission.json")
    ap.add_argument("--zip", default="", help="also write a zip containing only submission.json")
    args = ap.parse_args()

    if "/" not in args.image:
        raise SystemExit("--image must be registry/repository, e.g. ghcr.io/owner/quant-agent")
    registry, repository = args.image.split("/", 1)

    digest = args.digest
    if args.from_pushed_image:
        digest = digest_of_pushed_image(args.from_pushed_image)
        print(f"digest read from {args.from_pushed_image}: {digest}")
    if not digest:
        raise SystemExit("one of --digest or --from-pushed-image is required")

    descriptor = build(args.team_id, registry, repository, digest, args.phase)

    # The real check. `from_mapping` raises on anything the organizers' parser would reject, and it
    # catches at least one error class that passes JSON-Schema validation.
    try:
        parsed = SubmissionDescriptor.from_mapping(descriptor)
    except Exception as e:
        print(json.dumps(descriptor, indent=2))
        raise SystemExit(f"\nREJECTED by the toolkit's parser: {type(e).__name__}: {e}")

    out = pathlib.Path(args.out)
    out.write_text(json.dumps(descriptor, indent=2) + "\n")
    print(json.dumps(descriptor, indent=2))
    print(f"\nvalidated by SubmissionDescriptor.from_mapping -> {out}")
    print(f"image reference: {parsed.image_reference()}")

    if args.zip:
        # One file at the archive root, nothing else.
        with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(out, arcname="submission.json")
        print(f"wrote {args.zip} (contains only submission.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
