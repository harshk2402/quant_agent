# SPDX-License-Identifier: MIT
"""Build the submission zip: the sealed descriptor plus the team claim that proves who we are.

    python3 tools/make_submission.py --phase dev \
        --image ghcr.io/harshk2402/quant-agent \
        --digest sha256:<64 hex>

The Team Key is read from a hidden prompt. It is never an argument, never printed, and never
written to any file: shell history and process listings are both readable by things we do not
control. Nothing secret ends up in the zip — the claim carries a proof computed under the key, not
the key itself — but the key passes through this process, so the archive is created mode 0600 and
with O_EXCL so a half-written one is never left world-readable.

WHAT THE ARCHIVE CONTAINS — two files at the root, nothing else:

  submission.json   the descriptor, naming the image by digest
  team-claim.json   {schema_version, site_team_id, descriptor_sha256, proof}

WHY THE DETAILS MATTER. Four things here are not stylistic:

  * `team_id` is DERIVED, not assigned. It is a hash of the team number and the Team Key. Nobody
    issues it, and a different key gives a different id.
  * The claim's proof is bound to the sha256 of the EXACT submission.json bytes in the same zip.
    So the descriptor must be serialised precisely as the published recipe does it —
    `json.dumps(sealed, indent=2, sort_keys=True) + "\\n"` — and the claim must hash those bytes,
    not a re-serialisation of the parsed object. Getting this wrong means the claim is refused.
  * `descriptor_digest` covers every other field, so it is sealed last, after `team_id` is in place.
  * Claim schema 1.0, which carried the key in clear text, is REFUSED. This writes 2.0.

A held upload — no claim, a bad proof, or a claim that disagrees with an account already linked —
is never run and never scored, and it still spends one of the submission attempts. So the cost of
getting this wrong is an attempt, not a bad score.
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import hmac
import json
import os
import pathlib
import subprocess
import sys
import zipfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from qfbench2_common.contracts.descriptor import (  # noqa: E402
    SubmissionDescriptor,
    seal_descriptor_digest,
)

#: Our own code's licence, NOT the task data's. The tasks are CC BY-NC and are never vendored here.
LICENSE = "MIT"

#: Every Track 1 entry is `api`: the model is the organizer-hosted House endpoint and bundling a
#: model or adapter is not part of this competition. Any other value is held and never run.
CATEGORY = "api"

# --- the model disclosure ----------------------------------------------------------------------
# Each entry needs a name, a dated version, a training cutoff, an access and a revision. For an
# `api` submission two of those are not knowable from anything published:
#
#   * `version` — the pinned id is handed to the container at run time as $MODEL_NAME. We never
#     choose it and cannot read it from here.
#   * `training_cutoff` — not published for the House model.
#
# Saying so plainly is more honest than inventing a plausible-looking pin, which would be a false
# disclosure. The NAME is known: the Track 1 starter pack names the model behind $MODEL_ENDPOINT.
HOUSE_MODEL_NAME = "nvidia/nemotron-3-super-120b-a12b"
HOUSE_MODEL_VERSION = "pinned-by-organizer-supplied-as-MODEL_NAME-at-runtime"
HOUSE_MODEL_CUTOFF = "not-published-by-organizer"


def digest_of_pushed_image(tag: str) -> str:
    """Read the registry digest Docker recorded for a pushed image."""
    out = subprocess.run(
        ["docker", "image", "inspect", tag, "--format", '{{join .RepoDigests "\\n"}}'],
        capture_output=True, text=True,
    )
    digests = {d.split("@", 1)[1] for d in out.stdout.splitlines() if "@sha256:" in d}
    if not digests:
        raise SystemExit(f"no registry digest for {tag!r} — push it first")
    if len(digests) > 1:
        raise SystemExit(f"{tag!r} has several digests: {sorted(digests)}")
    return digests.pop()


def derive_team_id(team_number: int, team_key: str) -> str:
    """team_id = "team-" + sha256("agenthon2026-team-alias:<number>:<key>")[:32].

    The key is hashed exactly as issued — no trimming, no case changes — because any alteration
    produces a different id and an id that does not match the roster.
    """
    return "team-" + hashlib.sha256(
        b"agenthon2026-team-alias:" + str(team_number).encode() + b":" + team_key.encode()
    ).hexdigest()[:32]


def build_descriptor(team_id: str, registry: str, repository: str, digest: str,
                     phase: str) -> dict:
    body = {
        "schema_version": "1.0.0",
        "interface_version": "2.0",      # must match the image's qfbench2.interface_version label
        "competition_id": f"agenthon2026-coding-{phase}",
        "team_id": team_id,
        "track": "coding",
        "phase": phase,
        "category": CATEGORY,
        "image": {"registry": registry, "repository": repository, "digest": digest},
        "image_access": "public",        # the only route; "organizer_mirror" is not available
        "models": [{
            "name": HOUSE_MODEL_NAME,
            "version": HOUSE_MODEL_VERSION,
            "training_cutoff": HOUSE_MODEL_CUTOFF,
            "access": "api",
            "revision": HOUSE_MODEL_VERSION,
        }],
        "license": LICENSE,
    }
    # Sealed last: the digest covers every other field, so they must all be final first.
    return seal_descriptor_digest(body)


def descriptor_bytes_of(sealed: dict) -> bytes:
    """The exact bytes the archive will carry, and therefore the bytes the claim commits to.

    This serialisation is fixed by the published recipe. It must not be "tidied" — sort_keys, the
    two-space indent and the trailing newline are all part of what both ends hash.
    """
    return (json.dumps(sealed, indent=2, sort_keys=True) + "\n").encode("utf-8")


def build_claim(team_number: int, team_key: str, descriptor_sha256: str) -> dict:
    proof_key = hashlib.sha256(
        b"agenthon2026-team-proof-key:v2:" + team_key.encode()).digest()
    message = (b"agenthon2026-team-claim:v2:" + str(team_number).encode()
               + b":" + descriptor_sha256.encode())
    proof = hmac.new(proof_key, message, hashlib.sha256).hexdigest()
    return {
        "schema_version": "2.0",         # 1.0 carried the key in clear text and is refused
        "site_team_id": team_number,
        "descriptor_sha256": descriptor_sha256,
        "proof": proof,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team-number", type=int, required=True,
                    help="your agenthon.net team number (not the key)")
    ap.add_argument("--image", required=True,
                    help="registry/repository, e.g. ghcr.io/harshk2402/quant-agent")
    ap.add_argument("--digest", default="", help="sha256:<64 hex> of the PUSHED image")
    ap.add_argument("--from-pushed-image", default="",
                    help="read the digest from a local tag that has been pushed")
    ap.add_argument("--phase", default="dev", choices=["dev", "final"],
                    help="must match the phase you upload into; a mismatch is refused "
                         "before the image is pulled. Final & Verification is ONE phase: "
                         "competition_id agenthon2026-coding-final, phase final")
    ap.add_argument("--out", default="submission.zip")
    args = ap.parse_args()

    if "/" not in args.image:
        raise SystemExit("--image must be registry/repository")
    registry, repository = args.image.split("/", 1)

    digest = args.digest or (
        digest_of_pushed_image(args.from_pushed_image) if args.from_pushed_image else "")
    if not digest:
        raise SystemExit("one of --digest or --from-pushed-image is required")

    # Hidden prompt. Never an argument: shell history and `ps` are both readable by other things.
    team_key = getpass.getpass("Team Key (input hidden): ")
    if not team_key:
        raise SystemExit("no Team Key entered")

    team_id = derive_team_id(args.team_number, team_key)
    sealed = build_descriptor(team_id, registry, repository, digest, args.phase)

    # Validate with the toolkit's own parser, not bare jsonschema: the organizers ship an invalid
    # fixture that passes schema validation and is caught only here.
    try:
        parsed = SubmissionDescriptor.from_mapping(sealed)
    except Exception as e:
        raise SystemExit(f"descriptor REJECTED by the toolkit's parser: {type(e).__name__}: {e}")

    body = descriptor_bytes_of(sealed)
    descriptor_sha256 = hashlib.sha256(body).hexdigest()
    claim = build_claim(args.team_number, team_key, descriptor_sha256)
    del team_key                          # out of scope as early as possible

    out = pathlib.Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} already exists — remove it rather than overwriting a submission")
    # O_EXCL and 0600: a half-written archive is never left world-readable on a shared machine.
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with zipfile.ZipFile(os.fdopen(fd, "wb"), "w") as archive:
        archive.writestr("submission.json", body)
        archive.writestr("team-claim.json",
                         json.dumps(claim, separators=(",", ":"), sort_keys=True))

    print(json.dumps(sealed, indent=2, sort_keys=True))
    print()
    print(f"team_id           : {team_id}   (derived, not assigned)")
    print(f"image             : {parsed.image_reference()}")
    print(f"descriptor_sha256 : {descriptor_sha256}")
    print(f"claim             : schema 2.0, site_team_id={args.team_number}, proof computed")
    print(f"competition_id    : {sealed['competition_id']}  (phase={args.phase})")
    print()
    print(f"wrote {out} — upload this as it is. Nothing in it is secret.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
