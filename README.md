# quant_agent

Submission for **Agenthon 2026 — Track 1 (Coding)**: a Docker-image agent that solves
quantitative-finance coding tasks, implementing the track CLI contract

```
solve --task-dir /input --out /app/output
```

Category `api` — the contribution is the harness and prompts; the model is the
organizer-hosted house endpoint.

- [ARCHITECTURE.md](ARCHITECTURE.md) — how the agent works, function by function
- [PLAN.md](PLAN.md) — the competition plan, the traps found so far, and current status

## Repository layout — this repo is a sibling of the competition repos

```
Agenthon/
├── quant_agent/            ← this repo (the submission)
├── track1-coding-public/   ← the practice kit: 87 tasks + their graders   (clone, do not edit)
└── Agenthon2026-public/    ← shared toolkit + starter packs               (clone, do not edit)
```

They are deliberately **outside** this repo. Two reasons: their task data must never enter our
Docker build context (it is QF-Bench v1 material redistributed for non-commercial academic use —
copying it in would relicense it under ours, which is not ours to grant), and nested git repos are
a mess. `$QFBENCH_KIT` overrides the kit location if you keep it elsewhere.

| | |
|---|---|
| `agent/` | **shipped code.** `solve.py` (the CLI contract), `task_context.py` (parses each unit's I/O contract), `llm.py` (OpenAI-compatible client) |
| `tools/` | **dev only, never shipped.** Validators and harnesses |
| `tests/` | **dev only.** Contract tests — the failures that zero a unit without touching finance |
| `scripts/` | **dev only.** Thin wrappers over the build/run/grade commands |
| `docker/` | the submission image |

## Setup

The virtualenv lives one level up at `Agenthon/.venv`, shared with the two competition kits, and
already has both required packages:

```bash
../.venv/bin/pip install -e ../track1-coding-public
../.venv/bin/pip install "qfbench2-common @ git+https://github.com/Agenthon-2026/Agenthon2026-public.git@v2.3.1#subdirectory=common"
cp .env.example .env      # then add your dev API key
```

Both packages are required **together**: `qfbench2-smoke` ships in `qfbench2-common` but imports
`qfbench2_track_coding` from the kit, and neither works alone.

## Everyday commands

```bash
scripts/build.sh                              # build linux/amd64 + verify arch and label
scripts/run-unit.sh t1-zero-coupon-bootstrapping   # one unit, exactly as the harness runs it
scripts/selfgrade.sh t1-zero-coupon-bootstrapping  # that unit's REAL checker -> reward.json
scripts/conformance.sh                        # the organizers' sweep over all 87 units
```

`build.sh` always passes `--platform linux/amd64`. The fleet is x86-64; an arm64 image does not
run, and on Apple silicon a plain `docker build` produces one silently.

## Checks

```bash
../.venv/bin/python -m pytest tests/ -q      # contract tests (93)
python3 tools/check_llm.py                   # dev model reachable?
python3 tools/validate_task_context.py       # parser vs all 87 units' real graders
```

## License

MIT (see [LICENSE](LICENSE)). Covers this repository's code only — **not** the competition task
data, which is never vendored here.
