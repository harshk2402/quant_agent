# Architecture

How the agent works, function by function. For the competition plan, phases and open questions,
see [PLAN.md](PLAN.md).

## The one-paragraph version

The harness starts our container with `solve --task-dir /input --out /app/output`. We read the task
folder to work out **what files to write** and **where the data actually is**, ask a language model
to write a Python script, run that script in a scratch directory, and let it produce the
deliverables. Anything it fails to produce is backfilled with empty-but-valid files. We always exit
with status 0.

## The call chain

```
main()                              entry point, parses the CLI
 └─ solve()                         orchestrates one task
     ├─ resolve_output_dir()        where do deliverables go?
     ├─ parse_task()                what to write, where to read
     ├─ _generate_and_run()         the language-model half
     │   ├─ stage_inputs()          put data where the task text claims it is
     │   ├─ build_prompt()          assemble the request
     │   ├─ chat()                  call the model
     │   ├─ extract_code()          pull Python out of the reply
     │   └─ run_code()              execute it in a scratch directory
     └─ _write_placeholder()        backfill anything missing
```

---

## 1. Entry point — `agent/solve.py`

### `main(argv)`

Parses the command line. The significant detail is that **`solve` arrives as a positional
argument**, so it is declared as one. If it were rejected as unexpected, the container would exit
non-zero on every task, and the competition records that as our failure rather than an
infrastructure fault.

It uses `parse_known_args` rather than `parse_args`: an unrecognised flag is logged and ignored
instead of ending the run, and the `--task-dir` and `--out` values we did understand are still
honoured.

### `solve(task_dir, out)`

The orchestrator. It always returns 0, because a single uncaught exception forfeits all four
admissibility checks at once — a wrong answer is still scored, whereas an exception is not.

The order matters:

1. Resolve the output directory and create it.
2. Parse the task. If this throws, log and exit 0 — nothing is known about the contract, so there
   is nothing safe to write.
3. Run generation, wrapped so that failure is survivable.
4. Backfill any deliverable that does not exist yet.
5. Report how many files ended up non-empty.

Step 4 only fills gaps, so a real answer is never overwritten by an empty placeholder.

### `_generate_and_run(ctx, out_dir)`

The language-model half, separated so `solve()` stays readable. It is single-shot: stage, prompt,
call, extract, execute. There is deliberately **no retry** — that is Step 0.7 — which means this
reproduces FinanceZero, the baseline the submission has to beat, and gives us the number every
later change is measured against.

Failures are printed to stderr only, never into the output directory: a traceback quotes source
lines, and generated source may contain text copied from the task.

### `_write_placeholder(spec, ctx, out_dir)`

Writes a structurally valid but empty deliverable — a CSV with only a header row, `{}` for JSON, an
empty DataFrame for Parquet, a 1×1 transparent PNG.

Its content is derived **only from the parsed contract** (filenames and column names), never from
task or input text. That is what keeps canary markers out of the output.

---

## 2. Understanding the task — `agent/task_context.py`

The largest module, and the one that closes the most expensive failure mode: writing the answer
under the wrong filename.

### `parse_task(task_dir, probe_filesystem, use_checks)`

The main entry point. Produces a `TaskContext` describing what to write, where to read, and the
task's resource limits. It draws on:

| Function | Purpose |
|---|---|
| `_read_card()` | Reads `card.toml` — title, category, and the timeout, which varies (1200/1800/2400/3600/5400) |
| `_dockerfile_dests()` | Reads the task's Dockerfile `COPY` lines to learn where data lands at build time |
| `_manifest_inputs()` | Reads `manifest.json` for declared input files |
| `_gather_output_candidates()` | Extracts deliverable filenames from the prose |
| `_outputs_from_checks()` | Extracts them from the grading code, when it is available |

**Input resolution** builds a list of candidate locations for each file — the path the prose claims,
the Dockerfile destination, `/input/environment/data/`, `/app/data/` — then probes the filesystem
for the first that exists, falling back to a recursive search. Results are stored as **absolute**
paths, because generated code runs from a different working directory and a relative path would
not resolve there.

`use_checks=False` disables reading the grading code. The offline validator requires this: it
grades the prose parse *against* the grading code, so allowing the parser to read the same source
would make the comparison circular.

### `_gather_output_candidates(md, input_basenames)`

Finds deliverable filenames in English prose. Section headings vary considerably across the 87
practice tasks (`## Output`, `## Deliverables`, `## Step 11: Output Files`, and others), so heading
matching alone is unreliable and three strategies are layered:

1. Any explicit `/app/output/X` path anywhere in the document — the strongest and least ambiguous
   signal.
2. Filenames appearing in sections whose heading indicates output.
3. Bare backticked filenames, as a last resort.

Known input filenames are subtracted, so a sentence such as "read `prices.csv` and write
`results.json`" cannot mistake the input for a deliverable.

### `_outputs_from_checks(task_dir)`

Reads the task's own grading code when it is present in the mount. This is exact where prose is
inferred: on `t1-polars-api-migration` the prose names one deliverable and the grading code names
three more.

A line is ignored if it reaches through a data, reference, or log root, because **position rather
than spelling** distinguishes an input from a deliverable — inputs and reference files already
exist before the agent runs, so counting them would credit an agent that wrote nothing itself.

Returns an empty list when `checks/` is absent, in which case the prose parse is used alone.

### Supporting functions

- `_split_sections()` — splits markdown into sections, skipping fenced code blocks so that a `#`
  comment inside a Python example is not mistaken for a heading.
- `_columns_from()` — extracts column names from markdown tables, backticked comma lists, or a bare
  CSV header inside an indented code block.
- `resolve_output_dir()` — mirrors the graders' own logic: `$OUTPUT_DIR` if set, otherwise
  `/app/output` if it exists, otherwise `/output`.
- `render_output_contract()` — formats the contract for inclusion in the prompt.

---

## 3. Asking the model — `agent/prompt.py`

### `strip_canary(text)`

Removes canary marker lines, then redacts any remaining GUID-shaped token.

A canary is a unique random string the organizers plant in task text so they can detect benchmark
material leaking where it should not be. 66 of the 87 practice tasks carry one, and one of the four
admissibility checks fails any submission whose output contains it. A model that is never shown the
marker cannot reproduce it in a comment, which is the realistic way it would leak.

### `build_prompt(ctx, out_dir)`

Assembles the messages in a deliberate order:

1. The task title.
2. **Input files at their resolved paths**, noting explicitly where the task text disagrees.
3. **The output contract** — the exact filenames. A file written under the wrong name scores zero
   however correct its contents, so this is the most important thing the model is told.
4. Per-file schema descriptions.
5. The task description itself, canary-stripped, flagged as authoritative for the finance but not
   for the paths.

The system prompt fixes the rules: one code block only, offline, exact filenames, never write
`reward.json`, never copy task text into an output file, seed randomness from `QFBENCH_SEED`, and
handle edge cases rather than crashing.

### `extract_code(reply)`

Pulls the largest fenced Python block out of the reply, falling back to the raw text when it
appears to be a script. The fallback is deliberate: failing to extract forfeits the task, whereas
an incorrect extraction costs only one execution attempt.

---

## 4. Running the code — `agent/execute.py`

### `stage_inputs(ctx)`

Copies each resolved input to the path the task text claims it is at.

41 of the 87 tasks describe their data as living at `/app/data/...`, which is true when the
organizers build their own image for a task but not at evaluation time, when our image runs and the
task folder is mounted at `/input`. The prompt does state the real path, but a model reading a task
description will often use the path in front of it. Staging makes both routes work rather than
depending on the model to ignore its own instructions.

Copies rather than symlinks — generated code may open the file with any library — never overwrites
an existing file, and treats an unwritable destination as non-fatal.

### `exec_timeout_for(ctx)`

The execution budget, taken from the task's own card: half the total, capped at 900 seconds. The
remainder is left for generation, staging, and the repair loop once it exists.

### `run_code(code, out_dir, timeout, scratch)`

Writes the script to a **scratch directory, never the output directory**, and runs it as a
subprocess with its working directory set there, so that a stray relative write lands in scratch
rather than among the deliverables. `OUTPUT_DIR` and `QFBENCH_SEED` are passed through to the child.

Returns an `ExecResult` capturing stdout, stderr, the return code and whether it timed out.
`ExecResult.failure_text()` returns the **tail** of that output, because a traceback's final lines
carry the exception; this is what the repair loop will consume.

---

## 5. Talking to a model — `agent/llm.py`

### `resolve_config()`

Selects the backend from environment variables. `MODEL_ENDPOINT` always takes precedence:

| Mode | Trigger | Backend |
|---|---|---|
| scoring | `MODEL_ENDPOINT` is set | the organizers' endpoint, no key required |
| dev | `QFBENCH_DEV_API_KEY` is set | any OpenAI-compatible vendor (currently Gemini) |
| offline | `QFBENCH_FAKE_LLM=1` | canned replies — no network, no spend |

`load_dotenv()` never overwrites a real environment variable, so a local `.env` cannot shadow the
harness at scoring time.

### `chat(messages, ...)`

One chat-completion call over stdlib `urllib`. The sandbox image ships neither `openai` nor `httpx`,
and `urllib` honours the proxy environment variables automatically, which is how egress works at
scoring time. `temperature` defaults to 0 because submissions are rerun and compared.

Error handling is deliberately asymmetric: a 4xx response fails immediately, since a configuration
error should not consume the token budget, while 429 and 503 honour `Retry-After` and back off up
to 120 seconds, because rate limiting on a free tier is expected rather than exceptional.

`_ssl_context()` uses `certifi` when it is installed. A python.org macOS build ships no CA bundle,
which caused every local call to fail certificate verification until this was added. It is not
needed at scoring time, where the endpoint is reached over plain HTTP through the audited proxy.

---

## A real run

Live output from `t1-zero-coupon-bootstrapping`:

```
parse_task     1 input (/app/curve_data.json), 1 deliverable (results.json)
stage_inputs   copied to the path the instructions name
build_prompt   contract + resolved path + canary-stripped task text
chat           gemini-3.5-flash-lite, 1,187 in / 1,624 out tokens
extract_code   5,162 characters of Python
run_code       clean exit
selfgrade      6 passed, reward = 1.0
```

---

## Not built yet

| Step | What it adds |
|---|---|
| 0.4 | Output-contract validator — check deliverables parse and contain no canary before exiting |
| 0.6 | Offline evaluation harness — run all 87 tasks and produce a baseline pass rate |
| 0.7 | Repair loop — feed failures back to the model and retry, within the token budget |
| 1+ | A ported derivatives-pricing library, then the remaining task categories |
