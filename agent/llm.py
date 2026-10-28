# SPDX-License-Identifier: MIT
"""OpenAI-compatible LLM client for the QFBench T1 agent.  (Phase 0, Step 0.2)

One code path, two deployments
------------------------------
At scoring time the ONLY reachable model is the organizer-hosted server named by
`$MODEL_ENDPOINT`, speaking the OpenAI-compatible `/chat/completions` protocol with the
pinned id in `$MODEL_NAME`.  Vendor APIs (api.openai.com, api.anthropic.com, ...) are
refused by the audited proxy and no participant key is injected -- so a vendor key is
useful for LOCAL DEVELOPMENT ONLY and must never be baked into the image.

Because production is already OpenAI-compatible, this client speaks only that protocol
and treats the endpoint as configuration.  Dev and scoring differ by environment
variables, never by code -- the path exercised locally is the path that ships.

    scoring : MODEL_ENDPOINT + MODEL_NAME set by the harness (no key; proxy authenticates)
    dev     : QFBENCH_DEV_{API_KEY,BASE_URL,MODEL} from .env -- ANY vendor that publishes an
              OpenAI-compatible endpoint (Google Gemini, Anthropic, OpenAI, Ollama, vLLM).
              The provider is configuration, never code.
    offline : QFBENCH_FAKE_LLM=1 -> canned responses, no network, no spend

Transport is stdlib `urllib` on purpose: the sandbox base image ships neither `openai`
nor `httpx`, and adding a pinned SDK is exactly the dependency-churn risk we chose to
avoid.  `urllib` also honours HTTP_PROXY/HTTPS_PROXY automatically, which is how egress
works at scoring time.

⚠️  Dev capability != scoring capability.  The house model is an OPEN model (Qwen/Llama
class).  Prompts tuned against a stronger dev model can regress at scoring, so keep
prompts explicit and instruction-grounded, and treat dev pass rates as indicative only.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import sys
import pathlib
import ssl
import time
import urllib.error
import urllib.request

# Deliberately small/cheap default: the house model is an open model, so developing
# against a frontier model invites over-tuning.  Override with QFBENCH_DEV_MODEL.
DEFAULT_DEV_MODEL = "gemini-3.5-flash-lite"
DEFAULT_DEV_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"

# The per-unit allowance is counted in REQUESTS, not tokens. The Submission Format page is explicit
# (ruling of 2026-09-21): 25 admitted requests per unit, at most 4,000 output tokens per request,
# both applied by the House route itself — and "there is no per-unit token budget ... the per-unit
# token figure this page carried before that ruling is withdrawn".
#
# An earlier version of this file metered a 1,000,000/100,000 token budget, which was the withdrawn
# figure. It counted the wrong quantity, and it would never have bound: a solve uses single-digit
# requests. What binds is the request count, so that is what is tracked.
#
# A request is charged on ADMISSION, before forwarding, so a retry or a lost response still spends
# a slot. Only a request refused before admission is free. The reserve below therefore leaves room
# for transport retries rather than assuming every call lands.
BUDGET_REQUESTS = 25
MAX_OUTPUT_TOKENS = 4_000


class LLMError(RuntimeError):
    """Unrecoverable model-call failure (config, auth, or exhausted retries)."""


class BudgetExceeded(LLMError):
    """The per-unit request allowance is spent; stop rather than overspend."""


# ---------------------------------------------------------------------------
# .env loading (dev convenience; absent and ignored at scoring time)
# ---------------------------------------------------------------------------

def load_dotenv(path: str | os.PathLike[str] | None = None) -> None:
    """Load KEY=VALUE lines from .env into os.environ WITHOUT overriding real env vars.

    Real environment always wins, so a stray local .env can never shadow the harness's
    MODEL_ENDPOINT at scoring time.
    """
    p = pathlib.Path(path) if path else pathlib.Path(__file__).resolve().parents[1] / ".env"
    if not p.is_file():
        return
    for raw in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class LLMConfig:
    mode: str          # "scoring" | "dev" | "fake"
    base_url: str
    model: str
    api_key: str | None
    #: Further DEV keys, tried in order when one is refused for quota. Development only: the
    #: scoring route has exactly one token, issued by the harness, and nothing to rotate to.
    spare_keys: tuple[str, ...] = ()

    @property
    def is_scoring(self) -> bool:
        return self.mode == "scoring"


#: Signs that a refusal is about an allowance rather than about the request. Deliberately broad:
#: the cost of reading a genuine fault as a quota problem is one wasted retry on another key, while
#: the cost of the reverse is a sweep stopping with hours of allowance unused.
_QUOTA_SIGNS = ("429", "rate limit", "ratelimit", "quota", "resource_exhausted", "exhausted",
                "too many requests")


#: Routes that have refused the `reasoning` field, so it is sent once per route and then dropped.
#: Without this every call would pay a rejected request plus a retry, and at 25 requests per unit
#: that is most of the allowance spent on a field we already know is unwelcome.
_REASONING_REFUSED: set[str] = set()


def reasoning_control() -> dict | None:
    """The `reasoning` field to send, or None to leave the provider's default alone.

    `QFBENCH_REASONING=on` leaves it alone; anything else (including unset) disables it.
    Off by default because of what it costs us: this model spends most of every response
    deliberating, charged against the SAME 4,000-token ceiling as the answer, so on a hard unit it
    reasons for the whole budget and returns no code at all. Measured across eight controls on this
    provider -- system prompts, `/nothink`, `chat_template_kwargs`, `reasoning_effort` -- this is
    the only one that works: reasoning tokens go to zero and the answer still arrives.

    The deliberation still has to happen somewhere. It happens in the planning call, where the
    output is kept and fed forward, instead of being thrown away once per request.
    """
    if os.environ.get("QFBENCH_REASONING", "").strip().lower() == "on":
        return None
    return {"enabled": False}


def looks_like_quota(text: str) -> bool:
    return any(sign in (text or "").lower() for sign in _QUOTA_SIGNS)


def dev_provider() -> str:
    """The named development provider in use, or "" for the unprefixed `QFBENCH_DEV_*` variables.

    Set `QFBENCH_DEV_PROVIDER=nemotron` and the client reads `QFBENCH_NEMOTRON_API_KEY`,
    `QFBENCH_NEMOTRON_BASE_URL` and `QFBENCH_NEMOTRON_MODEL`. The point is that several providers'
    keys can sit in `.env` at once and switching between them is one line rather than re-pasting
    three -- which matters here because the two are not interchangeable: one is the model that
    actually grades us, and the other is a faster, stronger stand-in useful only for exercising
    the machinery.
    """
    load_dotenv()
    return re.sub(r"[^A-Z0-9_]", "", os.environ.get("QFBENCH_DEV_PROVIDER", "").strip().upper())


def _provider_var(name: str, provider: str | None = None) -> str:
    """Read `QFBENCH_<PROVIDER>_<NAME>`, falling back to `QFBENCH_DEV_<NAME>`."""
    provider = dev_provider() if provider is None else provider
    if provider:
        value = (os.environ.get(f"QFBENCH_{provider}_{name}") or "").strip()
        if value:
            return value
    return (os.environ.get(f"QFBENCH_DEV_{name}") or "").strip()


def dev_api_keys() -> list[str]:
    """Every development key configured, in the order they should be tried.

    The selected provider's key first, then its numbered spares, then the unprefixed
    `QFBENCH_DEV_API_KEY` family, then the older `OPENAI_API_KEY`. Several keys exist because a
    free tier has a daily allowance and a sweep is hundreds of requests: exhausting one should move
    to the next rather than abandon the run. Order is preserved and duplicates are dropped, so the
    same key listed twice cannot waste an attempt.
    """
    load_dotenv()                 # callers outside resolve_config() must see .env too
    provider = dev_provider()
    names: list[str] = []
    if provider:
        names.append(f"QFBENCH_{provider}_API_KEY")
        names += [f"QFBENCH_{provider}_API_KEY{n}" for n in range(2, 10)]
    names.append("QFBENCH_DEV_API_KEY")
    names += [f"QFBENCH_DEV_API_KEY{n}" for n in range(2, 10)]
    names.append("OPENAI_API_KEY")
    seen: list[str] = []
    for name in names:
        key = (os.environ.get(name) or "").strip()
        if key and key not in seen:
            seen.append(key)
    return seen


def resolve_config() -> LLMConfig:
    """Pick the backend from the environment. MODEL_ENDPOINT always wins."""
    load_dotenv()

    if os.environ.get("QFBENCH_FAKE_LLM") == "1":
        return LLMConfig("fake", "", os.environ.get("QFBENCH_DEV_MODEL", "fake-model"), None)

    endpoint = os.environ.get("MODEL_ENDPOINT", "").strip()
    if endpoint:
        model = os.environ.get("MODEL_NAME", "").strip()
        if not model:
            raise LLMError("MODEL_ENDPOINT is set but MODEL_NAME is empty -- the harness "
                           "pins the house model id in MODEL_NAME; refusing to guess.")
        # MODEL_ENDPOINT is an ORIGIN (scheme://host:port) with no path. The OpenAI-compatible API
        # is served under /v1, and the submission contract is explicit that a request to
        # MODEL_ENDPOINT/chat/completions, without the /v1, is refused with 403. Appending it here
        # is safe either way: a `.rstrip("/")` endpoint that already ends in /v1 is left alone.
        base = endpoint.rstrip("/")
        if not base.endswith("/v1"):
            base = base + "/v1"
        # The House route requires `Authorization: Bearer $MODEL_TOKEN` on every request and answers
        # 401 without it. This is per-unit and supplied by the harness; it is NOT the old
        # MODEL_API_KEY name, which does not exist in the evaluation environment.
        token = os.environ.get("MODEL_TOKEN", "").strip() or None
        if not token:
            # Not fatal here: fail at the call with the route's own error rather than refusing to
            # start, since a missing token is an environment fault we cannot fix from inside.
            print("[llm] warning: MODEL_ENDPOINT is set but MODEL_TOKEN is empty; the House route "
                  "will answer 401", file=sys.stderr)
        return LLMConfig("scoring", base, model, token)

    keys = dev_api_keys()
    key = keys[0] if keys else ""
    if not key:
        raise LLMError(
            "No model backend configured.\n"
            "  scoring : the harness sets MODEL_ENDPOINT + MODEL_NAME (nothing to do)\n"
            "  dev     : put QFBENCH_DEV_API_KEY=... in .env (see .env.example for\n"
            "            ready-made Gemini / Anthropic / OpenAI / Ollama blocks)\n"
            "  offline : export QFBENCH_FAKE_LLM=1 to run the loop without a model"
        )
    base = (_provider_var("BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or DEFAULT_DEV_BASE_URL)
    model = _provider_var("MODEL") or DEFAULT_DEV_MODEL
    return LLMConfig("dev", base.rstrip("/"), model, key, spare_keys=tuple(keys[1:]))


# ---------------------------------------------------------------------------
# usage metering
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    #: Of `output_tokens`, how many went on the model's internal reasoning rather than its answer.
    #: A thinking model is charged for both against the SAME per-request ceiling, so this is the
    #: share of the 4,000-token budget that never reaches the script. Measured once by hand it was
    #: 77 of 88 tokens; recorded here so the share is visible across a whole run instead.
    reasoning_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.calls += other.calls
        self.reasoning_tokens += other.reasoning_tokens

    def requests_remaining(self) -> int:
        """Admitted requests left in this unit's allowance."""
        return BUDGET_REQUESTS - self.calls

    def __str__(self) -> str:
        thinking = (f" ({self.reasoning_tokens} of them reasoning)"
                    if self.reasoning_tokens else "")
        return (f"{self.calls}/{BUDGET_REQUESTS} requests, "
                f"in={self.input_tokens} out={self.output_tokens} tokens{thinking}")


SESSION_USAGE = Usage()   # per-unit cumulative; the repair loop reads this to stop in time


@dataclasses.dataclass
class LLMResponse:
    text: str
    usage: Usage
    model: str
    mode: str
    #: Why the model stopped. "stop" is a complete answer; "length" means it ran into the output
    #: ceiling and the text is cut off mid-stream. Routes that omit the field give "" -- absence is
    #: not evidence of completion, which is why the assembler also checks that the code parses.
    finish_reason: str = ""
    #: The model's internal deliberation, when the route returns it separately from the answer.
    #: Worth keeping rather than discarding: on a hard task this model spends its ENTIRE 4,000-token
    #: response working out an approach and emits no content at all, so this is often the only thing
    #: we got for a whole request. Feeding it into the next call turns a wasted response into an
    #: input, which is the cheapest plan we will ever get.
    reasoning: str = ""

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


# ---------------------------------------------------------------------------
# the call
# ---------------------------------------------------------------------------

def _ssl_context():
    """TLS context for dev calls.

    A python.org macOS build ships no CA bundle, so HTTPS to a vendor endpoint fails with
    CERTIFICATE_VERIFY_FAILED even though the same call works under a system Python. Prefer
    `certifi` when it is installed and fall back to the system store otherwise.

    Not required at scoring time, where the house endpoint is reached over plain HTTP through the
    audited proxy; it exists so local development works under any interpreter.
    """
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return None          # urllib then uses the interpreter's default verification


def _post(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
        return json.loads(resp.read().decode())                  # urlopen honours *_PROXY env


def chat(
    messages: list[dict],
    *,
    temperature: float = 0.0,
    max_tokens: int = MAX_OUTPUT_TOKENS,
    seed: int | None = None,
    timeout: float = 180.0,
    retries: int = 4,
    config: LLMConfig | None = None,
    enforce_budget: bool = True,
) -> LLMResponse:
    """One chat-completion call. Deterministic by default (temperature=0).

    `max_tokens` is capped at the House route's per-request ceiling of 4,000 output tokens, whatever
    the caller asks for. Sampling settings are fixed in code rather than varied per call: the rules
    require it, though they do not require zero specifically.
    """
    cfg = config or resolve_config()

    if cfg.mode == "fake":
        return _fake_response(messages, cfg)

    if enforce_budget and SESSION_USAGE.requests_remaining() <= 0:
        raise BudgetExceeded(f"per-unit request allowance spent: {SESSION_USAGE}")

    # Hard cap, not a default: the House route applies 4,000 output tokens per request itself, and
    # asking for more risks the request being refused rather than merely truncated.
    max_tokens = min(max_tokens, MAX_OUTPUT_TOKENS)

    payload = {
        "model": cfg.model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    # temperature=0 is NOT determinism: these routes batch requests and accumulate floating point
    # differently depending on what else is in the batch, so ties break differently. Measured here
    # -- the same prompt twice produced 312- and 271-line scripts that diverged at line 9. A seed
    # is the only lever on that, and it matters twice over: without it no before/after comparison
    # means anything, and the competition re-runs submissions to check they reproduce.
    #
    # Sent only when asked for, and withdrawn on a 400 (below), because an unknown field is a hard
    # refusal on some routes and a refused call at scoring time costs the whole task.
    if seed is None:
        raw = os.environ.get("QFBENCH_MODEL_SEED", "").strip()
        seed = int(raw) if raw.lstrip("-").isdigit() else None
    if seed is not None:
        payload["seed"] = seed

    control = reasoning_control()
    if control is not None and cfg.base_url not in _REASONING_REFUSED:
        payload["reasoning"] = control
    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    url = f"{cfg.base_url}/chat/completions"

    # Development only. A free tier's allowance is daily and a full sweep is 86 units, so a second
    # key is the difference between finishing a run and resuming it tomorrow. The scoring route has
    # one token issued by the harness and `spare_keys` is empty there, so this never fires.
    key_ring = [k for k in (cfg.api_key, *cfg.spare_keys) if k]
    key_index = 0

    last: Exception | None = None
    for attempt in range(retries):
        # Count the attempt BEFORE sending. The House route charges a request on admission, so a
        # timeout, a dropped response or a retry all spend a slot even though no usable answer came
        # back. Counting only successes would let a few retries quietly overrun the 25-request
        # allowance and have later units refused.
        SESSION_USAGE.add(Usage(0, 0, 1))
        if enforce_budget and SESSION_USAGE.requests_remaining() < 0:
            raise BudgetExceeded(f"per-unit request allowance spent: {SESSION_USAGE}")
        try:
            data = _post(url, payload, headers, timeout)
            break
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:400]
            # 4xx other than rate-limit is a config or auth fault, not bad luck: retrying cannot
            # fix it and each retry spends another slot. 401 means MODEL_TOKEN is missing or wrong;
            # 403 at scoring most likely means the URL lacks the required /v1 prefix.
            if e.code in (400, 404, 422) and ("reasoning" in payload or "seed" in payload):
                # The route may not take an optional field. Drop both and retry once, whatever the
                # body says: a generic "unknown field" message would otherwise fail every call at
                # scoring time, and answering matters more than controlling the thinking or the
                # seed. Remembered per route so later calls do not pay the refused request again.
                payload.pop("reasoning", None)
                payload.pop("seed", None)
                _REASONING_REFUSED.add(cfg.base_url)
                print("[llm] this route rejected the request; retrying without the optional "
                      "`reasoning`/`seed` fields", file=sys.stderr)
                continue
            if e.code in (400, 401, 403, 404):
                raise LLMError(f"{cfg.mode} call failed ({e.code}) at {url}: {body}") from e
            last = LLMError(f"HTTP {e.code}: {body}")
            # A refused request on one key may be accepted on another: a per-minute limit is scoped
            # to the project the key belongs to, and a daily allowance certainly is. Rotating is
            # tried BEFORE waiting, because when it works the wait was unnecessary -- and when no
            # key is left the backoff below still happens.
            if (e.code == 429 and looks_like_quota(f"{e.code} {body}")
                    and key_index + 1 < len(key_ring)):
                key_index += 1
                headers["Authorization"] = f"Bearer {key_ring[key_index]}"
                print(f"[llm] key {key_index} was refused for quota; trying key "
                      f"{key_index + 1} of {len(key_ring)}", file=sys.stderr)
                continue
            # 429/503 is congestion rather than an error -- wait as long as told to.
            if e.code in (429, 503):
                hdr = (e.headers.get("Retry-After") or "").strip() if e.headers else ""
                retry_after = float(hdr) if hdr.replace(".", "", 1).isdigit() else None
                if attempt < retries - 1:
                    time.sleep(min(retry_after if retry_after else 2 ** attempt * 5, 120))
                    continue
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last = e
        if attempt < retries - 1:
            time.sleep(min(2 ** attempt, 20))
    else:
        raise LLMError(f"model call failed after {retries} attempts: {last}")

    try:
        choice = data["choices"][0]
        text = choice["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as e:
        raise LLMError(f"unexpected response shape: {str(data)[:400]}") from e
    # Recorded rather than inferred. Until now a response cut off at the ceiling was
    # indistinguishable from a complete one, which is how half the roster came to be generating
    # scripts that ran into the 4,000-token cap without anything noticing.
    finish_reason = str(choice.get("finish_reason") or "")
    reasoning = str(choice.get("message", {}).get("reasoning") or "")

    # Tokens only -- the request itself was already counted above.
    u = data.get("usage") or {}
    details = u.get("completion_tokens_details") or {}
    usage = Usage(int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0)), 1,
                  int(details.get("reasoning_tokens", 0) or 0))
    SESSION_USAGE.add(Usage(usage.input_tokens, usage.output_tokens, 0, usage.reasoning_tokens))
    return LLMResponse(text, usage, cfg.model, cfg.mode, finish_reason, reasoning)


def _fake_response(messages: list[dict], cfg: LLMConfig) -> LLMResponse:
    """Deterministic stub so loop mechanics can be tested without spending anything.

    Returns a trivially valid python block; override with QFBENCH_FAKE_REPLY.
    """
    canned = os.environ.get(
        "QFBENCH_FAKE_REPLY",
        "```python\nprint('fake llm: no model was called')\n```",
    )
    usage = Usage(sum(len(m.get("content", "")) // 4 for m in messages), len(canned) // 4, 1)
    SESSION_USAGE.add(usage)
    # QFBENCH_FAKE_FINISH lets a test drive the truncation path without a network call. The
    # assembler's whole job is reacting to "length", so it has to be reachable offline.
    return LLMResponse(canned, usage, cfg.model, "fake",
                       os.environ.get("QFBENCH_FAKE_FINISH", "stop"))


def complete(prompt: str, *, system: str | None = None, **kw) -> str:
    """Convenience wrapper: single-turn prompt in, text out."""
    msgs: list[dict] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    return chat(msgs, **kw).text


def list_models(config: LLMConfig | None = None) -> list[str]:
    """GET /models -- used by tools/check_llm.py to confirm what this key can actually see."""
    cfg = config or resolve_config()
    if cfg.mode == "fake":
        return [cfg.model]
    req = urllib.request.Request(f"{cfg.base_url}/models", method="GET")
    if cfg.api_key:
        req.add_header("Authorization", f"Bearer {cfg.api_key}")
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read().decode())
    return sorted(m.get("id", "") for m in data.get("data", []))