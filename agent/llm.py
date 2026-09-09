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
import pathlib
import ssl
import time
import urllib.error
import urllib.request

# Deliberately small/cheap default: the house model is an open model, so developing
# against a frontier model invites over-tuning.  Override with QFBENCH_DEV_MODEL.
DEFAULT_DEV_MODEL = "gemini-3.5-flash-lite"
DEFAULT_DEV_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"

# Provisional per-unit budget from SUBMISSION_CLI.md; not yet finalised by the organizers.
BUDGET_INPUT_TOKENS = 1_000_000
BUDGET_OUTPUT_TOKENS = 100_000


class LLMError(RuntimeError):
    """Unrecoverable model-call failure (config, auth, or exhausted retries)."""


class BudgetExceeded(LLMError):
    """The per-unit token budget would be exceeded; stop rather than overspend."""


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

    @property
    def is_scoring(self) -> bool:
        return self.mode == "scoring"


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
        return LLMConfig("scoring", endpoint.rstrip("/"), model,
                         os.environ.get("MODEL_API_KEY") or None)

    # Provider-neutral names, with the older OPENAI_* names accepted as fallbacks.
    key = (os.environ.get("QFBENCH_DEV_API_KEY")
           or os.environ.get("OPENAI_API_KEY", "")).strip()
    if not key:
        raise LLMError(
            "No model backend configured.\n"
            "  scoring : the harness sets MODEL_ENDPOINT + MODEL_NAME (nothing to do)\n"
            "  dev     : put QFBENCH_DEV_API_KEY=... in .env (see .env.example for\n"
            "            ready-made Gemini / Anthropic / OpenAI / Ollama blocks)\n"
            "  offline : export QFBENCH_FAKE_LLM=1 to run the loop without a model"
        )
    base = (os.environ.get("QFBENCH_DEV_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or DEFAULT_DEV_BASE_URL)
    return LLMConfig("dev", base.rstrip("/"),
                     os.environ.get("QFBENCH_DEV_MODEL", DEFAULT_DEV_MODEL), key)


# ---------------------------------------------------------------------------
# usage metering
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.calls += other.calls

    def remaining(self) -> tuple[int, int]:
        return (BUDGET_INPUT_TOKENS - self.input_tokens,
                BUDGET_OUTPUT_TOKENS - self.output_tokens)

    def __str__(self) -> str:
        ri, ro = self.remaining()
        return (f"{self.calls} calls, in={self.input_tokens} out={self.output_tokens} "
                f"(remaining in={ri} out={ro})")


SESSION_USAGE = Usage()   # per-unit cumulative; the repair loop reads this to stop in time


@dataclasses.dataclass
class LLMResponse:
    text: str
    usage: Usage
    model: str
    mode: str


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
    max_tokens: int = 8000,
    timeout: float = 180.0,
    retries: int = 4,
    config: LLMConfig | None = None,
    enforce_budget: bool = True,
) -> LLMResponse:
    """One chat-completion call. Deterministic by default (temperature=0).

    Reproducibility: the harness reruns and compares, so temperature defaults to 0 and
    callers should not raise it without a reason.
    """
    cfg = config or resolve_config()

    if cfg.mode == "fake":
        return _fake_response(messages, cfg)

    if enforce_budget:
        rem_in, rem_out = SESSION_USAGE.remaining()
        if rem_in <= 0 or rem_out <= 0:
            raise BudgetExceeded(f"per-unit token budget exhausted: {SESSION_USAGE}")
        max_tokens = min(max_tokens, max(rem_out, 1))

    payload = {
        "model": cfg.model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    url = f"{cfg.base_url}/chat/completions"

    last: Exception | None = None
    for attempt in range(retries):
        try:
            data = _post(url, payload, headers, timeout)
            break
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:400]
            # 4xx other than rate-limit is a config/prompt bug: retrying just wastes budget.
            if e.code in (400, 401, 403, 404):
                raise LLMError(f"{cfg.mode} call failed ({e.code}) at {url}: {body}") from e
            last = LLMError(f"HTTP {e.code}: {body}")
            # 429/503 on a free tier is normal, not an error -- wait as long as told to.
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
        text = data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as e:
        raise LLMError(f"unexpected response shape: {str(data)[:400]}") from e

    u = data.get("usage") or {}
    usage = Usage(int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0)), 1)
    SESSION_USAGE.add(usage)
    return LLMResponse(text, usage, cfg.model, cfg.mode)


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
    return LLMResponse(canned, usage, cfg.model, "fake")


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