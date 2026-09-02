"""Smoke-test the LLM client: which backend am I on, what can it reach, does a call work?

    python3 tools/check_llm.py            # show config + make one tiny call
    python3 tools/check_llm.py --list     # list model ids this key/endpoint exposes
"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from agent.llm import LLMError, SESSION_USAGE, chat, list_models, resolve_config  # noqa: E402

def main() -> int:
    try:
        cfg = resolve_config()
    except LLMError as e:
        print(f"CONFIG ERROR\n{e}"); return 1
    print(f"mode     : {cfg.mode}")
    print(f"base_url : {cfg.base_url or '(n/a)'}")
    print(f"model    : {cfg.model}")
    # Never print any part of the key -- length only, so logs/screenshares stay safe.
    print(f"api_key  : {'set (' + str(len(cfg.api_key)) + ' chars)' if cfg.api_key else 'none'}")

    if "--list" in sys.argv:
        try:
            ids = list_models(cfg)
        except Exception as e:
            print(f"\ncould not list models: {e}"); return 1
        print(f"\n{len(ids)} models visible; smallest/cheapest-looking first:")
        small = [m for m in ids if any(t in m for t in ("nano", "mini", "small", "3.5"))]
        for m in small or ids[:25]:
            print("   ", m)
        return 0

    try:
        r = chat([{"role": "user", "content": "Reply with exactly: OK"}], max_tokens=16)
    except LLMError as e:
        print(f"\nCALL FAILED\n{e}"); return 1
    print(f"\nreply    : {r.text.strip()!r}")
    print(f"usage    : {SESSION_USAGE}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
