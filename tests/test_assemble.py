# SPDX-License-Identifier: MIT
"""Tests for assembling one script out of several truncated responses.

The route caps a response at 4,000 output tokens, which is below what about half the roster's
scripts need, so this path decides whether those tasks produce a program or a SyntaxError. All of
it runs offline: `chat_fn` is injected with a scripted sequence of responses, so a truncation that
would need a live call to provoke is just a list entry here.

The cases that matter are the ugly joins. A model continuing its own output re-opens code fences,
apologises first, and re-emits the last line or two however firmly it is asked not to — and a
duplicated line is the dangerous one, because a repeated assignment parses cleanly and silently
changes what the program computes.
"""
from __future__ import annotations

import pathlib
import sys
import textwrap

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from agent import assemble  # noqa: E402
from agent.llm import LLMResponse, Usage  # noqa: E402


def _reply(text: str, finish: str = "stop") -> LLMResponse:
    return LLMResponse(text, Usage(0, 0, 1), "test-model", "fake", finish)


def _scripted(*replies):
    """A chat_fn that hands back the given replies in order, recording the turns it was sent."""
    seq = list(replies)
    calls: list[list[dict]] = []

    def chat_fn(messages, **_kw):
        calls.append(messages)
        return seq.pop(0) if seq else _reply("")

    chat_fn.calls = calls
    return chat_fn


# --- trimming a truncated script ---------------------------------------------------------------

def test_longest_parsable_prefix_drops_a_partial_statement():
    code = "import pandas as pd\nx = 1\ndf = pd.read_csv(pa"
    prefix, dropped = assemble.longest_parsable_prefix(code)
    assert prefix == "import pandas as pd\nx = 1"
    assert dropped == 1


def test_a_partial_function_is_trimmed_to_what_parses():
    """Only the incomplete tail goes. The half-written function stays, because it is valid Python
    and the model is about to continue inside it — rewinding past it would throw away work."""
    code = textwrap.dedent('''
        import math

        def price(s, k):
            d = math.log(s / k)
            return d

        def greeks(s, k):
            a = 1
            b = (a +''').strip()
    prefix, dropped = assemble.longest_parsable_prefix(code)
    assert dropped == 1
    assert prefix.endswith("a = 1")
    assert assemble.parse_error(prefix) is None


def test_truncation_inside_a_string_literal_is_handled():
    code = 'import os\nmsg = "the quick brown'
    prefix, _ = assemble.longest_parsable_prefix(code)
    assert prefix == "import os"


def test_rewinding_too_far_is_refused_rather_than_salvaged():
    """A truncation deep inside a long literal leaves too little to continue from usefully."""
    code = "data = {\n" + "".join(f'    "k{i}": {i},\n' for i in range(200))
    with pytest.raises(assemble.AssemblyError):
        assemble.longest_parsable_prefix(code)


def test_an_empty_generation_is_an_error_not_an_empty_prefix():
    with pytest.raises(assemble.AssemblyError):
        assemble.longest_parsable_prefix("")


# --- joining ------------------------------------------------------------------------------------

def test_a_continuation_wrapped_in_a_fence_is_unwrapped():
    joined = assemble.join("x = 1", "```python\ny = 2\n```")
    assert joined == "x = 1\ny = 2"
    assert assemble.parse_error(joined) is None


def test_a_conversational_opener_is_dropped():
    joined = assemble.join("x = 1", "Sure, here is the rest of the script:\n\ny = 2\n")
    assert joined == "x = 1\ny = 2"


def test_a_repeated_tail_is_removed():
    """The quiet killer: a duplicated assignment parses fine and changes the answer."""
    prefix = "import numpy as np\nreturns = px.pct_change()\nvol = returns.std()"
    joined = assemble.join(prefix, "vol = returns.std()\nsharpe = returns.mean() / vol\n")
    assert joined.count("vol = returns.std()") == 1
    assert "sharpe" in joined


def test_a_longer_repeated_run_is_removed_whole():
    prefix = "a = 1\nb = 2\nc = 3"
    joined = assemble.join(prefix, "b = 2\nc = 3\nd = 4")
    assert joined == "a = 1\nb = 2\nc = 3\nd = 4"


def test_an_indented_continuation_keeps_its_indentation():
    """A continuation that resumes inside a block must not be mistaken for prose and stripped."""
    prefix = "def f(x):\n    y = x + 1"
    joined = assemble.join(prefix, "    z = y * 2\n    return z\n")
    assert assemble.parse_error(joined) is None
    assert "    return z" in joined


def test_a_continuation_that_adds_nothing_leaves_the_prefix_alone():
    assert assemble.join("x = 1", "```python\n```") == "x = 1"


# --- the mechanical checks ----------------------------------------------------------------------

def test_parse_error_reports_a_line_and_passes_valid_code():
    assert assemble.parse_error("x = 1\ny = 2") is None
    why = assemble.parse_error("def f(:\n    pass")
    assert why and "line" in why


def test_duplicate_definitions_are_reported():
    code = "def f():\n    return 1\n\ndef f():\n    return 2\n"
    assert assemble.duplicate_definitions(code) == ["f is defined more than once"]


def test_duplicate_imports_are_reported():
    code = "import pandas as pd\nimport numpy as np\nimport pandas as pd\n"
    notes = assemble.duplicate_definitions(code)
    assert any("pandas" in n for n in notes)


def test_a_clean_script_has_no_notes():
    code = "import pandas as pd\n\ndef f():\n    return 1\n\nclass C:\n    pass\n"
    assert assemble.duplicate_definitions(code) == []


# --- the driver ---------------------------------------------------------------------------------

def test_a_complete_first_response_costs_one_call():
    chat_fn = _scripted(_reply("```python\nprint('done')\n```"))
    asm = assemble.generate_script([{"role": "user", "content": "go"}], chat_fn=chat_fn)
    assert asm.ok and asm.complete
    assert asm.responses == 1 and asm.continuations == 0
    assert asm.code == "print('done')"


def test_a_truncated_response_is_continued_and_joined():
    chat_fn = _scripted(
        _reply("```python\nimport pandas as pd\nx = 1\ndf = pd.read_csv(pa", finish="length"),
        _reply("df = pd.read_csv('in.csv')\ndf.to_csv('out.csv')\n"),
    )
    asm = assemble.generate_script([{"role": "user", "content": "go"}], chat_fn=chat_fn)
    assert asm.ok and asm.complete
    assert asm.responses == 2 and asm.continuations == 1
    assert asm.rewound_lines == 1
    assert "df.to_csv('out.csv')" in asm.code
    assert asm.code.count("import pandas as pd") == 1


def test_truncation_is_detected_without_a_finish_reason():
    """Some routes omit the field, so unparsable code is itself treated as a cut-off response."""
    chat_fn = _scripted(
        _reply("```python\nx = 1\ny = (1 +", finish=""),          # no signal at all
        _reply("y = 2\nprint(y)\n"),
    )
    asm = assemble.generate_script([{"role": "user", "content": "go"}], chat_fn=chat_fn)
    assert asm.ok and asm.continuations == 1
    assert "print(y)" in asm.code


def test_several_continuations_chain():
    chat_fn = _scripted(
        _reply("```python\na = 1\nb = (", finish="length"),
        _reply("b = 2\nc = (", finish="length"),
        _reply("c = 3\nd = 4\n"),
    )
    asm = assemble.generate_script([{"role": "user", "content": "go"}], chat_fn=chat_fn)
    assert asm.ok and asm.complete
    assert asm.continuations == 2
    assert asm.code == "a = 1\nb = 2\nc = 3\nd = 4"


def test_the_continuation_cap_is_respected_and_reported():
    chat_fn = _scripted(*[_reply("```python\nx = 1\ny = (", finish="length")] * 8)
    asm = assemble.generate_script([{"role": "user", "content": "go"}],
                                   chat_fn=chat_fn, max_continuations=2)
    assert asm.continuations == 2
    assert asm.responses == 3
    assert any("unfinished" in n for n in asm.notes)


def test_each_continuation_carries_the_script_so_far_and_not_the_earlier_instructions():
    """Input is uncapped, so the model always sees the whole file — but the turn list must not grow
    with every round, or a long assembly pays for its own history several times over."""
    chat_fn = _scripted(
        _reply("```python\na = 1\nb = (", finish="length"),
        _reply("b = 2\nc = (", finish="length"),
        _reply("c = 3\n"),
    )
    assemble.generate_script([{"role": "system", "content": "sys"},
                              {"role": "user", "content": "task"}], chat_fn=chat_fn)
    first, second, third = chat_fn.calls
    assert len(first) == 2
    assert len(second) == len(third) == 4          # system, task, assistant, continue
    assert "a = 1" in second[2]["content"]
    assert "b = 2" in third[2]["content"]          # the accumulated script, not just the first part


def test_an_unsalvageable_truncation_returns_nothing_rather_than_broken_code():
    """A truncation that landed too deep to continue from is re-asked rather than extended -- and
    when the re-ask gives nothing either, the caller gets nothing rather than broken code."""
    code = "data = {\n" + "".join(f'    "k{i}": {i},\n' for i in range(200))
    asm = assemble.generate_script([{"role": "user", "content": "go"}],
                                   chat_fn=_scripted(_reply(f"```python\n{code}", finish="length")))
    assert asm.code == "" and not asm.ok
    assert any("no usable code" in n for n in asm.notes)


def test_a_reply_with_no_code_is_reported():
    """A refusal and a response cut off at the ceiling both yield no script, but the log has to
    tell them apart: one ran out of room, the other chose not to answer, and they are worth
    different numbers of further attempts."""
    asm = assemble.generate_script([{"role": "user", "content": "go"}],
                                   chat_fn=_scripted(_reply("I cannot help with that.")))
    assert asm.code == "" and not asm.ok
    assert any("stopped of its own accord" in n for n in asm.notes)


def test_an_empty_reply_is_reported_as_no_code():
    asm = assemble.generate_script([{"role": "user", "content": "go"}],
                                   chat_fn=_scripted(_reply("   ")))
    assert asm.code == "" and any("no usable code" in n for n in asm.notes)


def test_duplicate_definitions_surface_as_notes_on_the_assembly():
    chat_fn = _scripted(
        _reply("```python\ndef f():\n    return 1\nx = (", finish="length"),
        _reply("def f():\n    return 2\n"),
    )
    asm = assemble.generate_script([{"role": "user", "content": "go"}], chat_fn=chat_fn)
    assert any("defined more than once" in n for n in asm.notes)


def test_a_passed_deadline_stops_the_assembly():
    import time
    chat_fn = _scripted(_reply("```python\nx = 1\ny = (", finish="length"),
                        _reply("y = 2\n"))
    asm = assemble.generate_script([{"role": "user", "content": "go"}], chat_fn=chat_fn,
                                   deadline=time.monotonic() - 1)
    assert asm.continuations == 0
    assert any("wall clock" in n for n in asm.notes)
    assert asm.code == "x = 1"                     # the parsable prefix is still handed back


def test_never_returns_code_that_does_not_parse():
    """The last gate: a caller must never be handed a script that is certain to fail."""
    chat_fn = _scripted(
        _reply("```python\nx = 1\nif True:", finish="length"),
        _reply("    pass\nif False:", finish="length"),
        _reply("", finish="stop"),
    )
    asm = assemble.generate_script([{"role": "user", "content": "go"}], chat_fn=chat_fn)
    assert asm.code == "" or assemble.parse_error(asm.code) is None


def test_token_usage_is_summed_per_response():
    """Counted from each reply, because SESSION_USAGE is mutated in place and a before/after
    snapshot of it is the same object — every difference would read zero."""
    def _with_usage(text, finish, tin, tout):
        return LLMResponse(text, Usage(tin, tout, 1), "m", "fake", finish)

    chat_fn = _scripted(_with_usage("```python\nx = 1\ny = (", "length", 1000, 900),
                        _with_usage("y = 2\n", "stop", 1200, 300))
    asm = assemble.generate_script([{"role": "user", "content": "go"}], chat_fn=chat_fn)
    assert asm.tokens_in == 2200 and asm.tokens_out == 1200


# --- development key rotation -------------------------------------------------------------------
#
# A free tier's allowance is daily and a full sweep is 86 units, so a second key is the difference
# between finishing a run and resuming tomorrow. Scoring has one harness-issued token and no spares.

def test_dev_keys_are_ordered_and_deduplicated(monkeypatch):
    from agent import llm
    monkeypatch.setattr(llm, "load_dotenv", lambda *a, **k: None)
    for name in ("QFBENCH_DEV_API_KEY", "QFBENCH_DEV_API_KEY2", "QFBENCH_DEV_API_KEY3",
                 "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("QFBENCH_DEV_API_KEY", "first")
    monkeypatch.setenv("QFBENCH_DEV_API_KEY3", "third")
    monkeypatch.setenv("OPENAI_API_KEY", "first")          # a duplicate must not waste an attempt
    assert llm.dev_api_keys() == ["first", "third"]


def test_the_scoring_route_has_no_spare_keys(monkeypatch):
    """Rotation is a development convenience and must never alter scoring behaviour."""
    from agent import llm
    monkeypatch.setattr(llm, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("MODEL_ENDPOINT", "http://house")
    monkeypatch.setenv("MODEL_NAME", "pinned")
    monkeypatch.setenv("MODEL_TOKEN", "t")
    monkeypatch.setenv("QFBENCH_DEV_API_KEY2", "spare")
    cfg = llm.resolve_config()
    assert cfg.mode == "scoring" and cfg.spare_keys == ()


def test_quota_refusals_are_told_from_real_faults():
    from agent.llm import looks_like_quota
    assert looks_like_quota("429 RESOURCE_EXHAUSTED: quota exceeded")
    assert looks_like_quota("Too Many Requests")
    assert not looks_like_quota("500 internal server error")
    assert not looks_like_quota("401 invalid api key")


# --- the stump problem --------------------------------------------------------------------------
#
# Found by measurement, not by review. After assembly was introduced, the dominant failure across
# seven units was `NameError` on a bare name alone on a line in the middle of a long script --
# `rankings_rows`, `pc1_`, `dST`. The cause: a truncation like `rankings_rows.append(row)` cut after
# its first token leaves `rankings_rows`, which is a VALID expression statement. The prefix parsed,
# so it looked complete, the continuation resumed after it, and the line's effect was lost.

@pytest.mark.parametrize("code,expected", [
    ("import x\nrows = []\nrankings_rows", "import x\nrows = []"),
    ("a = 1\nb = 2\ndf.append", "a = 1\nb = 2"),
    ("a = 1\nresults['key']", "a = 1"),
    ("a = 1\nx + y", "a = 1"),
    ("a = 1\nfoo, bar", "a = 1"),
])
def test_a_trailing_stump_is_dropped_even_though_it_parses(code, expected):
    prefix, dropped = assemble.longest_parsable_prefix(code)
    assert prefix == expected and dropped >= 1


def test_a_bare_call_is_kept_because_it_is_a_real_statement():
    """`write_outputs()` on its own line does something; dropping it would lose the deliverables."""
    prefix, dropped = assemble.longest_parsable_prefix("a = 1\nwrite_outputs()")
    assert prefix == "a = 1\nwrite_outputs()" and dropped == 0


def test_a_docstring_is_not_mistaken_for_a_stump():
    code = 'def f():\n    pass\n\n"""a trailing comment string"""'
    prefix, _ = assemble.longest_parsable_prefix(code)
    assert prefix.endswith('"""a trailing comment string"""')


def test_a_multiline_stump_is_cut_from_where_it_starts():
    """The expression may span lines; stepping one line at a time would accept a shorter stump."""
    code = "a = 1\nsome_frame.loc[\n    mask_one,\n    mask_two,\n]"
    prefix, _ = assemble.longest_parsable_prefix(code)
    assert prefix == "a = 1"


def test_a_stumped_truncation_is_continued_from_the_right_place():
    """End to end: the lost line must be regenerated, not stranded above the continuation."""
    chat_fn = _scripted(
        _reply("```python\nrows = []\nfor i in range(3):\n    pass\nrows", finish="length"),
        _reply("rows.append(1)\nprint(rows)\n"),
    )
    asm = assemble.generate_script([{"role": "user", "content": "go"}], chat_fn=chat_fn)
    assert asm.ok
    assert "rows.append(1)" in asm.code
    # The stump must not survive as a statement of its own.
    assert not any(line.strip() == "rows" for line in asm.code.splitlines())


# --- the static check, and the repair it drives --------------------------------------------------
#
# This replaced a hand-written list of AST node shapes that "looked like" a truncation stump. The
# analysis is delegated to pyflakes because scope rules are genuinely intricate -- comprehensions,
# class bodies, `global`, walrus, star-imports, except-clause names -- and the hand-rolled version
# was a worse approximation that missed shapes it had not enumerated.

pyflakes = pytest.importorskip("pyflakes")


def test_undefined_names_finds_what_a_lost_line_leaves_behind():
    code = ("import pandas as pd\n"
            "rows = []\n"
            "rankings_rows\n"
            "for i in range(3):\n"
            "    total = pc1_ + 1\n"
            "print(dST)\n")
    found = assemble.undefined_names(code)
    assert [f.split(" ")[0] for f in found] == ["rankings_rows", "pc1_", "dST"]
    assert "(line 3)" in found[0]


@pytest.mark.parametrize("code", [
    "import os\nx = 1\nprint(x, os.sep)\n",
    "xs = [1]\nys = [v * 2 for v in xs]\nprint(ys)\n",          # comprehension scope
    "if (n := 5) > 1:\n    print(n)\n",                          # walrus
    "try:\n    pass\nexcept ValueError as e:\n    print(e)\n",   # except-clause binding
    "class C:\n    a = 1\n    b = a + 1\nprint(C.b)\n",          # class body scope
])
def test_valid_scoping_is_not_reported(code):
    """The reason this is delegated: each of these trips a naive implementation."""
    assert assemble.undefined_names(code) == []


def test_a_lost_line_is_repaired_by_regenerating_from_the_break():
    """No patching. The script is cut back to the break and the continuation loop runs again, so a
    repair cannot fail in a way a continuation could not."""
    chat_fn = _scripted(
        # A complete-looking script whose seam lost the line that built `rankings_rows`.
        _reply("```python\nrows = []\nrankings_rows.append(1)\nprint(rows)\n```"),
        _reply("rankings_rows = []\nrankings_rows.append(1)\nprint(rankings_rows)\n"),
    )
    asm = assemble.generate_script([{"role": "user", "content": "go"}], chat_fn=chat_fn)
    assert asm.name_repairs == 1
    assert assemble.undefined_names(asm.code) == []
    assert "rankings_rows = []" in asm.code


def test_the_repair_prompt_says_which_name_was_lost():
    seen: list[str] = []

    def chat_fn(messages, **_kw):
        seen.append(messages[-1]["content"])
        if len(seen) == 1:
            return _reply("```python\nrows = []\nmissing_thing.append(1)\n```")
        return _reply("missing_thing = []\nmissing_thing.append(1)\n")

    assemble.generate_script([{"role": "user", "content": "go"}], chat_fn=chat_fn)
    assert "missing_thing" in seen[-1]
    assert "without ever assigning it" in seen[-1]


def test_a_script_that_keeps_breaking_is_reported_rather_than_patched_forever():
    chat_fn = _scripted(*[_reply("```python\nrows = []\nstill_missing.append(1)\n```")] * 8)
    asm = assemble.generate_script([{"role": "user", "content": "go"}], chat_fn=chat_fn)
    assert asm.name_repairs <= assemble.MAX_NAME_REPAIRS
    assert any("still used before assignment" in n for n in asm.notes)


def test_a_restated_cut_line_replaces_the_fragment_it_completes():
    """Evidence, not shape-matching: the model wrote the fuller version, so ours was a fragment."""
    joined = assemble.join("rows = []\nrankings_rows", "rankings_rows.append(row)\nprint(rows)")
    assert joined == "rows = []\nrankings_rows.append(row)\nprint(rows)"


def test_a_merely_similar_line_is_not_treated_as_a_restatement():
    joined = assemble.join("a = 1\nx = f(1)", "y = f(2)")
    assert joined == "a = 1\nx = f(1)\ny = f(2)"


def test_an_identical_line_is_ordinary_repetition():
    assert assemble.join("a = 1\nb = 2", "b = 2\nc = 3") == "a = 1\nb = 2\nc = 3"


def test_an_import_shadowed_by_a_loop_variable_is_reported():
    """Measured: a script died with `AttributeError: 'numpy.float64' object has no attribute
    'DataFrame'`, which is what `for pd in ...` does to the pandas alias. Checking whether `pd` is
    defined cannot catch it -- it is defined, just not what the rest of the code assumes."""
    code = ("import pandas as pd\n"
            "for pd in [1.0, 2.0]:\n"
            "    pass\n"
            "frame = pd.DataFrame([1])\n")
    found = assemble.undefined_names(code)
    assert found and found[0].startswith("pd (line 2)")


def test_ordinary_use_of_an_import_is_not_reported():
    assert assemble.undefined_names("import pandas as pd\nf = pd.DataFrame([1])\nprint(f)\n") == []


# --- named provider profiles ---------------------------------------------------------------
#
# Two providers' keys live in .env at once because they are not interchangeable: one is the model
# that actually grades us, the other is a faster, stronger stand-in for exercising the machinery.
# Switching between them has to be one line, or it will not get done.

def test_a_named_provider_selects_its_own_variables(monkeypatch):
    from agent import llm
    monkeypatch.setattr(llm, "load_dotenv", lambda *a, **k: None)
    for name in ("MODEL_ENDPOINT", "QFBENCH_FAKE_LLM", "QFBENCH_DEV_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("QFBENCH_DEV_PROVIDER", "nemotron")
    monkeypatch.setenv("QFBENCH_NEMOTRON_API_KEY", "k-nemo")
    monkeypatch.setenv("QFBENCH_NEMOTRON_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("QFBENCH_NEMOTRON_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
    monkeypatch.setenv("QFBENCH_GEMINI_API_KEY", "k-gem")          # present but not selected
    cfg = llm.resolve_config()
    assert cfg.base_url == "https://openrouter.ai/api/v1"
    assert cfg.model == "nvidia/nemotron-3-super-120b-a12b:free"
    assert cfg.api_key == "k-nemo"


def test_switching_provider_is_one_variable(monkeypatch):
    from agent import llm
    monkeypatch.setattr(llm, "load_dotenv", lambda *a, **k: None)
    for name in ("MODEL_ENDPOINT", "QFBENCH_FAKE_LLM", "QFBENCH_DEV_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("QFBENCH_NEMOTRON_API_KEY", "k-nemo")
    monkeypatch.setenv("QFBENCH_NEMOTRON_MODEL", "nemotron")
    monkeypatch.setenv("QFBENCH_GEMINI_API_KEY", "k-gem")
    monkeypatch.setenv("QFBENCH_GEMINI_MODEL", "gemini")
    monkeypatch.setenv("QFBENCH_DEV_PROVIDER", "GEMINI")
    assert llm.resolve_config().model == "gemini"
    monkeypatch.setenv("QFBENCH_DEV_PROVIDER", "NEMOTRON")
    assert llm.resolve_config().model == "nemotron"


def test_the_providers_spare_keys_are_tried_before_the_generic_ones(monkeypatch):
    from agent import llm
    monkeypatch.setattr(llm, "load_dotenv", lambda *a, **k: None)
    for n in ("", "2", "3"):
        monkeypatch.delenv(f"QFBENCH_DEV_API_KEY{n}", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("QFBENCH_DEV_PROVIDER", "nemotron")
    monkeypatch.setenv("QFBENCH_NEMOTRON_API_KEY", "n1")
    monkeypatch.setenv("QFBENCH_NEMOTRON_API_KEY2", "n2")
    monkeypatch.setenv("QFBENCH_DEV_API_KEY", "generic")
    assert llm.dev_api_keys() == ["n1", "n2", "generic"]


def test_an_unset_provider_keeps_the_old_unprefixed_behaviour(monkeypatch):
    from agent import llm
    monkeypatch.setattr(llm, "load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("QFBENCH_DEV_PROVIDER", raising=False)
    for name in ("MODEL_ENDPOINT", "QFBENCH_FAKE_LLM"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("QFBENCH_DEV_API_KEY", "plain")
    monkeypatch.setenv("QFBENCH_DEV_MODEL", "some-model")
    cfg = llm.resolve_config()
    assert cfg.api_key == "plain" and cfg.model == "some-model"


def test_the_scoring_route_ignores_the_provider_entirely(monkeypatch):
    """A dev convenience must never be able to redirect the run that is being graded."""
    from agent import llm
    monkeypatch.setattr(llm, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("QFBENCH_DEV_PROVIDER", "nemotron")
    monkeypatch.setenv("QFBENCH_NEMOTRON_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("MODEL_ENDPOINT", "http://house")
    monkeypatch.setenv("MODEL_NAME", "pinned")
    monkeypatch.setenv("MODEL_TOKEN", "t")
    cfg = llm.resolve_config()
    assert cfg.mode == "scoring" and cfg.base_url == "http://house/v1"


# --- a response with no code at all --------------------------------------------------------------
#
# The measured failure that cost the most: nemotron is a reasoning model whose thinking is charged
# against the same per-request output budget as its answer. On 8 of 15 units it spent all 4,000
# tokens reasoning and returned EMPTY content -- three times each, because the outer loop retried
# the identical prompt and got the identical result. 12,000 tokens per unit, not one line of code.
# A response that stops at `length` having emitted nothing has not refused; it has run out of room.

def test_an_empty_response_that_hit_the_ceiling_is_re_asked():
    chat_fn = _scripted(
        _reply("", finish="length"),                      # all budget spent on reasoning
        _reply("```python\nprint('ok')\n```"),
    )
    asm = assemble.generate_script([{"role": "user", "content": "go"}], chat_fn=chat_fn)
    assert asm.ok and asm.code == "print('ok')"
    assert asm.empty_retries == 1
    assert any("only reasoning" in n for n in asm.notes)


def test_the_re_ask_changes_the_instruction_rather_than_repeating_it():
    """Retrying the same prompt reproduces the same failure -- which is what the measured runs did."""
    seen = []

    def chat_fn(messages, **_kw):
        seen.append(messages[-1]["content"])
        return _reply("", finish="length") if len(seen) == 1 else _reply("```python\nx = 1\n```")

    assemble.generate_script([{"role": "user", "content": "the task"}], chat_fn=chat_fn)
    assert seen[0] == "the task"
    assert "entire output budget on internal reasoning" in seen[1]
    assert "```python as the very first characters" in seen[1]


def test_the_model_s_own_reasoning_is_not_fed_back():
    """It is draft, not answer, and returning it invites more of the behaviour we are interrupting."""
    seen = []

    def chat_fn(messages, **_kw):
        seen.append(messages)
        return _reply("", finish="length") if len(seen) == 1 else _reply("```python\nx = 1\n```")

    assemble.generate_script([{"role": "user", "content": "task"}], chat_fn=chat_fn)
    # The re-ask carries the task and the new instruction, and no assistant turn at all: there is
    # no partial script to resume from.
    assert [m["role"] for m in seen[1]] == ["user", "user"]


def test_a_refusal_is_re_asked_once_and_then_abandoned():
    """A model that stopped of its own accord is unlikely to change its mind, and each try costs a
    whole response."""
    chat_fn = _scripted(*[_reply("I cannot help with that.", finish="stop")] * 6)
    asm = assemble.generate_script([{"role": "user", "content": "go"}], chat_fn=chat_fn)
    assert asm.code == "" and not asm.ok
    assert asm.empty_retries == assemble.MAX_EMPTY_RETRIES_AFTER_STOP
    assert asm.responses == 1 + assemble.MAX_EMPTY_RETRIES_AFTER_STOP


def test_a_persistently_empty_model_is_bounded():
    chat_fn = _scripted(*[_reply("", finish="length")] * 8)
    asm = assemble.generate_script([{"role": "user", "content": "go"}], chat_fn=chat_fn)
    assert asm.empty_retries == assemble.MAX_EMPTY_RETRIES_AFTER_LENGTH
    assert asm.responses == 1 + assemble.MAX_EMPTY_RETRIES_AFTER_LENGTH
    assert any("no usable code after re-asking" in n for n in asm.notes)


def test_a_continuation_that_produces_only_reasoning_is_also_re_asked():
    """The same hole existed on the continuation path, where giving up would discard a script that
    is already most of the way written."""
    chat_fn = _scripted(
        _reply("```python\nrows = []\nfor i in range(3):\n    pass\nx = (", finish="length"),
        _reply("", finish="length"),                      # continuation spent on reasoning
        _reply("x = 1\nprint(x)\n"),
    )
    asm = assemble.generate_script([{"role": "user", "content": "go"}], chat_fn=chat_fn)
    assert asm.ok and "print(x)" in asm.code
    assert any("re-asking for code" in n for n in asm.notes)


def test_the_deadline_stops_the_re_asking():
    import time
    chat_fn = _scripted(*[_reply("", finish="length")] * 4)
    asm = assemble.generate_script([{"role": "user", "content": "go"}], chat_fn=chat_fn,
                                   deadline=time.monotonic() - 1)
    assert asm.empty_retries == 0
    assert any("ran out of wall clock" in n for n in asm.notes)


# --- plan first, implement second ----------------------------------------------------------------
#
# Measured on nemotron: a hard unit's first response spends all 4,000 tokens deliberating and
# returns EMPTY content; a follow-up forbidding reasoning moves the same deliberation into `content`
# as 15,000 characters of prose. The thinking is not optional. The only question is whether it is
# kept -- and if it is, the code call starts with the approach settled and the whole budget free.

@pytest.fixture
def plan_first(monkeypatch):
    monkeypatch.setenv("QFBENCH_NO_PLAN_FIRST", "0")


def test_the_plan_is_fed_into_the_code_call(plan_first):
    seen = []

    def chat_fn(messages, **_kw):
        seen.append(messages)
        if len(seen) == 1:
            return _reply("1. load the CSV\n2. price each option\n3. write results.csv")
        return _reply("```python\nprint('done')\n```")

    asm = assemble.generate_script([{"role": "user", "content": "task"}], chat_fn=chat_fn)
    assert asm.ok and asm.planned and asm.responses == 2
    # The plan call is the task plus a request for steps; the code call carries the plan back.
    assert "work out the approach in full" in seen[0][-1]["content"]
    code_turns = seen[1]
    assert any("price each option" in m["content"] for m in code_turns)
    assert "do not reconsider it" in code_turns[-1]["content"]


def test_a_planning_call_that_only_reasons_still_yields_a_plan(plan_first):
    """The commonest case on this model: 4,000 tokens of deliberation and no content at all. That
    reasoning IS the approach, and throwing it away is what forces us to buy it twice."""
    from agent.llm import LLMResponse, Usage
    replies = [LLMResponse("", Usage(0, 700, 1), "m", "fake", "length",
                           "We should use Crank-Nicolson with PSOR for the free boundary."),
               _reply("```python\nprint('ok')\n```")]

    def chat_fn(messages, **_kw):
        return replies.pop(0)

    asm = assemble.generate_script([{"role": "user", "content": "task"}], chat_fn=chat_fn)
    assert asm.planned and asm.ok
    assert any("only reasoning" in n for n in asm.notes)


def test_a_planning_call_that_produces_nothing_does_not_block_the_code_call(plan_first):
    replies = [_reply(""), _reply("```python\nprint('ok')\n```")]

    def chat_fn(messages, **_kw):
        return replies.pop(0)

    asm = assemble.generate_script([{"role": "user", "content": "task"}], chat_fn=chat_fn)
    assert asm.ok and not asm.planned
    assert any("produced nothing" in n for n in asm.notes)


def test_the_plan_call_gets_a_full_response(plan_first):
    """Measured: capped at 700 tokens the model spent 635 reasoning and emitted 45 characters of
    plan. It does not write a plan and stop -- it reasons its way to one, so a small cap buys a
    fragment. A whole response here bought back two elsewhere."""
    budgets = []

    def chat_fn(messages, **kw):
        budgets.append(kw.get("max_tokens"))
        return _reply("1. do the thing") if len(budgets) == 1 else _reply("```python\nx = 1\n```")

    assemble.generate_script([{"role": "user", "content": "task"}], chat_fn=chat_fn,
                             max_tokens=4000)
    assert budgets[0] == assemble.PLAN_MAX_TOKENS == 4000


def test_the_longer_of_reasoning_and_content_becomes_the_plan(plan_first):
    """Measured: one call emitted 45 characters of plan alongside 635 tokens of reasoning.
    Preferring content would have discarded almost everything the call paid for."""
    from agent.llm import LLMResponse, Usage
    replies = [LLMResponse("1. Define model parameters, seed RNG from QFB", Usage(0, 700, 1),
                           "m", "fake", "length",
                           "A much longer deliberation about Crank-Nicolson and the free boundary "
                           "that is plainly where the thinking actually went."),
               _reply("```python\nx = 1\n```")]
    seen = []

    def chat_fn(messages, **_kw):
        seen.append(messages)
        return replies.pop(0)

    asm = assemble.generate_script([{"role": "user", "content": "task"}], chat_fn=chat_fn)
    assert asm.planned
    assert "Crank-Nicolson" in seen[1][-2]["content"]
    assert any("reasoned more than it wrote" in n for n in asm.notes)


def test_an_overlong_plan_is_truncated_before_being_sent_back(plan_first):
    seen = []

    def chat_fn(messages, **_kw):
        seen.append(messages)
        return _reply("step. " * 4000) if len(seen) == 1 else _reply("```python\nx = 1\n```")

    assemble.generate_script([{"role": "user", "content": "task"}], chat_fn=chat_fn)
    assert "(plan truncated)" in seen[1][-1]["content"]


def test_planning_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("QFBENCH_NO_PLAN_FIRST", "1")
    chat_fn = _scripted(_reply("```python\nx = 1\n```"))
    asm = assemble.generate_script([{"role": "user", "content": "task"}], chat_fn=chat_fn)
    assert asm.ok and not asm.planned and asm.responses == 1


def test_a_passed_deadline_skips_planning(plan_first):
    import time
    chat_fn = _scripted(_reply("```python\nx = 1\n```"))
    asm = assemble.generate_script([{"role": "user", "content": "task"}], chat_fn=chat_fn,
                                   deadline=time.monotonic() - 1)
    assert not asm.planned


# --- the conventions ledger -----------------------------------------------------------------------
#
# Measured: a unit got the signal count exactly right and was 1.7% off on total return and 1.2%
# off on realised vol against the grader's hardcoded expectation. That is not a bug, it is a
# convention -- compounding, an annualisation factor, where a window starts -- and no checker can
# see it. The only lever is reading the task's stated rules before deciding anything.

def test_the_plan_asks_for_the_tasks_conventions_first():
    from agent.prompt import PLAN_INSTRUCTION
    assert "conventions ledger" in PLAN_INSTRUCTION
    assert PLAN_INSTRUCTION.index("conventions ledger") < PLAN_INSTRUCTION.index("THEN set out")


def test_the_ledger_forbids_substituting_a_familiar_default():
    from agent.prompt import PLAN_INSTRUCTION
    assert "Do not substitute a familiar default for a stated one" in PLAN_INSTRUCTION
    assert "if the task says 250 and you would normally use 252, the answer is 250" in PLAN_INSTRUCTION


def test_the_ledger_names_the_rules_that_move_numbers():
    from agent.prompt import PLAN_INSTRUCTION
    for rule in ("annualisation factor", "compounding", "day count", "window length",
                 "inclusive", "risk-free rate"):
        assert rule in PLAN_INSTRUCTION


def test_unstated_conventions_must_be_marked():
    from agent.prompt import PLAN_INSTRUCTION
    assert "UNSTATED" in PLAN_INSTRUCTION


def test_the_plan_demands_a_worked_example_for_every_window():
    """Measured: the plan quoted "t-12 through t-2 inclusive", reasoned "that's 11 months", and
    wrote `iloc[t-12:t]` -- 12 months -- in the same breath. The grader checks to 1e-12, so one
    extra month in the window was the whole unit. A formula hides an off-by-one; written-out
    indices cannot."""
    from agent.prompt import PLAN_INSTRUCTION
    assert "one worked example with the actual indices written out" in PLAN_INSTRUCTION
    assert "Write the indices, not the formula" in PLAN_INSTRUCTION
