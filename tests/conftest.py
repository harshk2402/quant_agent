# SPDX-License-Identifier: MIT
"""Shared test configuration.

The suite must never call a language model. `solve()` generates by default, and the contract tests
exercise it once per public task — so without this the suite would fire 87 live API calls on every
run, taking minutes, costing tokens and producing different results each time.

`QFBENCH_NO_LLM=1` skips generation entirely, which is what these tests are for: they cover the
deterministic half of the agent (parsing, the contract, placeholders, validation). Generation is
verified separately by running a task for real.
"""
from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True, scope="session")
def _no_model_calls_in_tests():
    os.environ["QFBENCH_NO_LLM"] = "1"
    yield
