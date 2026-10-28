# SPDX-License-Identifier: MIT
"""Named subsets of the practice tasks, and the one task held out of the denominator.

A full sweep is ~45 minutes and 86 model calls. That is the right thing to run before a
submission and the wrong thing to run while changing a prompt, so these fixed subsets exist to
give a signal in two or three minutes.

Two rules make the numbers comparable over time:

  * **The lists are fixed and committed.** Re-sampling between runs would mean a change in score
    could be the change you made or could be the tasks you drew. Fixed lists remove that doubt.
  * **Each set is stratified by measured OUTCOME, not just by topic.** A set of only failing tasks
    would show any change as an improvement; a set of only passing ones would show every change as
    neutral. Both are needed, plus tasks that run cleanly and still get the wrong answer, because
    those three call for entirely different work.

Outcomes referenced below come from the first baseline sweep (33 of 87 tasks, pass@1 0.182). They
are a starting point for selection, not a permanent label — a task that crashes today may run
cleanly once the repair loop exists, which is the point.

EXCLUDED is not a convenience. A task nobody can pass is a constant subtracted from every score,
and leaving it in the denominator makes each run look worse by the same fixed amount while hiding
real movement in the noise. Taking it out has to be visible rather than quiet, though, which is why
the reason is written here and the sweep prints what it skipped — a shrinking denominator is
exactly how a benchmark gets talked into flattering itself.
"""
from __future__ import annotations

#: The default working set: 16 tasks, roughly 3 minutes, spanning the phase buckets in PLAN.md
#: and all three measured outcomes. Intended for the repair-loop work (Step 0.7).
REPRESENTATIVE = [
    # --- passing today: regression guard. If a change breaks these, it is not an improvement.
    "t1-EXAMPLE-bs-greeks-pde",          # derivatives pricing; the exemplar
    "t1-cir-bond-pricing",               # fixed income
    "t1-bollinger-backtest-aapl",        # backtesting
    "t1-copula-equity-fitting",          # factor / statistical
    "t1-delta-hedging-pnl-simulation",   # derivatives, path-dependent

    # --- crashed today: the repair loop's target, and the largest single bucket (55%)
    "t1-barone-adesi-whaley",            # derivatives pricing, American approximation
    "t1-american-option-fd-new",         # PDE grid, multi-file output
    "t1-bs-greeks-pde",                  # PDE; the non-exemplar variant
    "t1-credit-portfolio-var-cvar",      # risk management, large Monte Carlo
    "t1-corporate-action-adjustment",    # data processing rather than modelling
    "t1-earnings-surprise-calculator",   # simple task, so a crash here is a harness signal
    "t1-double-sort",                    # factor research

    # --- ran cleanly but wrong: domain knowledge, not plumbing. Repair should NOT fix these,
    #     which makes them a useful control on whether repair is doing what we think.
    "t1-asian-option-levy-curran",       # derivatives pricing, analytic approximation
    "t1-credit-migration-matrix",        # credit
    "t1-dcc-garch-portfolio-var",        # risk, multivariate GARCH

]

#: Five tasks, under a minute. For "did I break the pipeline", not for measuring quality.
SMOKE = [
    "t1-EXAMPLE-bs-greeks-pde",
    "t1-cir-bond-pricing",
    "t1-barone-adesi-whaley",
    "t1-asian-option-levy-curran",
]

#: Tasks held out of the roster, with the reason. Keyed by unit id so the sweep can both skip them
#: and say that it did.
EXCLUDED: dict[str, str] = {
    # The deliverable is a Python module, not data, and the unit's checker imports that module and
    # asserts on files which only exist once it has been RUN. Nothing the agent can write to the
    # output directory satisfies it, because the grader never executes what we hand in. Verified by
    # reading the checker rather than inferred from the task failing (see PLAN.md 4.8).
    "t1-polars-api-migration": "checker asserts on artefacts of running the deliverable, which the "
                               "grader never runs — unreachable, not merely hard",
}


def is_excluded(unit_id: str) -> bool:
    return unit_id in EXCLUDED


SETS = {
    "representative": REPRESENTATIVE,
    "smoke": SMOKE,
}


def resolve(name: str) -> list[str]:
    if name not in SETS:
        raise KeyError(f"unknown set {name!r}; available: {', '.join(sorted(SETS))}")
    return list(SETS[name])
