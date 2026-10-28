# SPDX-License-Identifier: MIT
"""Classical-finance invariants on our own output, before the container exits.

WHY THIS EXISTS. Everything else in `validate.py` asks whether a file is *admissible* — present,
parseable, free of canaries. This module asks a harder question: *could these numbers possibly be
right?* It cannot know the answer to the task, but a great many wrong answers are wrong in a way
that is visible without the answer, because finance has relationships that hold for every correct
result regardless of the method used to produce it. A call option worth less than its intrinsic
value is wrong. A survival probability that rises with time is wrong. Expected shortfall below
value-at-risk at the same confidence is wrong. None of those require a reference solution.

That is also precisely how we are graded. Track 1's fourth gate, `g3_domain_semantics`, applies
category-specific sanity checks, and the public graders' assertion names (`prices_positive`,
`sigma_range`, `vega_positive`, `survival_monotone`) show the shape of them. So this layer is an
attempt to run a version of the grader's own reasoning against our output while there is still time
to repair it — at zero model cost, deterministically, in milliseconds.

HOW A CHECK FINDS ITS DATA. We never know a task's column names in advance, so each check resolves
the *roles* it needs (a spot price, a strike, a maturity) from column names by pattern. If a role is
missing, or two columns claim it, the check declines to run rather than guessing. Silence here means
"not applicable", never "verified".

THE RULE ABOUT FALSE ALARMS. A false alarm is more expensive than a missed problem: it sends the
repair loop to rewrite code that was already correct, spending an attempt and part of a 25-request
budget on nothing, and the rewrite may be worse. Three habits keep that under control:

  * **Unit ambiguity is resolved in the output's favour.** A rate may be 0.05 or 5.0; a maturity may
    be years, days or trading days. A relational check is evaluated under every plausible convention
    and fires only if *all* of them are violated. One consistent reading that works is a pass.
  * **Systematic, not incidental.** Relational checks look at the whole column; a single odd row in
    a thousand is far more likely to be an edge case the task intended than a broken formula.
  * **ERROR only where no convention can save it.** A negative hazard rate is meaningless under any
    convention — that is an error. A gamma that is negative could be a spread or a short position
    unless the category says otherwise — that warns.

Findings are returned as plain `(level, code, message)` tuples, so this module imports nothing from
`validate` and can be tested on its own.
"""

from __future__ import annotations

import math
import re
import warnings

#: Mirrors validate.ERROR / validate.WARNING without importing them (keeps the dependency one-way).
ERROR = "error"
WARNING = "warning"

#: One finding: (level, code, message). `code` is stable so the repair loop can branch on it.
Violation = tuple[str, str, str]


def _norm(name: object) -> str:
    """Canonical form of a column name: lowercase, non-alphanumerics collapsed to underscores."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


# ---------------------------------------------------------------------------
# role resolution
# ---------------------------------------------------------------------------
#
# A role is a quantity a check needs; the patterns are matched against the NORMALISED column name.
# They are anchored on purpose: a substring match would bind `strike_delta` to the strike role and
# invent a violation out of nothing.

_ROLES: dict[str, list[str]] = {
    "spot":        [r"^s$", r"^spot$", r"^spot_price$", r"^underlying(_price)?$", r"^s0$"],
    "strike":      [r"^k$", r"^strike$", r"^strike_price$"],
    "rate":        [r"^r$", r"^rate$", r"^risk_free(_rate)?$", r"^rf$", r"^r_d$",
                    r"^domestic_rate$", r"^r_domestic$"],
    "rate_foreign": [r"^r_f$", r"^foreign_rate$", r"^r_foreign$", r"^q$", r"^div(idend)?_yield$"],
    "maturity":    [r"^t$", r"^tau$", r"^maturity$", r"^time_to_(expiry|maturity)$", r"^ttm$",
                    r"^expiry_years$", r"^years?$", r"^tenor(_years)?$", r"^term$"],
    "call_price":  [r"^call(_price|_value|_premium)?$", r"^c$", r"^price_call$"],
    "put_price":   [r"^put(_price|_value|_premium)?$", r"^p$", r"^price_put$"],
    "call_delta":  [r"^call_delta$", r"^delta_call$"],
    "put_delta":   [r"^put_delta$", r"^delta_put$"],
    "gamma":       [r"^gamma$"],
    "vega":        [r"^vega$"],
    "bid":         [r"^bid$", r"^bid_price$", r"^best_bid$"],
    "ask":         [r"^ask$", r"^ask_price$", r"^best_ask$", r"^offer$"],
    "high":        [r"^high$", r"^day_high$", r"^high_price$"],
    "low":         [r"^low$", r"^day_low$", r"^low_price$"],
    "vwap":        [r"^vwap$", r"^volume_weighted_avg_price$", r"^execution_price$"],
    "forward":     [r"^f$", r"^forward$", r"^forward_rate$", r"^fwd$"],
    "discount":    [r"^df$", r"^discount$", r"^discount_factor$"],
    "survival":    [r"^survival(_prob(ability)?)?$", r"^s_t$", r"^q_survival$"],
    "hazard":      [r"^hazard(_rate)?$", r"^lambda$", r"^intensity$", r"^default_intensity$"],
    "mod_duration": [r"^mod(ified)?_duration$", r"^moddur$"],
    "dv01":        [r"^dv01$", r"^pv01$", r"^dollar_duration$"],
    "price":       [r"^price$", r"^clean_price$", r"^dirty_price$", r"^bond_price$"],
}


def _resolve(columns, role: str) -> str | None:
    """The single column filling `role`, or None when absent or ambiguous.

    Ambiguity returns None deliberately. Two columns both looking like a strike means we do not
    know which one a relationship is about, and a check run on the wrong one is a false alarm.
    """
    pats = [re.compile(p) for p in _ROLES[role]]
    hits = [c for c in columns if any(p.match(_norm(c)) for p in pats)]
    return hits[0] if len(hits) == 1 else None


def _roles_of(columns) -> dict[str, str]:
    """Every role that resolves unambiguously in this set of columns."""
    out = {}
    for role in _ROLES:
        col = _resolve(columns, role)
        if col is not None:
            out[role] = col
    return out


# ---------------------------------------------------------------------------
# scalar bounds, by column name
# ---------------------------------------------------------------------------
#
# These extend validate._VALUE_RULES into the vocabulary of the ten task categories. They are
# bounds on a single quantity, so they apply to a column of a table and to a scalar in a JSON
# document alike. Every one is a definitional bound — violating it is not a wrong number, it is a
# number that cannot mean what its name says.

#: (pattern, low, high, level, why) — bounds inclusive, None meaning unbounded.
_BOUNDS: list[tuple[re.Pattern, float | None, float | None, str, str]] = [
    (re.compile(r"^(hazard|hazard_rate|lambda|intensity|default_intensity)$"),
     0.0, None, ERROR, "a default intensity is a rate of arrival and cannot be negative"),
    (re.compile(r"^(survival|survival_prob|survival_probability|s_t)$"),
     0.0, 1.0, ERROR, "a survival probability is a probability"),
    (re.compile(r"^(recovery|recovery_rate)$"),
     0.0, 1.0, ERROR, "a recovery rate is a fraction of face value"),
    (re.compile(r"^(cva|dva|fva)$"),
     0.0, None, ERROR, "a valuation adjustment is an expected loss and cannot be negative"),
    (re.compile(r"^(ic|information_coefficient|mean_ic|avg_ic)$"),
     -1.0, 1.0, ERROR, "an information coefficient is a rank correlation"),
    (re.compile(r"^(turnover|avg_turnover|turnover_.*)$"),
     0.0, None, ERROR, "turnover is a sum of absolute weight changes"),
    (re.compile(r"^(effective_spread|quoted_spread|spread_bps|half_spread)$"),
     0.0, None, ERROR, "a transaction cost cannot be negative"),
    (re.compile(r"^(n_.*|num_.*|.*_count|count)$"),
     0.0, None, ERROR, "a count cannot be negative"),
    (re.compile(r"^(default_prob|pd|prob_default|default_probability)$"),
     0.0, 1.0, ERROR, "a default probability is a probability"),
    # A negative standard deviation is not a convention, it is a sign error or a square root taken
    # of a negative variance. The `_pct` names are excluded because those are quoted out of 100.
    (re.compile(r"^(sigma|sigma_(?!.*pct).*|.*_sigma|local_vol|implied_vol.*|"
                r"(annualised|annualized|conditional|unconditional)_vol)$"),
     0.0, None, ERROR, "a volatility is a standard deviation and cannot be negative"),
    # A Sharpe outside this range is not impossible, but at this magnitude it nearly always means
    # the annualisation was applied twice or the denominator is a variance rather than a deviation.
    (re.compile(r"^(sharpe|sharpe_ratio|annualised_sharpe|annualized_sharpe)$"),
     -20.0, 20.0, WARNING, "a Sharpe of this size usually means double annualisation"),
    # Sentiment has no bound we can assert. [-1,1] and [0,1] are the common conventions, so a
    # score outside them is worth noting — but a task is free to define its own scale, and one in
    # the public roster does exactly that (`tone_score = 1000 * (hawkish - dovish) / tokens`),
    # which is why there is no error tier here however large the number gets.
    (re.compile(r"^(sentiment|sentiment_score|.*_sentiment|tone|tone_score)$"),
     -1.0, 1.0, WARNING, "sentiment is normally declared on [-1, 1] or [0, 1]; check the "
                         "instruction for the scale this task defines"),
]


def _bound_findings(name: str, lo: float, hi: float) -> list[Violation]:
    """Bounds violated by a quantity whose observed range is [lo, hi]."""
    out: list[Violation] = []
    key = _norm(name)
    for pattern, low, high, level, why in _BOUNDS:
        if not pattern.match(key):
            continue
        if low is not None and lo < low - 1e-12:
            out.append((level, "invariant_bound",
                        f"'{name}' reaches {lo:.6g}, below the floor of {low:g} — {why}"))
        if high is not None and hi > high + 1e-12:
            out.append((level, "invariant_bound",
                        f"'{name}' reaches {hi:.6g}, above the ceiling of {high:g} — {why}"))
    return out


# ---------------------------------------------------------------------------
# unit conventions
# ---------------------------------------------------------------------------

def _discount_candidates(rate, maturity):
    """Every plausible reading of (rate, maturity) as a discount factor exp(-rT).

    A rate may be quoted as 0.05 or as 5.0; a maturity may be in years, calendar days or trading
    days. Four of these six combinations are usually nonsense for a given task — but which four is
    not knowable from a column name, so a check is run against all of them and fires only when
    every reading is violated.
    """
    import numpy as np
    for rate_scale, rate_label in ((1.0, "r as given"), (0.01, "r/100")):
        for time_scale, time_label in ((1.0, "T in years"), (1 / 365.0, "T/365"),
                                       (1 / 252.0, "T/252")):
            yield (np.exp(-rate * rate_scale * maturity * time_scale),
                   f"{rate_label}, {time_label}")


def _tolerance(scale) -> float:
    """Absolute tolerance for a relationship at a given price scale.

    1e-3 is the figure the category reference quotes for put-call parity, but it is an absolute USD
    tolerance and a task may price in units where that is unreasonably tight (a notional in
    millions) or needlessly loose (a rate in decimals). Scaling with the data keeps the check
    meaningful at both ends; the floor keeps it from becoming vacuous near zero.
    """
    return max(1e-3, 1e-5 * float(scale))


# ---------------------------------------------------------------------------
# relational checks on a table
# ---------------------------------------------------------------------------

def _col(df, name):
    """A column as float64, with non-numeric entries becoming NaN."""
    import pandas as pd
    return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype="float64")


def _fraction_bad(mask) -> float:
    import numpy as np
    finite = np.isfinite(mask.astype("float64"))
    n = int(finite.sum())
    return float(np.nansum(mask)) / n if n else 0.0


#: A relationship has to fail on more than this share of rows to count. One bad row in a long table
#: is far more likely to be a boundary case the task asked for (a zero maturity, a zero strike) than
#: a broken formula, and the repair loop cannot do anything useful with it.
_SYSTEMATIC = 0.10


def _check_put_call_parity(df, roles) -> list[Violation]:
    """C - P = S - K.exp(-rT), the only exact relationship in option pricing.

    It is model-free: it follows from being able to hold the two options and the stock at once, so
    it holds for Black-Scholes, for a PDE solve, for Monte Carlo, and for the true market price. A
    violation is therefore a genuine bug, not a modelling choice — which is why this is the single
    most valuable check in the module.
    """
    import numpy as np
    need = ("call_price", "put_price", "spot", "strike", "rate", "maturity")
    if not all(k in roles for k in need):
        return []
    c, p = _col(df, roles["call_price"]), _col(df, roles["put_price"])
    s, k = _col(df, roles["spot"]), _col(df, roles["strike"])
    r, t = _col(df, roles["rate"]), _col(df, roles["maturity"])

    scale = np.nanmedian(np.abs(s))
    tol = _tolerance(scale if np.isfinite(scale) else 1.0)
    # A foreign rate or dividend yield discounts the spot leg as well (Garman-Kohlhagen, and the
    # equity formula with a continuous dividend); without one the term is simply absent.
    q = _col(df, roles["rate_foreign"]) if "rate_foreign" in roles else np.zeros_like(s)

    best = None
    for discount, label in _discount_candidates(r, t):
        for q_disc, _ in _discount_candidates(q, t):
            resid = np.abs((c - p) - (s * q_disc - k * discount))
            bad = _fraction_bad(resid > tol)
            worst = float(np.nanmax(resid)) if np.isfinite(resid).any() else float("nan")
            if best is None or bad < best[0]:
                best = (bad, worst, label)
    if best is None or best[0] <= _SYSTEMATIC:
        return []
    bad, worst, label = best
    return [(ERROR, "invariant_put_call_parity",
             f"put-call parity C - P = S - K*exp(-rT) fails on {bad:.0%} of rows "
             f"(largest residual {worst:.6g}, best unit reading: {label}). This identity is "
             f"model-free, so one of the prices, the discounting, or the sign convention is wrong.")]


def _check_price_bounds(df, roles) -> list[Violation]:
    """A call is worth at least its discounted intrinsic value and never more than the spot.

    Both bounds are arbitrage bounds rather than model outputs: below the lower one you could buy
    the option and replicate the payoff for less than it costs; above the upper one you would rather
    own the stock. The put bounds are the mirror image.
    """
    import numpy as np
    out: list[Violation] = []
    if not all(k in roles for k in ("spot", "strike", "rate", "maturity")):
        return out
    s, k = _col(df, roles["spot"]), _col(df, roles["strike"])
    r, t = _col(df, roles["rate"]), _col(df, roles["maturity"])
    scale = np.nanmedian(np.abs(s))
    tol = _tolerance(scale if np.isfinite(scale) else 1.0)

    if "call_price" in roles:
        c = _col(df, roles["call_price"])
        below = min(_fraction_bad(c < np.maximum(0.0, s - k * d) - tol)
                    for d, _ in _discount_candidates(r, t))
        if below > _SYSTEMATIC:
            out.append((ERROR, "invariant_price_bounds",
                        f"'{roles['call_price']}' is below max(0, S - K*exp(-rT)) on "
                        f"{below:.0%} of rows under every rate/time convention — a call cannot be "
                        f"worth less than its discounted intrinsic value."))
        above = _fraction_bad(c > s + tol)
        if above > _SYSTEMATIC:
            out.append((ERROR, "invariant_price_bounds",
                        f"'{roles['call_price']}' exceeds the spot on {above:.0%} of rows — a call "
                        f"can never be worth more than the share it is a right to buy."))
    if "put_price" in roles:
        p = _col(df, roles["put_price"])
        below = min(_fraction_bad(p < np.maximum(0.0, k * d - s) - tol)
                    for d, _ in _discount_candidates(r, t))
        if below > _SYSTEMATIC:
            out.append((ERROR, "invariant_price_bounds",
                        f"'{roles['put_price']}' is below max(0, K*exp(-rT) - S) on {below:.0%} of "
                        f"rows under every convention — below the put's discounted intrinsic."))
    return out


def _check_greek_signs(df, roles, category: str) -> list[Violation]:
    """Delta bounds, and the sign of gamma and vega.

    The delta bounds are hard for a vanilla option: a call cannot move more than a dollar for a
    dollar on the stock, or buying the stock would be cheaper. Gamma and vega are positive for any
    long vanilla European position, but a *spread*, a short position or a portfolio column
    legitimately goes the other way — so those are errors only in the two categories whose
    deliverable is a per-option pricing table, and warnings everywhere else.
    """
    import numpy as np
    out: list[Violation] = []
    eps = 1e-9
    # Resolved rather than compared literally: a card saying `derivatives_pricing`, `pricing` or
    # `fx-pricing` is the same kind of task as one saying `derivatives-pricing`.
    canon = canonical_categories(category)
    prices_options = bool({"derivatives-pricing", "fx"} & set(canon))

    for role, lo, hi in (("call_delta", 0.0, 1.0), ("put_delta", -1.0, 0.0)):
        if role not in roles:
            continue
        v = _col(df, roles[role])
        vmin, vmax = float(np.nanmin(v)), float(np.nanmax(v))
        if vmin < lo - eps or vmax > hi + eps:
            # A delta quoted in percent (25 for a 25-delta option) is a convention, not an error,
            # and saying so stops the repair loop rewriting the pricer instead of the formatting.
            # The threshold is 5 rather than 1 because the two cases look different: a percent
            # column runs out to 50 or 90, whereas a delta of 1.4 is not any convention's version
            # of a number bounded by one — it is a bug.
            pct = 5.0 <= max(abs(vmin), abs(vmax)) <= 100.0
            out.append((WARNING if pct else ERROR, "invariant_delta_bounds",
                        f"'{roles[role]}' spans [{vmin:.6g}, {vmax:.6g}], outside ({lo:g}, {hi:g})"
                        + (" — looks like a percent convention rather than a decimal."
                           if pct else " — outside the no-arbitrage bound for a vanilla option.")))

    for role, why in (("gamma", "value is convex in the spot for a long vanilla option"),
                      ("vega", "more volatility can only add value to an option")):
        if role not in roles:
            continue
        v = _col(df, roles[role])
        bad = _fraction_bad(v < -eps)
        if bad > _SYSTEMATIC:
            out.append((ERROR if prices_options else WARNING, f"invariant_{role}_sign",
                        f"'{roles[role]}' is negative on {bad:.0%} of rows — {why}"
                        + ("." if prices_options else
                           " (acceptable for a spread or a short position, so verify the intent).")))
    return out


def _check_bid_ask(df, roles) -> list[Violation]:
    """bid <= ask. A crossed book is a free lunch: buy at the ask, sell at the higher bid.

    Real deliverables rarely have columns called plainly `bid` and `ask`; the public roster writes
    `call_bid`/`call_ask` and `bid_price_btc`/`ask_price_btc`. So rather than resolving two fixed
    roles, this pairs up any two columns that are the same name with "bid" and "ask" swapped, which
    finds those without matching a `bid` against an unrelated `ask`.
    """
    out: list[Violation] = []
    names = {_norm(c): c for c in df.columns}
    for key, col in names.items():
        if "bid" not in key:
            continue
        twin = names.get(key.replace("bid", "ask")) or names.get(key.replace("bid", "offer"))
        if twin is None:
            continue
        bad = _fraction_bad(_col(df, col) > _col(df, twin) + 1e-12)
        if bad > 0:
            out.append((ERROR, "invariant_crossed_book",
                        f"'{col}' exceeds '{twin}' on {bad:.0%} of rows — a crossed book is an "
                        f"arbitrage and cannot be an output."))
    return out


def _check_vwap_range(df, roles) -> list[Violation]:
    """An average of executed prices lies between the day's low and its high."""
    if not all(k in roles for k in ("vwap", "high", "low")):
        return []
    v, hi, lo = (_col(df, roles["vwap"]), _col(df, roles["high"]), _col(df, roles["low"]))
    bad = _fraction_bad((v > hi + 1e-9) | (v < lo - 1e-9))
    if bad <= _SYSTEMATIC:
        return []
    return [(ERROR, "invariant_vwap_range",
             f"'{roles['vwap']}' falls outside [low, high] on {bad:.0%} of rows — an average of "
             f"traded prices cannot sit outside the range of traded prices.")]


def _check_monotone_in_time(df, roles) -> list[Violation]:
    """Discount factors and survival probabilities fall as the horizon lengthens.

    Both are the present value of something certain only if nothing has happened yet: a dollar later
    is worth less than a dollar sooner, and more time means more chances to default. Checked after
    sorting by the maturity column, since the rows need not arrive in order.
    """
    import numpy as np
    out: list[Violation] = []
    time_col = roles.get("maturity")
    if time_col is None:
        return out
    t = _col(df, time_col)
    order = np.argsort(t, kind="stable")
    for role, label, why in (
            ("discount", "discount factor", "a dollar received later is worth less than one "
                                            "received sooner"),
            ("survival", "survival probability", "more elapsed time means more chances to "
                                                 "default, so survival can only fall")):
        if role not in roles:
            continue
        v = _col(df, roles[role])[order]
        ok = np.isfinite(v)
        v = v[ok]
        if v.size < 2:
            continue
        rises = np.diff(v) > 1e-9
        if rises.any():
            out.append((ERROR, f"invariant_{role}_monotone",
                        f"'{roles[role]}' increases with '{time_col}' at "
                        f"{int(rises.sum())} of {rises.size} steps — {why}."))
    return out


def _check_dv01_consistency(df, roles) -> list[Violation]:
    """DV01 = modified duration x price x 1e-4, up to the notional it is quoted on.

    This is an algebraic identity, so a mismatch is a bug — except that DV01 is routinely quoted per
    100 of face, per million of notional, or in basis points, and modified duration never is. A
    clean power-of-ten ratio is therefore a quoting convention and is left alone; anything else is
    reported, as a warning, because the convention space is wide enough that we may not have
    thought of the one in use.
    """
    import numpy as np
    if not all(k in roles for k in ("dv01", "mod_duration", "price")):
        return []
    implied = np.abs(_col(df, roles["mod_duration"]) * _col(df, roles["price"]) * 1e-4)
    actual = np.abs(_col(df, roles["dv01"]))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(implied > 0, actual / implied, np.nan)
    med = float(np.nanmedian(ratio)) if np.isfinite(ratio).any() else float("nan")
    if not np.isfinite(med) or med <= 0:
        return []
    decades = math.log10(med)
    if abs(decades - round(decades)) < 0.01:          # a notional convention, not an error
        return []
    return [(WARNING, "invariant_dv01_consistency",
             f"'{roles['dv01']}' is {med:.4g}x modified duration x price x 1e-4, which is not a "
             f"notional rescaling — check the duration, the compounding, or the basis-point step.")]


def _check_covered_interest_parity(df, roles) -> list[Violation]:
    """F = S.exp((r_d - r_f).T). Violating it is borrow-here-invest-there money.

    A warning rather than an error: the identity is exact under continuous compounding, and FX
    forwards are also quoted on simple or discrete compounding and in forward *points* rather than
    outright rates, which a column name does not distinguish.
    """
    import numpy as np
    need = ("forward", "spot", "rate", "rate_foreign", "maturity")
    if not all(k in roles for k in need):
        return []
    f, s = _col(df, roles["forward"]), _col(df, roles["spot"])
    rd, rf = _col(df, roles["rate"]), _col(df, roles["rate_foreign"])
    t = _col(df, roles["maturity"])
    tol = _tolerance(np.nanmedian(np.abs(s)))
    best = 1.0
    for rate_scale in (1.0, 0.01):
        for time_scale in (1.0, 1 / 365.0, 1 / 252.0):
            expected = s * np.exp((rd - rf) * rate_scale * t * time_scale)
            best = min(best, _fraction_bad(np.abs(f - expected) > tol))
    if best <= _SYSTEMATIC:
        return []
    return [(WARNING, "invariant_covered_interest_parity",
             f"'{roles['forward']}' departs from S*exp((r_d - r_f)*T) on {best:.0%} of rows under "
             f"every rate and day-count reading — check which currency is the base.")]


def _check_var_es_ordering(df_like) -> list[Violation]:
    """Expected shortfall is at least VaR, and VaR grows with confidence.

    ES is the mean of the losses beyond the VaR threshold, so it cannot be smaller than the
    threshold itself — an average of things each at least x is at least x. Both are compared in
    absolute value because the sign convention (loss positive, or P&L negative) is not knowable
    from a column name and is irrelevant to the ordering.
    """
    import numpy as np
    out: list[Violation] = []
    # A column named `var_rel_error` or `es_rmse` is a diagnostic ABOUT the risk number, not the
    # risk number itself, and nothing requires one to exceed the other. Measured on real sweep
    # output, this was the only false alarm the module produced, so the qualifier is excluded here
    # rather than reasoned away.
    qualifier = re.compile(r"_(err|error|rel|relative|diff|ratio|resid(ual)?|rmse|mae|mape|bias|"
                           r"se|std|stderr|t_?stat|p_?value|z|pct|change)(_|$)")
    cols = {_norm(c): c for c in df_like if not qualifier.search(_norm(c) + "_")}

    def _conf(key: str) -> str | None:
        m = re.search(r"(\d{2,5})", key)
        if not m:
            return None
        raw = m.group(1)
        return raw.rstrip("0") or raw        # 95, 950, 9500 all mean the same confidence level

    var_cols = {k: v for k, v in cols.items() if re.match(r"^(var|value_at_risk)(_|$)", k)}
    es_cols = {k: v for k, v in cols.items()
               if re.match(r"^(es|cvar|expected_shortfall|c_var)(_|$)", k)}

    # ES >= VaR, matched by confidence level where both name one, otherwise pairwise when each
    # side has exactly one column (any other arrangement is too ambiguous to pair safely).
    pairs: list[tuple[str, str]] = []
    for vk, vc in var_cols.items():
        for ek, ec in es_cols.items():
            if _conf(vk) and _conf(vk) == _conf(ek):
                pairs.append((vc, ec))
    if not pairs and len(var_cols) == 1 and len(es_cols) == 1:
        pairs = [(next(iter(var_cols.values())), next(iter(es_cols.values())))]

    for vc, ec in pairs:
        v, e = np.abs(_col(df_like, vc)), np.abs(_col(df_like, ec))
        bad = _fraction_bad(e < v - 1e-9)
        if bad > _SYSTEMATIC:
            out.append((ERROR, "invariant_es_ge_var",
                        f"|{ec}| is below |{vc}| on {bad:.0%} of rows — expected shortfall is the "
                        f"mean of the losses beyond VaR and cannot be smaller than VaR itself. A "
                        f"common cause is averaging all losses instead of only the tail."))

    # VaR non-decreasing in confidence: 99% is a more conservative threshold than 95%.
    levelled = sorted(((float(_conf(k)) if len(_conf(k)) > 1 else float(_conf(k)) * 10, v)
                       for k, v in var_cols.items() if _conf(k)), key=lambda kv: kv[0])
    for (c1, n1), (c2, n2) in zip(levelled, levelled[1:]):
        if c2 <= c1:
            continue
        m1 = float(np.nanmean(np.abs(_col(df_like, n1))))
        m2 = float(np.nanmean(np.abs(_col(df_like, n2))))
        if np.isfinite(m1) and np.isfinite(m2) and m2 < m1 - 1e-9:
            out.append((ERROR, "invariant_var_monotone",
                        f"|{n2}| ({m2:.6g}) is below |{n1}| ({m1:.6g}) — a higher confidence level "
                        f"is a more conservative threshold and cannot give a smaller loss."))
    return out


def _check_risk_monotone_in_confidence(df) -> list[Violation]:
    """A risk number quoted per confidence level must move in one direction as that level changes.

    Which direction is not knowable: a column called `alpha` may hold 0.01 (a tail probability, so
    the loss grows as it falls) or 0.99 (a confidence, so the loss grows as it rises). But it must
    be one or the other — a VaR that rises from 95% to 97.5% and then falls at 99% is not a
    convention, it is a broken quantile. Checking monotonicity without caring about the sign of the
    slope sidesteps the ambiguity entirely.
    """
    import numpy as np
    out: list[Violation] = []
    names = {_norm(c): c for c in df.columns}
    level_col = next((v for k, v in names.items()
                      if k in ("confidence_level", "confidence", "alpha", "quantile",
                               "conf_level", "level")), None)
    if level_col is None:
        return out
    risk_cols = [v for k, v in names.items()
                 if re.match(r"^(var|es|cvar|expected_shortfall|value_at_risk)(_|$)", k)
                 or k.endswith("_var") or k.endswith("_cvar") or k.endswith("_es")]
    if not risk_cols:
        return out

    levels = _col(df, level_col)
    # Several rows can share a level (one per method or copula); compare the mean at each level,
    # since a per-method comparison would be between different models rather than along one curve.
    for col in risk_cols:
        values = np.abs(_col(df, col))
        frame = {}
        for lv, val in zip(levels, values):
            if np.isfinite(lv) and np.isfinite(val):
                frame.setdefault(lv, []).append(val)
        if len(frame) < 3:                     # two points are monotone by definition
            continue
        series = [float(np.mean(frame[k])) for k in sorted(frame)]
        steps = np.diff(series)
        if (steps > 1e-12).any() and (steps < -1e-12).any():
            out.append((ERROR, "invariant_risk_monotone",
                        f"'{col}' is not monotone in '{level_col}' "
                        f"({', '.join(f'{v:.6g}' for v in series)}) — a quantile of one loss "
                        f"distribution must move in a single direction as the level changes."))
    return out


def _check_stochastic_matrix(df) -> list[Violation]:
    """The rows of a transition matrix are probability distributions and sum to one.

    Recognised rather than assumed: the table must be all probabilities, with three or more
    numeric columns, and its typical row must already sum to one. A table of cumulative default
    probabilities by horizon is also all probabilities but its rows sum to much more than one, so
    it is left alone — which is the point of testing the typical row before judging any row.
    """
    import numpy as np
    num = df.select_dtypes(include="number")
    if num.shape[1] < 3 or num.shape[0] < 2:
        return []
    m = num.to_numpy(dtype="float64")
    if not np.isfinite(m).all() or m.min() < -1e-9 or m.max() > 1.0 + 1e-9:
        return []
    sums = m.sum(axis=1)
    if abs(float(np.median(sums)) - 1.0) > 0.02:
        return []                               # not a row-stochastic matrix; nothing to say
    off = np.abs(sums - 1.0) > 1e-4
    if not off.any():
        return []
    return [(ERROR, "invariant_row_stochastic",
             f"{int(off.sum())} of {off.size} rows of this transition matrix sum to "
             f"{sums[off][0]:.6g} rather than 1 — every row is a distribution over where a name "
             f"can end up, including staying put and defaulting.")]


def _check_self_reported_residual(df) -> list[Violation]:
    """A residual the task asked us to report, which is large.

    Several units ask for the thing to be verified alongside the answer: a `parity_error`, a
    `pde_residual`. Those columns are the task's own statement of its invariant, so a large value
    is the output confessing. "Large" is judged against the prices in the same table rather than
    absolutely, and only an order-of-magnitude failure is reported, because the tolerance the task
    had in mind is not written in the column name.
    """
    import numpy as np
    out: list[Violation] = []
    resid = re.compile(r"(parity_error|parity_diff|pde_residual|replication_error|"
                       r"arbitrage_violation|parity_residual)")
    price_like = re.compile(r"(price|premium|value|npv)")
    num = df.select_dtypes(include="number")
    with np.errstate(all="ignore"), warnings.catch_warnings():
        # A wholly blank price column makes nanmedian warn; it is a value we then discard, and the
        # warning would otherwise surface in test output as if the data had caused it.
        warnings.simplefilter("ignore", RuntimeWarning)
        scales = [float(np.nanmedian(np.abs(num[c]))) for c in num.columns
                  if price_like.search(_norm(c)) and not resid.search(_norm(c))]
    scale = next((v for v in scales if np.isfinite(v) and v > 0), None)
    if scale is None:
        return out
    for col in num.columns:
        if not resid.search(_norm(col)):
            continue
        worst = float(np.nanmax(np.abs(num[col].to_numpy(dtype="float64"))))
        if np.isfinite(worst) and worst > 0.01 * scale:
            out.append((WARNING, "invariant_self_reported_residual",
                        f"'{col}' reaches {worst:.6g}, which is {worst / scale:.1%} of the price "
                        f"scale in this file — the task asked for this residual as a check on its "
                        f"own answer, and at this size it is reporting a failure."))
    return out


def _check_correlation_matrix(df) -> list[Violation]:
    """A correlation matrix is symmetric, unit-diagonal and positive semi-definite.

    PSD is the one that is easy to break and impossible to see by eye: a matrix assembled
    pairwise, or patched by hand, can imply a portfolio with negative variance. Only run when the
    file actually looks like a square matrix with ones down the diagonal, so an ordinary table with
    a `corr` column is never dragged in here.
    """
    import numpy as np
    num = df.select_dtypes(include="number")
    n = num.shape[1]
    if n < 2 or num.shape[0] != n:
        return []
    m = num.to_numpy(dtype="float64")
    if not np.isfinite(m).all() or not np.allclose(np.diag(m), 1.0, atol=1e-6):
        return []
    out: list[Violation] = []
    if not np.allclose(m, m.T, atol=1e-6):
        out.append((ERROR, "invariant_corr_symmetry",
                    "the correlation matrix is not symmetric — corr(i, j) must equal corr(j, i)."))
        m = 0.5 * (m + m.T)
    smallest = float(np.min(np.linalg.eigvalsh(m)))
    if smallest < -1e-6:
        out.append((ERROR, "invariant_corr_psd",
                    f"the correlation matrix is not positive semi-definite (smallest eigenvalue "
                    f"{smallest:.3g}) — it implies a portfolio with negative variance. Project it "
                    f"onto the PSD cone, or build it from a return matrix rather than pairwise."))
    return out


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------

def check_frame(df, category: str = "") -> list[Violation]:
    """Every applicable invariant for one tabular deliverable.

    `category` is the unit card's `[metadata] category`. It is used only to decide how loud a
    finding should be, never to skip a check: a task labelled `backtesting` that happens to emit a
    column of call prices still gets its parity checked, because the identity holds either way.
    """
    import importlib.util
    if importlib.util.find_spec("numpy") is None:
        return []                 # every check below needs it; an unverifiable file is not a bad one
    out: list[Violation] = []
    columns = list(df.columns)
    roles = _roles_of(columns)

    for check in (lambda: _check_put_call_parity(df, roles),
                  lambda: _check_price_bounds(df, roles),
                  lambda: _check_greek_signs(df, roles, category),
                  lambda: _check_bid_ask(df, roles),
                  lambda: _check_vwap_range(df, roles),
                  lambda: _check_monotone_in_time(df, roles),
                  lambda: _check_dv01_consistency(df, roles),
                  lambda: _check_covered_interest_parity(df, roles),
                  lambda: _check_var_es_ordering(df),
                  lambda: _check_risk_monotone_in_confidence(df),
                  lambda: _check_stochastic_matrix(df),
                  lambda: _check_self_reported_residual(df),
                  lambda: _check_correlation_matrix(df)):
        try:
            out.extend(check())
        except Exception:
            continue          # an invariant that cannot run must not become the reason we fail

    num = df.select_dtypes(include="number")
    for col in num.columns:
        series = num[col].dropna()
        if series.empty:
            continue
        out.extend(_bound_findings(str(col), float(series.min()), float(series.max())))
    return _dedupe(out)


def check_json(obj, category: str = "") -> list[Violation]:
    """The same invariants against a JSON deliverable.

    JSON output arrives in two shapes. A list of records is a table in disguise and is handed to
    `check_frame`. Anything else is flattened to its scalar leaves, which supports the bounds and
    the VaR/ES ordering — the row-wise relationships need aligned columns and are skipped.
    """
    out: list[Violation] = []
    if isinstance(obj, list) and obj and all(isinstance(r, dict) for r in obj):
        try:
            import pandas as pd
            return check_frame(pd.DataFrame(obj), category)
        except ImportError:
            return []

    flat: dict[str, float] = {}

    def _walk(node, prefix: str = "") -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                _walk(value, str(key))               # the leaf's own key names the quantity
        elif isinstance(node, list):
            vals = [v for v in node if isinstance(v, (int, float)) and not isinstance(v, bool)]
            if vals and prefix:
                out.extend(_bound_findings(prefix, float(min(vals)), float(max(vals))))
            for v in node:
                if isinstance(v, (dict, list)):
                    _walk(v, prefix)
        elif isinstance(node, (int, float)) and not isinstance(node, bool) and prefix:
            flat.setdefault(prefix, float(node))
            out.extend(_bound_findings(prefix, float(node), float(node)))

    _walk(obj)

    if flat:
        try:
            import pandas as pd
            out.extend(_check_var_es_ordering(pd.DataFrame([flat])))
        except ImportError:
            pass
    return _dedupe(out)


def _dedupe(found: list[Violation]) -> list[Violation]:
    """One finding per (code, message). Nested JSON in particular repeats the same leaf name."""
    seen: set[tuple[str, str]] = set()
    out: list[Violation] = []
    for level, code, message in found:
        if (code, message) in seen:
            continue
        seen.add((code, message))
        out.append((level, code, message))
    return out


# ---------------------------------------------------------------------------
# the same knowledge, pointed the other way: guidance for the prompt
# ---------------------------------------------------------------------------
#
# Checking invariants after the fact can only trigger a rewrite, which costs an attempt and a slice
# of a 25-request budget. Telling the model the invariants up front costs nothing and prevents the
# same errors — and the conventions below (theta per day, sigma as a decimal, excess returns in a
# Sharpe) are mistakes of unit and definition that a capable model makes only because nobody said
# which convention the grader expects.
#
# Keyed by the unit card's `[metadata] category`. The ten ids are fixed by the competition.

_SHARED = (
    "- State units and conventions explicitly in the code, and follow the instruction where it "
    "declares one: a right answer in the wrong unit scores the same as a wrong answer.\n"
    "- Never use data timestamped at or after the moment a decision is made. Look-ahead bias is "
    "checked for directly."
)

CATEGORY_GUIDANCE: dict[str, str] = {
    "derivatives-pricing": (
        "- Put-call parity must hold exactly: C - P = S - K*exp(-r*T). It is model-free, so it "
        "holds whatever method you use.\n"
        "- Call delta lies in (0, 1); put delta in (-1, 0); gamma and vega are positive for long "
        "vanilla European options.\n"
        "- A call is worth at least max(0, S - K*exp(-r*T)) and never more than S; a put at least "
        "max(0, K*exp(-r*T) - S).\n"
        "- Option value is non-decreasing in maturity at a fixed strike, and convex in the strike.\n"
        "- Report theta per day only if asked, and divide the per-year figure by 365 when you do.\n"
        "- sigma and r are decimals (0.20, not 20) unless the instruction says otherwise."
    ),
    "fixed-income": (
        "- Discount factors start at DF(0) = 1 and strictly decrease with maturity.\n"
        "- Price falls as yield rises; a bond whose coupon equals its yield prices at par.\n"
        "- DV01 = modified duration * price * 1e-4. Keep one notional convention throughout and "
        "say which it is.\n"
        "- A par swap has zero NPV at inception by definition.\n"
        "- Be explicit about compounding frequency and day count, and about clean versus dirty "
        "price — most errors in this category are convention errors, not algebra."
    ),
    "credit": (
        "- Survival probability starts at 1 and is monotonically decreasing in time; "
        "S(T) = exp(-integral of lambda).\n"
        "- Hazard rates are strictly positive; recovery rates lie in [0, 1] (40% is the standard "
        "assumption when none is given).\n"
        "- At the par spread the premium leg and the protection leg have equal present value.\n"
        "- Premium-leg cash flows are conditional on survival to each payment date.\n"
        "- CVA is an expected loss and is never negative."
    ),
    "factor-research": (
        "- Rank or standardise the signal cross-sectionally within each date before forming "
        "portfolios; never sort on the raw level across dates.\n"
        "- A long-short portfolio is dollar-neutral: the weights sum to zero.\n"
        "- The information coefficient is a rank (Spearman) correlation and lies in [-1, 1].\n"
        "- Sharpe uses excess returns (r - rf) in both numerator and denominator, and annualises "
        "by sqrt(periods per year), not by the number of periods.\n"
        "- The signal date must strictly precede the return window it predicts.\n"
        "- Turnover is a sum of absolute weight changes and is non-negative."
    ),
    "backtesting": (
        "- The portfolio is self-financing: V_n = V_(n-1) * (1 + r_n), and the terminal value "
        "equals the product of gross returns times the initial value.\n"
        "- Trade on information available strictly before the trade timestamp; a signal from a "
        "close cannot be executed at that same close.\n"
        "- Apply the stated transaction costs on traded notional, not on held notional.\n"
        "- Maximum drawdown is the largest peak-to-trough decline, so it is at least as large in "
        "magnitude as the worst single period.\n"
        "- Sharpe uses excess returns and annualises by sqrt(periods per year).\n"
        "- Fit parameters only on data preceding the period you evaluate them on."
    ),
    "risk-management": (
        "- Expected shortfall (CVaR) at a given confidence is at least the VaR at that same "
        "confidence: it is the mean of the losses BEYOND the threshold, not of all losses.\n"
        "- VaR is non-decreasing in confidence: the 99% figure is at least the 95% figure.\n"
        "- State the sign convention (loss positive or P&L negative) and hold to it in every "
        "column.\n"
        "- Scale a one-day figure to h days by sqrt(h) under a normal model; never by h.\n"
        "- Be consistent about simple versus log returns throughout.\n"
        "- A covariance or correlation matrix must be symmetric and positive semi-definite."
    ),
    "microstructure": (
        "- The book is never crossed: bid <= ask at every timestamp.\n"
        "- Spreads and transaction costs are non-negative.\n"
        "- A liquidation schedule never re-buys: every scheduled quantity has the same sign, and "
        "the schedule sums exactly to the target position.\n"
        "- Market impact works against the aggressor: buying lifts the price, selling depresses "
        "it.\n"
        "- An average execution price lies within the period's low-high range.\n"
        "- Distinguish the quoted spread from the effective spread (twice the signed distance from "
        "the midpoint), and count the time grid carefully — N intervals, not N+1."
    ),
    "fx": (
        "- Covered interest parity: F = S * exp((r_domestic - r_foreign) * T).\n"
        "- A currency triangle is consistent: the product of the three rates around the cycle is "
        "1, up to the bid-ask spread.\n"
        "- Garman-Kohlhagen is Black-Scholes with the foreign rate in the role of a dividend yield; "
        "the delta carries the exp(-r_f*T) factor and put-call parity holds with q = r_foreign.\n"
        "- Be explicit about which currency is the base: EUR/USD = 1.08 means one euro buys 1.08 "
        "dollars, and inverting it flips every sign downstream."
    ),
    "nlp-on-finance": (
        "- Keep scores inside the range the instruction declares, and state the convention.\n"
        "- Draw classification labels only from the declared label set; never invent a category.\n"
        "- The document date must precede the return window in any event study.\n"
        "- Counts are non-negative integers.\n"
        "- Strip boilerplate (forward-looking-statement disclaimers appear in nearly every "
        "transcript) and respect the speaker role where the data distinguishes one.\n"
        "- Preserve the magnitude of parsed figures: thousands, millions and billions are easy to "
        "confuse and the error is silent."
    ),
    "cross-domain": (
        "- This task spans more than one area of finance. Solve each part on its own terms and "
        "satisfy the invariants of EVERY area it touches, not just the one you find most "
        "familiar — partial solutions score nothing.\n"
        "- Pass data between the parts in one consistent set of units: a yield emitted in basis "
        "points and consumed as a decimal is the classic failure here.\n"
        "- Before aggregating risk across asset classes, convert everything to a common risk "
        "factor and basis."
    ),
}


# The ten ids above are what the category reference documents. The cards in the field do not keep
# to them: only 53 of the 87 public units carry a canonical id, and the other 34 write things like
# `derivatives_pricing`, `credit-risk`, `risk-modeling`, `extreme-value-theory`, `fx-pricing` or
# `fixed-income-nlp`. A plain dictionary lookup therefore silently misses a third of the roster,
# which is why these keywords exist. They are ordered deliberately: the first match wins for a
# single-topic card, and a card naming two areas (`fixed-income-nlp`) gets both blocks.
_KEYWORDS: list[tuple[str, str]] = [
    (r"nlp|text|tone|sentiment|transcript|filing|news|event[-_ ]driven", "nlp-on-finance"),
    (r"credit|cds|default|issuer|hazard", "credit"),
    (r"fx|currency|forex|cross[-_ ]currency", "fx"),
    (r"fixed[-_ ]income|rates?\b|curve|bond|swap|interest[-_ ]rate", "fixed-income"),
    (r"microstructure|execution|order[-_ ]book|liquidity|tick", "microstructure"),
    (r"backtest|strategy|trading|cross[-_ ]sectional[-_ ]strateg", "backtesting"),
    (r"factor|alpha|attribution|predictive", "factor-research"),
    (r"risk[-_ ](manage|model|analy|manag)|market[-_ ]risk|var\b|cvar|value[-_ ]at[-_ ]risk|extreme[-_ ]value|dependence|copula|stress|basel|capital", "risk-management"),
    (r"derivativ|pricing|option|volatility|vol\b|greeks|stochastic[-_ ]process", 
     "derivatives-pricing"),
    (r"cross[-_ ]domain|cross[-_ ]asset|multi[-_ ]asset", "cross-domain"),
]

#: Beyond this, the prompt is spending more on reminders than on the task. A card naming three
#: areas gets the two strongest matches and the cross-domain warning, which is the useful part.
_MAX_BLOCKS = 2


def canonical_categories(category: str) -> list[str]:
    """The canonical category ids a card's free-text category refers to, most specific first."""
    raw = re.sub(r"[_\s]+", "-", (category or "").strip().lower())
    if raw in CATEGORY_GUIDANCE:
        return [raw]
    matched = [canon for pattern, canon in _KEYWORDS if re.search(pattern, raw)]
    # Deduplicate while keeping order; `_KEYWORDS` order is the priority.
    return list(dict.fromkeys(matched))


def guidance_for(category: str) -> str:
    """The invariant and convention reminders for one card's category.

    An unknown or absent category still gets the shared block: the two warnings in it (declare your
    units, do not look ahead) are the two most common ways a correct method produces a failing
    answer, and neither depends on the domain.
    """
    canon = canonical_categories(category)
    blocks = [CATEGORY_GUIDANCE[c] for c in canon[:_MAX_BLOCKS] if c in CATEGORY_GUIDANCE]
    if len(canon) > 1 and "cross-domain" not in canon[:_MAX_BLOCKS]:
        # Two areas in one card is a cross-domain task whether or not it says so, and the unit
        # conventions between the halves are where those fail.
        blocks.append(CATEGORY_GUIDANCE["cross-domain"])
    return "\n".join(blocks + [_SHARED])
