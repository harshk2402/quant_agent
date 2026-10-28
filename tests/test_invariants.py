# SPDX-License-Identifier: MIT
"""Tests for the classical-finance invariant layer.

Every check gets two tests: one that it fires on output which violates the relationship, and one
that it stays silent on output which satisfies it. The second kind matters more. A missed violation
costs us the chance to repair one task; a false alarm spends an attempt and part of a 25-request
budget rewriting code that was already correct, and the rewrite can be worse than what it replaced.

Several tests therefore feed *correct* answers written under an unusual convention — a rate in
percent, a maturity in days, a delta quoted as 25 rather than 0.25 — and assert that nothing fires,
because the checks are supposed to try every plausible reading before complaining.
"""
from __future__ import annotations

import math
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from agent import invariants as iv  # noqa: E402

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")


def _codes(found):
    return {code for _level, code, _msg in found}


def _levels(found, code):
    return {level for level, c, _msg in found if c == code}


# --- a reference option book, priced correctly ------------------------------------------------

def _bs(s, k, r, t, sigma):
    """Black-Scholes call and put, so the fixtures are genuinely arbitrage-free rather than
    hand-written numbers that happen to look plausible."""
    from statistics import NormalDist
    nd = NormalDist()
    d1 = (math.log(s / k) + (r + 0.5 * sigma ** 2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    call = s * nd.cdf(d1) - k * math.exp(-r * t) * nd.cdf(d2)
    put = k * math.exp(-r * t) * nd.cdf(-d2) - s * nd.cdf(-d1)
    return call, put, nd.cdf(d1), nd.cdf(d1) - 1.0


def _option_book(rate_in_percent=False, maturity_in_days=False):
    rows = []
    for k in (80.0, 95.0, 100.0, 110.0, 130.0):
        for t in (0.25, 1.0, 2.0):
            call, put, dc, dp = _bs(100.0, k, 0.03, t, 0.2)
            rows.append({
                "spot": 100.0, "strike": k,
                "rate": 3.0 if rate_in_percent else 0.03,
                "maturity": t * 365 if maturity_in_days else t,
                "call_price": call, "put_price": put,
                "call_delta": dc, "put_delta": dp,
                "gamma": 0.01, "vega": 20.0,
            })
    return pd.DataFrame(rows)


def test_correct_option_book_is_silent():
    assert iv.check_frame(_option_book(), "derivatives-pricing") == []


@pytest.mark.parametrize("percent,days", [(True, False), (False, True), (True, True)])
def test_correct_option_book_under_other_unit_conventions(percent, days):
    """A right answer with the rate in percent or the maturity in days must not be flagged."""
    found = iv.check_frame(_option_book(percent, days), "derivatives-pricing")
    assert found == [], found


def test_put_call_parity_violation_is_an_error():
    df = _option_book()
    df["put_price"] = df["put_price"] * 1.5          # a plausible-looking but inconsistent put
    assert "invariant_put_call_parity" in _codes(iv.check_frame(df, "derivatives-pricing"))


def test_parity_tolerates_a_single_odd_row():
    """One broken row in fifteen is an edge case, not a broken formula."""
    df = _option_book()
    df.loc[0, "put_price"] = df.loc[0, "put_price"] + 5.0
    assert "invariant_put_call_parity" not in _codes(iv.check_frame(df, "derivatives-pricing"))


def test_call_below_intrinsic_is_an_error():
    df = _option_book().drop(columns=["put_price"])   # no parity check, so bounds must catch it
    df["call_price"] = df["call_price"] * 0.1
    assert "invariant_price_bounds" in _codes(iv.check_frame(df, "derivatives-pricing"))


def test_call_above_spot_is_an_error():
    df = _option_book().drop(columns=["put_price"])
    df["call_price"] = df["spot"] * 1.2
    assert "invariant_price_bounds" in _codes(iv.check_frame(df, "derivatives-pricing"))


def test_delta_out_of_bounds_is_an_error():
    df = _option_book()
    df["call_delta"] = 1.4
    assert _levels(iv.check_frame(df, "derivatives-pricing"),
                   "invariant_delta_bounds") == {iv.ERROR}


def test_delta_in_percent_only_warns():
    """25-delta quoting is a convention. Rewriting the pricer over it would be a waste."""
    df = _option_book()
    df["call_delta"] = df["call_delta"] * 100.0
    assert _levels(iv.check_frame(df, "derivatives-pricing"),
                   "invariant_delta_bounds") == {iv.WARNING}


def test_negative_vega_errors_for_a_pricing_task_and_warns_elsewhere():
    df = _option_book()
    df["vega"] = -20.0
    assert _levels(iv.check_frame(df, "derivatives-pricing"), "invariant_vega_sign") == {iv.ERROR}
    assert _levels(iv.check_frame(df, "backtesting"), "invariant_vega_sign") == {iv.WARNING}


# --- fixed income and credit -------------------------------------------------------------------

def test_discount_factors_must_fall_with_maturity():
    good = pd.DataFrame({"maturity": [1, 2, 5, 10], "discount_factor": [0.97, 0.94, 0.86, 0.74]})
    assert iv.check_frame(good, "fixed-income") == []
    bad = good.copy()
    bad.loc[2, "discount_factor"] = 0.99
    assert "invariant_discount_monotone" in _codes(iv.check_frame(bad, "fixed-income"))


def test_unsorted_rows_are_sorted_before_the_monotonicity_check():
    shuffled = pd.DataFrame({"maturity": [5, 1, 10, 2],
                             "discount_factor": [0.86, 0.97, 0.74, 0.94]})
    assert iv.check_frame(shuffled, "fixed-income") == []


def test_survival_probability_must_fall_and_stay_a_probability():
    bad = pd.DataFrame({"tenor_years": [1, 3, 5], "survival_probability": [0.98, 0.99, 0.93],
                        "hazard_rate": [0.01, 0.01, 0.02]})
    codes = _codes(iv.check_frame(bad, "credit"))
    assert "invariant_survival_monotone" in codes

    out_of_range = pd.DataFrame({"survival_probability": [1.2, 0.9], "hazard_rate": [-0.01, 0.01]})
    msgs = iv.check_frame(out_of_range, "credit")
    assert _codes(msgs) == {"invariant_bound"}
    assert len(msgs) == 2                 # one for the probability, one for the negative intensity


def test_dv01_notional_conventions_are_not_flagged():
    """DV01 per 100 of face is a hundred times the per-unit figure. That is quoting, not a bug."""
    df = pd.DataFrame({"price": [98.5, 101.2], "modified_duration": [7.1, 4.3]})
    df["dv01"] = df["modified_duration"] * df["price"] * 1e-4
    assert iv.check_frame(df, "fixed-income") == []
    scaled = df.assign(dv01=df["dv01"] * 100.0)
    assert iv.check_frame(scaled, "fixed-income") == []
    wrong = df.assign(dv01=df["dv01"] * 2.5)
    assert _levels(iv.check_frame(wrong, "fixed-income"),
                   "invariant_dv01_consistency") == {iv.WARNING}


# --- risk ---------------------------------------------------------------------------------------

def test_expected_shortfall_below_var_is_an_error():
    good = pd.DataFrame({"var_95": [0.021], "es_95": [0.033]})
    assert iv.check_frame(good, "risk-management") == []
    bad = pd.DataFrame({"var_95": [0.033], "es_95": [0.021]})
    assert "invariant_es_ge_var" in _codes(iv.check_frame(bad, "risk-management"))


def test_var_and_es_compare_by_magnitude_whatever_the_sign_convention():
    """Losses are quoted positive by some tasks and negative by others; the ordering is the same."""
    negative = pd.DataFrame({"var_99": [-0.031], "cvar_99": [-0.047]})
    assert iv.check_frame(negative, "risk-management") == []


def test_var_must_grow_with_confidence():
    bad = pd.DataFrame({"var_95": [0.030], "var_99": [0.019]})
    assert "invariant_var_monotone" in _codes(iv.check_frame(bad, "risk-management"))
    good = pd.DataFrame({"var_95": [0.019], "var_99": [0.030]})
    assert iv.check_frame(good, "risk-management") == []


def test_non_psd_correlation_matrix_is_an_error():
    good = pd.DataFrame(np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.2], [0.3, 0.2, 1.0]]),
                        columns=list("abc"))
    assert iv.check_frame(good, "risk-management") == []
    bad = pd.DataFrame(np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]]),
                       columns=list("abc"))
    assert "invariant_corr_psd" in _codes(iv.check_frame(bad, "risk-management"))


def test_an_ordinary_table_is_not_mistaken_for_a_correlation_matrix():
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    assert iv.check_frame(df, "factor-research") == []


# --- microstructure -----------------------------------------------------------------------------

def test_crossed_book_is_an_error():
    df = pd.DataFrame({"bid": [100.1, 100.2], "ask": [100.0, 100.3]})
    assert "invariant_crossed_book" in _codes(iv.check_frame(df, "microstructure"))


def test_vwap_outside_the_days_range_is_an_error():
    df = pd.DataFrame({"low": [99.0] * 4, "high": [101.0] * 4, "vwap": [105.0] * 4})
    assert "invariant_vwap_range" in _codes(iv.check_frame(df, "microstructure"))
    ok = df.assign(vwap=[100.0] * 4)
    assert iv.check_frame(ok, "microstructure") == []


def test_negative_spread_and_turnover_are_errors():
    df = pd.DataFrame({"effective_spread": [-0.01, 0.02], "turnover": [0.4, -0.1]})
    found = iv.check_frame(df, "microstructure")
    assert _codes(found) == {"invariant_bound"} and len(found) == 2


# --- fx ----------------------------------------------------------------------------------------

def test_covered_interest_parity_warns_rather_than_errors():
    s, rd, rf, t = 1.08, 0.04, 0.02, 1.0
    good = pd.DataFrame({"spot": [s], "r_d": [rd], "r_f": [rf], "maturity": [t],
                         "forward": [s * math.exp((rd - rf) * t)]})
    assert iv.check_frame(good, "fx") == []
    bad = good.assign(forward=[s * math.exp((rf - rd) * t)])     # the differential, inverted
    assert _levels(iv.check_frame(bad, "fx"),
                   "invariant_covered_interest_parity") == {iv.WARNING}


# --- nlp ---------------------------------------------------------------------------------------

def test_sentiment_out_of_the_usual_range_only_ever_warns():
    """A task may define its own scale, and one in the public roster does: `t1-fomc-tone-event-
    study` specifies `tone_score = 1000 * (hawkish - dovish) / tokens`. A correct answer there
    reaches 125, so any error tier on this quantity would be a false alarm on a passing unit."""
    for values in ([0.0, 1.4], [0.0, 125.0], [0.0, 5000.0]):
        found = iv.check_frame(pd.DataFrame({"sentiment_score": values}), "nlp-on-finance")
        assert _levels(found, "invariant_bound") == {iv.WARNING}


def test_a_metric_about_var_is_not_mistaken_for_var():
    """`es_rel_error` is a diagnostic, not an expected shortfall; nothing orders the two."""
    df = pd.DataFrame({"var_rel_error": [0.04], "es_rel_error": [0.01]})
    assert iv.check_frame(df, "risk-management") == []


def test_negative_counts_are_an_error():
    df = pd.DataFrame({"risk_factor_count": [3, -1]})
    assert _levels(iv.check_frame(df, "nlp-on-finance"), "invariant_bound") == {iv.ERROR}


# --- role resolution ---------------------------------------------------------------------------

def test_an_ambiguous_role_disables_the_check_rather_than_guessing():
    """Two columns both looking like a strike: we cannot know which the relationship is about."""
    df = _option_book().rename(columns={"strike": "k"})
    df["strike"] = df["k"] * 2.0
    assert iv.check_frame(df, "derivatives-pricing") == []


def test_a_similarly_named_column_does_not_bind_a_role():
    """`strike_delta` is not a strike, and must not be read as one."""
    df = pd.DataFrame({"strike_delta": [0.4, 0.6]})
    assert iv._resolve(df.columns, "strike") is None


# --- json --------------------------------------------------------------------------------------

def test_json_records_are_checked_like_a_table():
    records = [{"bid": 100.1, "ask": 100.0}, {"bid": 99.0, "ask": 99.5}]
    assert "invariant_crossed_book" in _codes(iv.check_json(records, "microstructure"))


def test_nested_json_scalars_are_checked_by_leaf_name():
    doc = {"portfolio": {"metrics": {"var_99": 0.02, "es_99": 0.01},
                         "credit": {"recovery_rate": 1.4}}}
    codes = _codes(iv.check_json(doc, "cross-domain"))
    assert {"invariant_es_ge_var", "invariant_bound"} <= codes


def test_a_clean_json_report_is_silent():
    doc = {"var_99": 0.031, "es_99": 0.047, "sharpe_ratio": 1.2, "turnover": 0.35}
    assert iv.check_json(doc, "risk-management") == []


# --- guidance ----------------------------------------------------------------------------------

def test_every_category_has_guidance_and_an_unknown_one_still_gets_the_shared_block():
    for category in ("derivatives-pricing", "fixed-income", "credit", "factor-research",
                     "backtesting", "risk-management", "microstructure", "fx",
                     "nlp-on-finance", "cross-domain"):
        assert len(iv.guidance_for(category)) > 200
    assert "Look-ahead bias" in iv.guidance_for("")
    assert "Look-ahead bias" in iv.guidance_for("not-a-category")


# --- checks added after measuring the module against real sweep output -------------------------
#
# The first pass of this module raised nothing on 87 units' worth of real deliverables, because its
# checks resolved fixed column names that real tasks do not use. Everything below was added from
# what the public roster actually writes.

def test_bid_ask_pairs_are_found_under_a_prefix_or_a_suffix():
    """The roster writes `call_bid`/`call_ask` and `bid_price_btc`/`ask_price_btc`, never `bid`."""
    prefixed = pd.DataFrame({"call_bid": [1.2, 1.4], "call_ask": [1.1, 1.5]})
    assert "invariant_crossed_book" in _codes(iv.check_frame(prefixed, "derivatives-pricing"))
    suffixed = pd.DataFrame({"bid_price_btc": [0.05], "ask_price_btc": [0.04]})
    assert "invariant_crossed_book" in _codes(iv.check_frame(suffixed, "derivatives-pricing"))
    ok = pd.DataFrame({"call_bid": [1.1], "call_ask": [1.2],
                       "put_bid": [0.8], "put_ask": [0.9]})
    assert iv.check_frame(ok, "derivatives-pricing") == []


def test_an_unrelated_bid_and_ask_are_not_paired():
    """`bid_size` against `ask_price` would be a comparison of a quantity with a price."""
    df = pd.DataFrame({"bid_size": [500.0], "ask_price": [100.0]})
    assert iv.check_frame(df, "microstructure") == []


def test_risk_must_be_monotone_in_the_confidence_column():
    """Long format: one VaR column and a confidence level per row, which is how tasks write it."""
    zigzag = pd.DataFrame({"confidence_level": [0.95, 0.975, 0.99],
                           "VaR": [0.021, 0.040, 0.030]})
    assert "invariant_risk_monotone" in _codes(iv.check_frame(zigzag, "risk-management"))
    rising = pd.DataFrame({"confidence_level": [0.95, 0.975, 0.99],
                           "VaR": [0.021, 0.027, 0.033]})
    assert iv.check_frame(rising, "risk-management") == []


def test_a_tail_probability_convention_is_accepted_as_readily_as_a_confidence():
    """`alpha` of 0.01 means the 99% level, so the loss FALLS as alpha rises. Also monotone."""
    df = pd.DataFrame({"alpha": [0.01, 0.025, 0.05], "var_dollar": [0.033, 0.027, 0.021]})
    assert iv.check_frame(df, "risk-management") == []


def test_several_rows_per_confidence_level_are_averaged_not_compared():
    """One row per method at each level; comparing across methods would be meaningless."""
    df = pd.DataFrame({"method": ["hist", "normal"] * 3,
                       "confidence_level": [0.95, 0.95, 0.975, 0.975, 0.99, 0.99],
                       "var_dollar": [0.020, 0.022, 0.026, 0.028, 0.032, 0.034]})
    assert iv.check_frame(df, "risk-management") == []


def test_transition_matrix_rows_must_sum_to_one():
    good = pd.DataFrame({"rating": ["AAA", "AA", "A"],
                         "AAA": [0.90, 0.05, 0.01], "AA": [0.08, 0.90, 0.05],
                         "A": [0.02, 0.05, 0.94]})
    assert iv.check_frame(good, "credit") == []
    bad = good.copy()
    bad.loc[1, "AA"] = 0.70                       # 0.80 total: 20% of the mass has vanished
    assert "invariant_row_stochastic" in _codes(iv.check_frame(bad, "credit"))


def test_a_cumulative_probability_table_is_not_a_transition_matrix():
    """Rows of probabilities that legitimately sum to much more than one must be left alone."""
    df = pd.DataFrame({"rating": ["AAA", "BBB"],
                       "horizon_1": [0.001, 0.02], "horizon_3": [0.004, 0.06],
                       "horizon_5": [0.008, 0.11], "horizon_10": [0.02, 0.22]})
    assert iv.check_frame(df, "credit") == []


def test_a_large_self_reported_residual_warns():
    """Tasks that ask for a `parity_error` are stating their own invariant; a big one confesses."""
    small = pd.DataFrame({"call_price": [10.0, 12.0], "parity_error": [1e-9, 2e-9]})
    assert iv.check_frame(small, "derivatives-pricing") == []
    large = pd.DataFrame({"call_price": [10.0, 12.0], "parity_error": [0.0, 3.5]})
    assert _levels(iv.check_frame(large, "derivatives-pricing"),
                   "invariant_self_reported_residual") == {iv.WARNING}


def test_negative_volatility_is_an_error():
    """Measured: a failing unit emitted a negative implied vol from an approximation formula whose
    discriminant the instruction explicitly says to clip."""
    df = pd.DataFrame({"sigma_cmh": [0.21, -0.039]})
    assert _levels(iv.check_frame(df, "derivatives-pricing"), "invariant_bound") == {iv.ERROR}


def test_a_percent_quoted_volatility_is_not_flagged():
    df = pd.DataFrame({"mark_iv_pct": [45.0, 62.0], "sigma_pct": [45.0, 50.0]})
    assert iv.check_frame(df, "derivatives-pricing") == []


# --- category resolution -----------------------------------------------------------------------
#
# Only 53 of the 87 public cards use one of the ten documented category ids. The rest invent their
# own, so resolution by keyword is not a nicety.

@pytest.mark.parametrize("raw,expected", [
    ("derivatives_pricing", ["derivatives-pricing"]),
    ("pricing", ["derivatives-pricing"]),
    ("volatility-modeling", ["derivatives-pricing"]),
    # A swaption task genuinely needs both halves, and resolving to both is the right answer.
    ("interest-rate-derivatives", ["fixed-income", "derivatives-pricing"]),
    ("credit-risk", ["credit"]),
    ("credit-analysis", ["credit"]),
    ("risk-modeling", ["risk-management"]),
    ("extreme-value-theory", ["risk-management"]),
    ("execution", ["microstructure"]),
    ("performance-attribution", ["factor-research"]),
    ("cross-asset-analysis", ["cross-domain"]),
])
def test_free_text_categories_resolve(raw, expected):
    assert iv.canonical_categories(raw) == expected


def test_a_card_naming_two_areas_gets_both_and_the_cross_domain_warning():
    assert set(iv.canonical_categories("fixed-income-nlp")) == {"fixed-income", "nlp-on-finance"}
    text = iv.guidance_for("fixed-income-nlp")
    assert "Discount factors" in text and "boilerplate" in text and "spans more than one" in text


def test_an_unrecognised_category_yields_the_shared_block_alone():
    assert iv.canonical_categories("tool-using") == []
    assert iv.guidance_for("tool-using").strip() == iv._SHARED.strip()


def test_greek_severity_follows_the_resolved_category_not_the_literal_id():
    """A card saying `derivatives_pricing` must be treated like one saying `derivatives-pricing`."""
    df = _option_book().assign(vega=-20.0)
    assert _levels(iv.check_frame(df, "derivatives_pricing"), "invariant_vega_sign") == {iv.ERROR}
