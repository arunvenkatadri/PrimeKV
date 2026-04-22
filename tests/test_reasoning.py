"""Tests for the reasoning-persistence harness.

Focus here is on the reporting math — especially the filtered
pass-rate logic that was missing from the original constraint
charts. We don't run a real model; outcomes are constructed by hand.
"""

from primekv.reasoning import (
    ReasoningOutcome,
    ReasoningReport,
    ReasoningTest,
    default_reasoning_suite,
)


def _mk(cache, test, passed, seed=None):
    return ReasoningOutcome(cache=cache, test=test, seed=seed, passed=passed, generated="")


def test_pass_rate_raw():
    r = ReasoningReport(outcomes=[
        _mk("full", "t1", True),
        _mk("full", "t2", False),
        _mk("pkv", "t1", True),
        _mk("pkv", "t2", True),
    ])
    assert r.pass_rate("full") == 0.5
    assert r.pass_rate("pkv") == 1.0


def test_filtered_pass_rate_ignores_tests_full_fails():
    # full passes only t1. A cache that passes t1+t2 should score 100%
    # on the filtered metric (since t2 is excluded) — even though its
    # raw pass rate is also 100%. A cache that passes only t2 should
    # score 0% filtered — because t2 is outside the valid denominator.
    r = ReasoningReport(outcomes=[
        _mk("full", "t1", True),
        _mk("full", "t2", False),
        _mk("good", "t1", True),
        _mk("good", "t2", True),
        _mk("lucky", "t1", False),
        _mk("lucky", "t2", True),
    ])
    assert r.filtered_pass_rate("good") == 1.0
    # "lucky" gets credit on t2 (which full failed) and is excluded.
    # Its only valid-test outcome is t1, which it failed → 0%.
    assert r.filtered_pass_rate("lucky") == 0.0
    # Raw rate would have been 50% — the exact bug the filter prevents.
    assert r.pass_rate("lucky") == 0.5


def test_filtered_rate_uses_matching_seed_keys():
    # The filter keys by (test, seed), not just test. So "full" passing
    # t1 with seed=0 doesn't validate t1 outcomes at other seeds.
    r = ReasoningReport(outcomes=[
        _mk("full", "t1", True, seed=0),
        _mk("full", "t1", False, seed=1),
        _mk("pkv", "t1", True, seed=0),
        _mk("pkv", "t1", True, seed=1),
    ])
    # Only (t1, seed=0) is in the valid set; pkv passed it → 100%.
    assert r.filtered_pass_rate("pkv") == 1.0


def test_per_test_matrix_majority_vote_across_seeds():
    r = ReasoningReport(outcomes=[
        _mk("pkv", "t1", True, seed=0),
        _mk("pkv", "t1", True, seed=1),
        _mk("pkv", "t1", False, seed=2),
    ])
    matrix = r.per_test_matrix()
    # 2/3 seeds passed → majority True.
    assert matrix["t1"]["pkv"] is True


def test_summary_includes_n_valid_tests():
    r = ReasoningReport(outcomes=[
        _mk("full", "t1", True),
        _mk("full", "t2", False),
        _mk("pkv", "t1", True),
        _mk("pkv", "t2", True),
    ])
    s = r.summary()
    assert s["full"]["n_valid_tests"] == 1
    assert s["pkv"]["filtered_pass_rate"] == 1.0


def test_default_suite_runs_without_errors():
    # Just instantiation + scorer shape; no model required.
    tests = default_reasoning_suite()
    assert len(tests) >= 4
    for t in tests:
        assert isinstance(t, ReasoningTest)
        # Scorer must accept a string and return something truthy/falsy.
        assert t.scorer("") in (True, False)
