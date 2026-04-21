"""Tests for primekv.reasoning_eval.

The model-free tests cover data structures and grading. A slow end-to-
end test runs the real ``run_reasoning_eval`` against GPT-2 tiny and
just asserts the report has the right shape — pass/fail rates on tiny
GPT-2 are not meaningful, but the pipeline must not crash.
"""

from __future__ import annotations

import pytest

from primekv.reasoning_eval import (
    ReasoningReport,
    ReasoningResult,
    ReasoningTest,
    default_tests,
    grade,
    run_reasoning_eval,
)


def test_default_tests_non_empty_and_well_formed():
    tests = default_tests()
    assert len(tests) >= 3
    for t in tests:
        assert t.name
        assert t.category
        assert t.setup
        assert t.question
        # Every test has at least one grading criterion.
        assert t.expected_any or t.forbidden


def test_reasoning_test_build_prompt_concatenates_parts():
    t = ReasoningTest(
        name="x",
        setup="SETUP",
        filler="FILLER",
        question="Q?",
    )
    prompt = t.build_prompt()
    assert "SETUP" in prompt
    assert "FILLER" in prompt
    assert "Q?" in prompt
    # Order: setup → filler → question.
    assert prompt.index("SETUP") < prompt.index("FILLER") < prompt.index("Q?")


def test_grade_forbidden_substring_fails():
    t = ReasoningTest(
        name="x",
        setup="",
        filler="",
        question="",
        forbidden=("blue",),
    )
    passed, reason = grade(t, "Answer: the sky is blue today.")
    assert not passed
    assert "blue" in reason


def test_grade_expected_substring_passes():
    t = ReasoningTest(
        name="x",
        setup="",
        filler="",
        question="",
        expected_any=("42",),
    )
    passed, reason = grade(t, "Answer: she is 42 years old.")
    assert passed
    assert "42" in reason


def test_grade_prefers_tail_after_answer_marker():
    """Text before 'Answer:' is part of the prompt; only the tail counts."""
    t = ReasoningTest(
        name="x",
        setup="",
        filler="",
        question="",
        forbidden=("blue",),
    )
    # Prompt contains "blue" (in the rule), but tail does not.
    gen = "Never mention blue. Answer: the sky is clear and bright."
    passed, _ = grade(t, gen)
    assert passed


def test_grade_expected_missing_fails():
    t = ReasoningTest(
        name="x",
        setup="",
        filler="",
        question="",
        expected_any=("42",),
    )
    passed, reason = grade(t, "Answer: I do not know.")
    assert not passed
    assert "42" in reason or "none" in reason.lower()


def test_report_pass_rate_computes_per_cache():
    report = ReasoningReport()
    report.results = [
        ReasoningResult(
            cache="a", test="t1", category="x", passed=True,
            generated="", reason="", memory_bytes=0,
        ),
        ReasoningResult(
            cache="a", test="t2", category="x", passed=False,
            generated="", reason="", memory_bytes=0,
        ),
        ReasoningResult(
            cache="b", test="t1", category="x", passed=True,
            generated="", reason="", memory_bytes=0,
        ),
    ]
    rates = report.pass_rate_per_cache()
    assert rates["a"] == 0.5
    assert rates["b"] == 1.0


def test_report_to_markdown_includes_caches_and_tests():
    report = ReasoningReport()
    report.results = [
        ReasoningResult(
            cache="a", test="t1", category="x", passed=True,
            generated="", reason="", memory_bytes=0,
        ),
        ReasoningResult(
            cache="b", test="t1", category="x", passed=False,
            generated="", reason="", memory_bytes=0,
        ),
    ]
    md = report.to_markdown()
    assert "t1" in md
    assert "a" in md
    assert "b" in md
    assert "PASS" in md
    assert "FAIL" in md


@pytest.mark.slow
def test_run_reasoning_eval_end_to_end_smoke():
    """Pipeline must not crash on tiny-gpt2."""
    transformers = pytest.importorskip("transformers")
    from primekv.baselines import FullCache

    try:
        tok = transformers.AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
        model = transformers.AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
    except (OSError, Exception) as e:
        pytest.skip(f"tiny-gpt2 unavailable (likely no network): {e}")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.eval()

    num_layers = model.config.num_hidden_layers
    caches = {"full": FullCache(num_layers=num_layers)}
    report = run_reasoning_eval(caches, model, tok, tests=default_tests()[:1])
    assert len(report.results) == 1
    assert report.results[0].cache == "full"
