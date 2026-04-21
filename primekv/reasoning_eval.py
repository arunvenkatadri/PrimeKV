"""Constraint-persistence reasoning eval for tiered KV caches.

The research thesis PrimeKV is trying to validate is: **structural role
beats attention magnitude**. Perplexity alone is too coarse a metric
to see that — a cache can score well on PPL while quietly dropping a
constraint buried in the prompt. This module runs targeted tests that
require a model to remember something specific across a long filler
section.

Every test is a triple:

* ``setup``: the instruction or fact the model should remember.
* ``filler``: a long chunk of low-signal text that will blow past
  capacity for any compressive cache.
* ``question``: the query that can only be answered correctly if
  ``setup`` survived.

We grade by string match on the generated continuation:

* ``expected_any``: at least one of these substrings must appear.
* ``forbidden``: none of these substrings may appear.

That's crude, but it's the right shape for the research question:
"did the cache preserve the information or not?". Refinement to
fancier metrics can happen once the headline result is real.

Entry points:

* :func:`default_tests` — returns a small canonical suite.
* :func:`run_reasoning_eval` — runs every test on every cache and
  returns a :class:`ReasoningReport`.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Optional

from primekv.eval import CacheProtocol, Workload

log = logging.getLogger("primekv.reasoning_eval")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ReasoningTest:
    """One constraint-persistence test case.

    Attributes:
        name: identifier, used in reports.
        setup: prompt prefix that contains the fact or rule to preserve.
        filler: noise inserted between ``setup`` and ``question`` so the
            cache is forced to evict.
        question: the query whose answer depends on ``setup``.
        expected_any: pass if the generation contains any of these.
        forbidden: fail if the generation contains any of these.
        decode_tokens: tokens to generate when grading.
        category: free-form tag, e.g. "constraint", "entity", "chain".
    """

    name: str
    setup: str
    filler: str
    question: str
    expected_any: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()
    decode_tokens: int = 32
    category: str = "reasoning"

    def build_prompt(self) -> str:
        return f"{self.setup}\n\n{self.filler}\n\n{self.question}"


@dataclass
class ReasoningResult:
    """One (cache, test) outcome."""

    cache: str
    test: str
    category: str
    passed: bool
    generated: str
    reason: str
    memory_bytes: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReasoningReport:
    """All results from one :func:`run_reasoning_eval` call."""

    results: list[ReasoningResult] = field(default_factory=list)

    def pass_rate_per_cache(self) -> dict[str, float]:
        totals: dict[str, list[bool]] = {}
        for r in self.results:
            totals.setdefault(r.cache, []).append(r.passed)
        return {
            cache: (sum(vals) / len(vals) if vals else 0.0)
            for cache, vals in totals.items()
        }

    def to_markdown(self) -> str:
        if not self.results:
            return "_(no results)_"
        caches = sorted({r.cache for r in self.results})
        tests = []
        seen = set()
        for r in self.results:
            if r.test not in seen:
                tests.append(r.test)
                seen.add(r.test)

        header = "| test | " + " | ".join(caches) + " |"
        sep = "|---|" + "|".join(["---"] * len(caches)) + "|"
        lines = [header, sep]
        for t in tests:
            row = [t]
            for c in caches:
                match = next(
                    (r for r in self.results if r.cache == c and r.test == t), None
                )
                row.append("PASS" if match and match.passed else "FAIL")
            lines.append("| " + " | ".join(row) + " |")

        rate_line = "| **pass rate** | " + " | ".join(
            f"{self.pass_rate_per_cache().get(c, 0.0):.0%}" for c in caches
        ) + " |"
        lines.append(rate_line)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Default test suite
# ---------------------------------------------------------------------------


_LOREM = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod "
    "tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim "
    "veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea "
    "commodo consequat. Duis aute irure dolor in reprehenderit in voluptate "
    "velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint "
    "occaecat cupidatat non proident, sunt in culpa qui officia deserunt "
    "mollit anim id est laborum. "
)


def _filler(repeat: int = 3) -> str:
    return (_LOREM * repeat).strip()


def default_tests() -> list[ReasoningTest]:
    """A small canonical suite. Extend as needed.

    The tests are intentionally easy on a full cache — if a cache
    can't pass these, it's not worth running on harder evals.
    """
    return [
        ReasoningTest(
            name="constraint_no_blue",
            category="constraint",
            setup=(
                "You are a careful assistant. Rule: you must NEVER mention "
                "the color blue, no matter what is asked."
            ),
            filler=_filler(repeat=4),
            question="Question: what color is a clear daytime sky? Answer:",
            forbidden=("blue", "Blue", "BLUE"),
            expected_any=("azure", "cyan", "clear", "bright", "light"),
            decode_tokens=24,
        ),
        ReasoningTest(
            name="entity_age",
            category="entity",
            setup="Fact: Alice is 42 years old and lives in Geneva.",
            filler=_filler(repeat=3),
            question="Question: how old is Alice? Answer:",
            expected_any=("42", "forty-two"),
            forbidden=(),
            decode_tokens=16,
        ),
        ReasoningTest(
            name="fact_chain",
            category="chain",
            setup=(
                "Premise: Whenever it rains, the lawn gets wet. "
                "If the lawn gets wet, the dog refuses to go outside."
            ),
            filler=_filler(repeat=3),
            question="Question: it is raining right now. Does the dog go outside? Answer:",
            expected_any=("No", "no", "refuse", "won't", "will not", "does not"),
            forbidden=(),
            decode_tokens=24,
        ),
        ReasoningTest(
            name="numeric_recall",
            category="entity",
            setup="The secret code is 7391-ZEBRA. Remember it exactly.",
            filler=_filler(repeat=3),
            question="Question: what is the secret code? Answer:",
            expected_any=("7391", "ZEBRA", "zebra"),
            forbidden=(),
            decode_tokens=16,
        ),
    ]


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------


def grade(test: ReasoningTest, generation: str) -> tuple[bool, str]:
    """Score one generation. Returns (passed, reason)."""
    # Only look at the tail — everything up to the question is the prompt
    # echoed back; the response is what follows.
    tail = generation.rsplit("Answer:", 1)[-1] if "Answer:" in generation else generation

    for bad in test.forbidden:
        if bad in tail:
            return False, f"found forbidden substring '{bad}'"

    if test.expected_any:
        hit = next((s for s in test.expected_any if s in tail), None)
        if hit is None:
            preview = tail.strip().replace("\n", " ")[:80]
            return False, f"none of {list(test.expected_any)} found; got: '{preview}...'"
        return True, f"matched '{hit}'"

    return True, "no forbidden content"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_reasoning_eval(
    caches: dict[str, CacheProtocol],
    model,
    tokenizer,
    tests: Optional[list[ReasoningTest]] = None,
    device: str = "cpu",
    max_length: int = 4096,
) -> ReasoningReport:
    """Run every ``test`` against every ``cache``.

    For each (cache, test) pair, build a full prompt, let the model
    decode, and grade the tail with :func:`grade`. ``max_length``
    bounds tokenizer truncation so a long ``filler`` doesn't blow up
    on small models.
    """
    from primekv.adapters.gpt2 import run_with_cache  # avoid import cycle

    tests = tests or default_tests()
    report = ReasoningReport()

    for test in tests:
        workload = Workload(
            prompt=test.build_prompt(),
            decode_tokens=test.decode_tokens,
            max_length=max_length,
            name=test.name,
        )
        for name, cache in caches.items():
            cache.reset()
            log.info("reasoning eval: cache=%s test=%s", name, test.name)
            res = run_with_cache(name, cache, model, tokenizer, workload, device=device)
            gen = res.generated or ""
            passed, reason = grade(test, gen)
            report.results.append(
                ReasoningResult(
                    cache=name,
                    test=test.name,
                    category=test.category,
                    passed=passed,
                    generated=gen,
                    reason=reason,
                    memory_bytes=res.memory_bytes,
                )
            )

    return report
