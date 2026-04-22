"""Task-level reasoning-persistence harness.

Perplexity averages over every decoded token and hides whether the
*right* tokens were retained — which is exactly where eviction-based
caches can quietly lose information that quantization-only caches
keep. This module scores caches by binary pass/fail on small
reasoning tasks where the scorer knows which fact the model should
have held on to.

Two concepts:

* :class:`ReasoningTest` — a (prompt, scorer) pair plus decode budget.
  The scorer takes the cache-generated completion and returns
  ``True``/``False``.
* :class:`ReasoningReport` — collected outcomes with a
  **filtered pass rate**: only count tests where the ``full`` baseline
  already passes. Without this filter a cache that happens to
  hallucinate the right answer on tests ``full`` fails can spuriously
  appear to beat ``full`` — the constraint-persistence charts that
  reported ``spacy > full`` fell into exactly this trap.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

from primekv.eval import CacheProtocol, Workload

log = logging.getLogger("primekv.reasoning")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ReasoningTest:
    """One scripted task with a scorer.

    Attributes:
        name: Short identifier used in reports (e.g. ``"numeric_recall"``).
        prompt: Prompt fed to the model. Should set up the question and
            leave the model to complete the answer.
        scorer: Callable that takes the *generated continuation only*
            (excluding the prompt) and returns ``True`` if the cache
            preserved enough information to produce the correct answer.
        decode_tokens: How many tokens to decode. Keep this short —
            scoring gets noisier with longer continuations.
        max_length: Tokenizer truncation length for the prompt.
    """

    name: str
    prompt: str
    scorer: Callable[[str], bool]
    decode_tokens: int = 32
    max_length: int = 512


@dataclass
class ReasoningOutcome:
    """One (cache, test, seed) run."""

    cache: str
    test: str
    seed: Optional[int]
    passed: bool
    generated: str


@dataclass
class ReasoningReport:
    """Collected reasoning-test outcomes.

    Use :meth:`filtered_pass_rate` for the headline metric. The raw
    :meth:`pass_rate` is only honest when the ``full`` baseline passes
    every test — otherwise it credits caches for getting lucky on
    questions the uncompressed model can't answer.
    """

    outcomes: list[ReasoningOutcome] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ lookup

    def caches(self) -> list[str]:
        return sorted({o.cache for o in self.outcomes})

    def tests(self) -> list[str]:
        return sorted({o.test for o in self.outcomes})

    def for_cache(self, cache: str) -> list[ReasoningOutcome]:
        return [o for o in self.outcomes if o.cache == cache]

    # ------------------------------------------------------------------ metrics

    def pass_rate(self, cache: str) -> float:
        """Fraction of outcomes where ``cache`` passed. Raw, unfiltered."""
        outs = self.for_cache(cache)
        if not outs:
            return float("nan")
        return sum(1 for o in outs if o.passed) / len(outs)

    def passing_tests(self, cache: str) -> set[tuple[str, Optional[int]]]:
        """Set of ``(test_name, seed)`` keys that ``cache`` passed."""
        return {(o.test, o.seed) for o in self.for_cache(cache) if o.passed}

    def filtered_pass_rate(self, cache: str, baseline: str = "full") -> float:
        """Pass rate over tests the ``baseline`` (full) cache already passes.

        This is the honest cache-quality metric: among tests the
        uncompressed model can actually answer correctly, what fraction
        survives under ``cache``? Tests where even ``full`` fails are
        model-capability failures, not cache failures, and are excluded.
        """
        valid_keys = self.passing_tests(baseline)
        if not valid_keys:
            return float("nan")
        outs = [o for o in self.for_cache(cache) if (o.test, o.seed) in valid_keys]
        if not outs:
            return float("nan")
        return sum(1 for o in outs if o.passed) / len(outs)

    def summary(self, baseline: str = "full") -> dict:
        """One-shot metric rollup for every cache in the report."""
        valid_keys = self.passing_tests(baseline)
        out: dict[str, dict[str, float]] = {}
        for cache in self.caches():
            out[cache] = {
                "raw_pass_rate": self.pass_rate(cache),
                "filtered_pass_rate": self.filtered_pass_rate(cache, baseline),
                "n_tests": len(self.for_cache(cache)),
                "n_valid_tests": len(valid_keys),
            }
        return out

    # ------------------------------------------------------------------ matrix

    def per_test_matrix(self) -> dict[str, dict[str, Optional[bool]]]:
        """``{test_name: {cache_name: passed}}`` — aggregated over seeds.

        When a test was run against a cache with multiple seeds, the
        value is the majority vote (ties break to True). Seeds with no
        outcome are absent. Useful for the per-test heatmap.
        """
        matrix: dict[str, dict[str, list[bool]]] = {}
        for o in self.outcomes:
            matrix.setdefault(o.test, {}).setdefault(o.cache, []).append(o.passed)
        out: dict[str, dict[str, Optional[bool]]] = {}
        for test, row in matrix.items():
            out[test] = {
                cache: (sum(results) * 2 >= len(results)) if results else None
                for cache, results in row.items()
            }
        return out


# ---------------------------------------------------------------------------
# Default suite
# ---------------------------------------------------------------------------


def _contains_any(text: str, needles: list[str]) -> bool:
    t = text.lower()
    return any(n.lower() in t for n in needles)


def _contains_none(text: str, needles: list[str]) -> bool:
    t = text.lower()
    return not any(n.lower() in t for n in needles)


def default_reasoning_suite() -> list[ReasoningTest]:
    """A small built-in suite for smoke-testing the harness.

    These are deliberately simple and short so they run on CPU with
    GPT-2. Replace with a real benchmark (LongBench, RULER,
    needle-in-haystack) for any serious evaluation — this only exists
    to exercise the plumbing.
    """
    tests: list[ReasoningTest] = []

    tests.append(ReasoningTest(
        name="numeric_recall",
        prompt=(
            "Alice received 427 apples from her grandfather on her birthday. "
            "She gave some to her friends and ate some herself. "
            "Later, when asked how many apples she received, Alice answered: "
        ),
        scorer=lambda text: "427" in text,
        decode_tokens=8,
        max_length=128,
    ))

    tests.append(ReasoningTest(
        name="entity_age",
        prompt=(
            "Dr. Eleanor Vasquez, a 52-year-old marine biologist, discovered "
            "a new species of bioluminescent jellyfish in 2019. "
            "Question: How old is Dr. Vasquez? Answer: She is "
        ),
        scorer=lambda text: "52" in text,
        decode_tokens=8,
        max_length=128,
    ))

    tests.append(ReasoningTest(
        name="first_entity",
        prompt=(
            "The conference was opened by Sarah Chen, followed by remarks from "
            "Michael Rodriguez and Priya Patel. After a short break, James "
            "O'Brien presented the keynote address. "
            "Question: Who opened the conference? Answer: "
        ),
        scorer=lambda text: "sarah" in text.lower() or "chen" in text.lower(),
        decode_tokens=8,
        max_length=128,
    ))

    tests.append(ReasoningTest(
        name="constraint_no_blue",
        prompt=(
            "You are describing a sunset. Do not mention the color blue. "
            "Describe the scene briefly: "
        ),
        scorer=lambda text: _contains_none(text, ["blue"]),
        decode_tokens=16,
        max_length=64,
    ))

    tests.append(ReasoningTest(
        name="list_membership",
        prompt=(
            "The committee has seven members: Anna, Ben, Carlos, Diana, "
            "Emily, Frank, and Grace. "
            "Question: Is Diana on the committee? Answer: "
        ),
        scorer=lambda text: _contains_any(text, ["yes", "she is", "diana is"]),
        decode_tokens=8,
        max_length=128,
    ))

    return tests


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _strip_prompt(generated: str, prompt: str) -> str:
    """Return only the model's continuation, stripping the echoed prompt."""
    if generated.startswith(prompt):
        return generated[len(prompt):]
    # Some tokenizers normalize whitespace on decode; fall back to a
    # cheap suffix lookup.
    marker = prompt.strip()[-40:] if len(prompt) > 40 else prompt.strip()
    idx = generated.rfind(marker)
    if idx >= 0:
        return generated[idx + len(marker):]
    return generated


def run_reasoning_suite(
    caches: dict[str, CacheProtocol],
    tests: list[ReasoningTest],
    model,
    tokenizer,
    seeds: Optional[list[Optional[int]]] = None,
    device: str = "cpu",
    sample_top_k: int = 0,
    sample_temperature: float = 1.0,
    cache_factories: Optional[dict[str, Callable[[], CacheProtocol]]] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> ReasoningReport:
    """Run every (cache, test, seed) combination and score pass/fail.

    Args:
        caches: Cache instances keyed by name. Each is reset between
            tests via ``cache.reset()``. If ``cache_factories`` is
            given, it is preferred — factories produce a fresh cache
            per run, which matters for caches with lingering state
            (PrimeKV's classifier assignment, H2O's attention sums).
        tests: :class:`ReasoningTest` instances.
        model, tokenizer: HuggingFace objects.
        seeds: If ``None``, run once with ``seed=None`` (deterministic
            argmax if ``sample_top_k == 0``). Otherwise run each test
            once per seed.
        sample_top_k: >0 enables top-k sampling. Required for seeds to
            actually change outputs.
        cache_factories: Optional ``{name: () -> cache}``. Takes
            precedence over the ``caches`` dict when building a fresh
            cache per run. Use this for PrimeKV because ``reset()``
            keeps the classifier state.

    Returns a :class:`ReasoningReport`.
    """
    from primekv.adapters.gpt2 import run_with_cache  # lazy: avoid cycles

    seeds = seeds if seeds is not None else [None]
    factories = cache_factories or {}
    report = ReasoningReport(
        meta={
            "n_tests": len(tests),
            "n_seeds": len(seeds),
            "sample_top_k": sample_top_k,
            "sample_temperature": sample_temperature,
        }
    )

    for test in tests:
        for seed in seeds:
            for name in caches.keys():
                if name in factories:
                    cache = factories[name]()
                else:
                    cache = caches[name]
                    cache.reset()

                workload = Workload(
                    prompt=test.prompt,
                    decode_tokens=test.decode_tokens,
                    max_length=test.max_length,
                    name=test.name,
                    seed=seed,
                    sample_top_k=sample_top_k,
                    sample_temperature=sample_temperature,
                )
                if progress:
                    progress(f"running test={test.name} cache={name} seed={seed}")

                result = run_with_cache(name, cache, model, tokenizer, workload, device=device)
                generated_full = result.generated or ""
                continuation = _strip_prompt(generated_full, test.prompt)
                try:
                    passed = bool(test.scorer(continuation))
                except Exception as exc:  # pragma: no cover
                    log.warning("scorer raised on test=%s cache=%s: %s", test.name, name, exc)
                    passed = False

                report.outcomes.append(ReasoningOutcome(
                    cache=name,
                    test=test.name,
                    seed=seed,
                    passed=passed,
                    generated=continuation,
                ))

    return report


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_reasoning_report(
    report: ReasoningReport,
    output_path: Optional[str] = None,
    baseline: str = "full",
):
    """Render the filtered pass-rate bar + per-test heatmap.

    The left panel is the honest headline: filtered pass rate per
    cache, with the baseline shown separately. The right panel is the
    per-test matrix, same as the constraint-persistence charts we've
    been producing.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    caches = report.caches()
    tests = report.tests()

    fig, (ax_bar, ax_mat) = plt.subplots(1, 2, figsize=(12, 5))

    # --- bar: filtered pass rate ---------------------------------------
    rates = [report.filtered_pass_rate(c, baseline) for c in caches]
    bar_colors = ["#cc3333" if c == baseline else "#3377aa" for c in caches]
    ax_bar.bar(range(len(caches)), rates, color=bar_colors)
    ax_bar.set_xticks(range(len(caches)))
    ax_bar.set_xticklabels(caches, rotation=30, ha="right")
    ax_bar.set_ylim(0, 1.05)
    ax_bar.set_ylabel(f"pass rate (over tests '{baseline}' passes)")
    ax_bar.set_title("Filtered pass rate: honest cache quality")
    for i, r in enumerate(rates):
        if r == r:  # not NaN
            ax_bar.text(i, r + 0.02, f"{r*100:.0f}%", ha="center", fontsize=9)

    # --- matrix: per-test pass/fail ------------------------------------
    matrix = report.per_test_matrix()
    grid = np.zeros((len(tests), len(caches)))
    for i, test in enumerate(tests):
        for j, cache in enumerate(caches):
            val = matrix.get(test, {}).get(cache)
            grid[i, j] = 1.0 if val else 0.0
    ax_mat.imshow(grid, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax_mat.set_xticks(range(len(caches)))
    ax_mat.set_xticklabels(caches, rotation=30, ha="right")
    ax_mat.set_yticks(range(len(tests)))
    ax_mat.set_yticklabels(tests)
    for i in range(len(tests)):
        for j in range(len(caches)):
            mark = "✓" if grid[i, j] > 0.5 else "✗"
            ax_mat.text(j, i, mark, ha="center", va="center", color="white", fontsize=11)
    ax_mat.set_title("Per-test outcome (green = pass, red = fail)")

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=120)
    return fig
