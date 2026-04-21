"""Frozen evaluation harness for the autoresearch loop.

This module is **not** edited by the agent. It defines the single
function :func:`evaluate` that scores one candidate PrimeKV
configuration and returns a stable, comparable scalar.

The score is designed to reward the thing we actually care about:
preserving *constrained* content under compression. Concretely:

    score = reasoning_pass_rate - lambda_ppl * perplexity_delta

where ``reasoning_pass_rate`` is the fraction of constraint tests a
cache passed, and ``perplexity_delta`` is ``ppl_candidate - ppl_full``
(positive means the candidate is worse). ``lambda_ppl`` trades off
the two signals.

The evaluate function is intentionally CPU-friendly (GPT-2 small,
short prompts) so the loop can iterate fast on a laptop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from primekv.baselines import FullCache
from primekv.cache import PrimeKVCache
from primekv.eval import Workload, run_comparison
from primekv.reasoning_eval import default_tests, run_reasoning_eval

log = logging.getLogger("autoresearch.prepare")


@dataclass
class EvalResult:
    """One candidate's evaluation."""

    score: float
    ppl_full: Optional[float]
    ppl_candidate: Optional[float]
    reasoning_pass_rate: float
    compression_ratio: float
    notes: str = ""


def load_model(model_name: str = "sshleifer/tiny-gpt2", device: str = "cpu"):
    """Load a tiny HF model for fast iteration. Separate function so the
    agent can swap it in ``experiment.py`` if it wants a bigger base."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    return model, tok


def evaluate(
    candidate: PrimeKVCache,
    model: Any,
    tokenizer: Any,
    prompt: str,
    decode_tokens: int = 16,
    lambda_ppl: float = 0.01,
    device: str = "cpu",
) -> EvalResult:
    """Score ``candidate`` against a fresh ``FullCache`` baseline.

    The harness is small by design — one workload for perplexity,
    the canonical reasoning suite for pass rate. A candidate that
    regresses perplexity and fails reasoning tests scores lower
    than the baseline.
    """
    num_layers = model.config.num_hidden_layers
    full = FullCache(num_layers=num_layers)

    caches = {"full": full, "candidate": candidate}
    workload = Workload(prompt=prompt, decode_tokens=decode_tokens, max_length=512)
    report = run_comparison(caches, workload, model, tokenizer, device=device)

    res_by_name = {r.name: r for r in report.results}
    ppl_full = res_by_name["full"].perplexity
    ppl_cand = res_by_name["candidate"].perplexity
    ratio = res_by_name["candidate"].compression_ratio

    # Reset before reasoning eval — run_comparison already did, but be explicit.
    full.reset()
    candidate.reset()
    reasoning = run_reasoning_eval(
        {"candidate": candidate},
        model,
        tokenizer,
        tests=default_tests(),
        device=device,
        max_length=1024,
    )
    pass_rate = reasoning.pass_rate_per_cache().get("candidate", 0.0)

    ppl_delta = 0.0
    if ppl_full is not None and ppl_cand is not None:
        ppl_delta = float(ppl_cand - ppl_full)

    score = pass_rate - lambda_ppl * ppl_delta
    notes = (
        f"pass_rate={pass_rate:.2f} ppl_full={ppl_full} ppl_cand={ppl_cand} "
        f"ratio={ratio:.2f}"
    )
    log.info("candidate score=%.4f %s", score, notes)
    return EvalResult(
        score=score,
        ppl_full=ppl_full,
        ppl_candidate=ppl_cand,
        reasoning_pass_rate=pass_rate,
        compression_ratio=ratio,
        notes=notes,
    )
