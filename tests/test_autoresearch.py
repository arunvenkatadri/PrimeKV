"""Tests for the autoresearch scaffold (non-network)."""

from __future__ import annotations

from autoresearch import experiment
from primekv.cache import PrimeKVCache


def test_experiment_build_candidate_returns_primekv_cache():
    cache = experiment.build_candidate(num_layers=4)
    assert isinstance(cache, PrimeKVCache)
    assert cache.num_layers == 4


def test_run_strip_code_fences_handles_plain_code():
    from autoresearch.run import _strip_code_fences

    code = "def f():\n    return 1\n"
    assert _strip_code_fences(code).strip() == code.strip()


def test_run_strip_code_fences_removes_triple_backticks():
    from autoresearch.run import _strip_code_fences

    wrapped = "```python\ndef f():\n    return 1\n```\n"
    out = _strip_code_fences(wrapped)
    assert "```" not in out
    assert "def f():" in out


def test_program_md_exists_and_is_non_empty():
    from autoresearch.run import PROGRAM_PATH

    assert PROGRAM_PATH.exists()
    assert len(PROGRAM_PATH.read_text()) > 200
