"""Tests for primekv/sweep.py.

Uses the synthetic fake model from tests/test_adapters.py so no HF
downloads are required.
"""

import json

import torch

from primekv.sweep import (
    SweepPoint,
    SweepReport,
    plot_report,
    sweep_ablate_primekv,
    sweep_pareto,
    sweep_vs_length,
)

from tests.test_adapters import _FakeCausalLM, _FakeTokenizer


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


def test_sweep_report_serialization_roundtrip():
    report = SweepReport(
        mode="pareto",
        axis_label="capacity",
        points=[
            SweepPoint(
                cache="full",
                sweep_axis="capacity",
                sweep_value=float("inf"),
                memory_bytes=1000,
                compression_ratio=1.0,
                perplexity=7.3,
                prefill_ms=10.0,
                decode_ms=20.0,
                tokens_per_second=50.0,
            ),
            SweepPoint(
                cache="primekv",
                sweep_axis="capacity",
                sweep_value=16.0,
                memory_bytes=400,
                compression_ratio=2.5,
                perplexity=8.1,
                prefill_ms=12.0,
                decode_ms=22.0,
                tokens_per_second=45.0,
            ),
        ],
        meta={"model": "gpt2"},
    )
    d = report.to_dict()
    json.dumps(d)  # must be JSON-serializable
    assert d["mode"] == "pareto"
    assert len(d["points"]) == 2

    csv = report.to_csv()
    assert "cache,sweep_axis" in csv
    assert "full" in csv and "primekv" in csv

    groups = report.by_cache()
    assert set(groups.keys()) == {"full", "primekv"}


# ---------------------------------------------------------------------------
# Pareto sweep
# ---------------------------------------------------------------------------


def test_sweep_pareto_produces_expected_points():
    model = _FakeCausalLM(num_layers=2, num_heads=2, head_dim=4, vocab=17)
    tok = _FakeTokenizer(vocab=17)

    report = sweep_pareto(
        model=model,
        tokenizer=tok,
        prompt="hello world",
        capacities=[4, 8],
        caches=["full", "h2o", "primekv"],
        decode_tokens=2,
        max_length=8,
        device="cpu",
    )
    assert report.mode == "pareto"
    # full -> 1 point; h2o -> 2; primekv -> 2 = 5 total.
    assert len(report.points) == 5

    # Full cache's compression ratio should be 1.0.
    full_pts = [p for p in report.points if p.cache == "full"]
    assert len(full_pts) == 1
    assert full_pts[0].compression_ratio == 1.0

    # Capacity caches should have one point per capacity.
    h2o_pts = [p for p in report.points if p.cache == "h2o"]
    assert len(h2o_pts) == 2
    assert sorted(p.sweep_value for p in h2o_pts) == [4.0, 8.0]


def test_sweep_pareto_compression_ratio_is_normalized():
    """The harness normalizes every run's compression_ratio vs the 'full' cache."""
    model = _FakeCausalLM(num_layers=2, num_heads=2, head_dim=4, vocab=17)
    tok = _FakeTokenizer(vocab=17)

    report = sweep_pareto(
        model=model,
        tokenizer=tok,
        prompt="hello world",
        capacities=[4],
        caches=["full", "uniform_int4"],
        decode_tokens=1,
        max_length=8,
        device="cpu",
    )
    full_pt = next(p for p in report.points if p.cache == "full")
    int4_pt = next(p for p in report.points if p.cache == "uniform_int4")
    # uniform_int4 should be more compressed than full.
    assert int4_pt.compression_ratio > full_pt.compression_ratio


# ---------------------------------------------------------------------------
# vs length
# ---------------------------------------------------------------------------


def test_sweep_vs_length_produces_one_point_per_cache_per_length():
    model = _FakeCausalLM(num_layers=2, num_heads=2, head_dim=4, vocab=17)
    tok = _FakeTokenizer(vocab=17)

    report = sweep_vs_length(
        model=model,
        tokenizer=tok,
        prompt="a" * 64,
        lengths=[8, 16],
        caches=["full", "primekv"],
        capacity=4,
        decode_tokens=1,
        device="cpu",
    )
    assert report.mode == "vs_length"
    assert len(report.points) == 4  # 2 caches x 2 lengths

    full_lengths = [p.sweep_value for p in report.points if p.cache == "full"]
    assert sorted(full_lengths) == [8.0, 16.0]


# ---------------------------------------------------------------------------
# ablate
# ---------------------------------------------------------------------------


def test_sweep_ablate_anchor_prefix_len_runs():
    model = _FakeCausalLM(num_layers=2, num_heads=2, head_dim=4, vocab=17)
    tok = _FakeTokenizer(vocab=17)

    report = sweep_ablate_primekv(
        model=model,
        tokenizer=tok,
        prompt="hello world",
        axis="anchor_prefix_len",
        values=[0, 4, 8],
        capacity=4,
        decode_tokens=1,
        max_length=8,
        device="cpu",
    )
    assert report.mode == "ablate"
    assert report.axis_label == "anchor_prefix_len"
    assert len(report.points) == 3
    assert all(p.cache == "primekv" for p in report.points)


def test_sweep_ablate_rejects_unknown_axis():
    model = _FakeCausalLM(num_layers=2, num_heads=2, head_dim=4, vocab=17)
    tok = _FakeTokenizer(vocab=17)
    try:
        sweep_ablate_primekv(
            model=model,
            tokenizer=tok,
            prompt="x",
            axis="nonexistent_axis",
            values=[1, 2],
            device="cpu",
        )
        assert False, "should have raised"
    except ValueError as e:
        assert "unknown ablation axis" in str(e)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def test_plot_report_produces_a_figure(tmp_path):
    report = SweepReport(
        mode="pareto",
        axis_label="capacity",
        points=[
            SweepPoint(
                cache="full",
                sweep_axis="capacity",
                sweep_value=float("inf"),
                memory_bytes=1000,
                compression_ratio=1.0,
                perplexity=7.0,
                prefill_ms=1,
                decode_ms=1,
                tokens_per_second=1,
            ),
            SweepPoint(
                cache="primekv",
                sweep_axis="capacity",
                sweep_value=16.0,
                memory_bytes=400,
                compression_ratio=2.5,
                perplexity=8.0,
                prefill_ms=1,
                decode_ms=1,
                tokens_per_second=1,
            ),
            SweepPoint(
                cache="primekv",
                sweep_axis="capacity",
                sweep_value=4.0,
                memory_bytes=100,
                compression_ratio=10.0,
                perplexity=11.0,
                prefill_ms=1,
                decode_ms=1,
                tokens_per_second=1,
            ),
        ],
    )
    out = tmp_path / "plot.png"
    fig = plot_report(report, output_path=str(out))
    assert out.exists()
    assert out.stat().st_size > 0
    # Cleanup.
    import matplotlib.pyplot as plt

    plt.close(fig)
