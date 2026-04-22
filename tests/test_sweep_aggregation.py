"""Tests for seed aggregation in the sweep harness.

These don't need a model — we build SweepReports by hand and check
that aggregate_reports correctly combines them into mean/std per cell.
"""

import math

from primekv.sweep import SweepPoint, SweepReport, aggregate_reports


def _point(cache, sweep_value, ppl, mem, extra=None):
    return SweepPoint(
        cache=cache,
        sweep_axis="capacity",
        sweep_value=float(sweep_value),
        memory_bytes=mem,
        compression_ratio=1.0,
        perplexity=ppl,
        prefill_ms=1.0,
        decode_ms=1.0,
        tokens_per_second=10.0,
        extra=extra or {},
    )


def _report(points, seed):
    return SweepReport(
        mode="pareto",
        axis_label="capacity",
        points=points,
        meta={"seed": seed},
    )


def test_aggregate_single_report_is_idempotent():
    r = _report([_point("pkv", 32, 1.5, 100)], seed=0)
    agg = aggregate_reports([r])
    assert len(agg.points) == 1
    assert agg.points[0].perplexity == 1.5
    # std over one sample is 0.
    assert agg.points[0].extra["perplexity_std"] == 0.0
    assert agg.points[0].extra["n_runs"] == 1


def test_aggregate_means_and_stds_across_runs():
    reports = [
        _report([_point("pkv", 32, 1.0, 100)], seed=0),
        _report([_point("pkv", 32, 2.0, 100)], seed=1),
        _report([_point("pkv", 32, 3.0, 100)], seed=2),
    ]
    agg = aggregate_reports(reports)
    assert len(agg.points) == 1
    p = agg.points[0]
    assert p.perplexity == 2.0  # (1+2+3)/3
    # Population stdev of {1,2,3} ≈ 0.8165.
    assert math.isclose(p.extra["perplexity_std"], math.sqrt(2 / 3), rel_tol=1e-6)
    assert p.extra["n_runs"] == 3
    assert agg.meta["seeds"] == [0, 1, 2]


def test_aggregate_groups_by_cache_and_sweep_value_and_extra():
    # Same cache and capacity, different `precision` extra → separate cells.
    reports = [
        _report([
            _point("pkv", 32, 1.0, 100, extra={"precision": "int4"}),
            _point("pkv", 32, 1.5, 200, extra={"precision": "fp16"}),
        ], seed=0),
        _report([
            _point("pkv", 32, 2.0, 100, extra={"precision": "int4"}),
            _point("pkv", 32, 2.5, 200, extra={"precision": "fp16"}),
        ], seed=1),
    ]
    agg = aggregate_reports(reports)
    cells = {(p.cache, p.sweep_value, p.extra["precision"]): p for p in agg.points}
    assert len(cells) == 2
    assert cells[("pkv", 32.0, "int4")].perplexity == 1.5
    assert cells[("pkv", 32.0, "fp16")].perplexity == 2.0


def test_aggregate_rejects_mismatched_modes():
    r1 = _report([_point("pkv", 32, 1.0, 100)], seed=0)
    r2 = SweepReport(mode="vs_length", axis_label="prompt_length", points=[
        _point("pkv", 32, 1.0, 100)
    ], meta={"seed": 1})
    try:
        aggregate_reports([r1, r2])
    except ValueError:
        return
    raise AssertionError("expected ValueError for mismatched modes")
