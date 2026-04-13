import json

from primekv.eval import CacheResult, CompareReport, Workload


def test_workload_defaults():
    w = Workload(prompt="hi")
    assert w.decode_tokens == 32
    assert w.max_length == 256
    assert w.name == "default"


def test_cache_result_as_row_keys():
    r = CacheResult(
        name="full",
        memory_bytes=1_048_576,
        compression_ratio=2.0,
        perplexity=12.345,
        prefill_ms=1.5,
        decode_ms=2.5,
        tokens_per_second=40.0,
    )
    row = r.as_row()
    assert set(row.keys()) == {
        "cache",
        "memory_MB",
        "ratio",
        "ppl",
        "prefill_ms",
        "decode_ms",
        "tok_per_s",
    }
    assert row["memory_MB"] == 1.0
    assert row["ratio"] == 2.0


def test_compare_report_to_markdown_and_json():
    report = CompareReport(
        workload=Workload(prompt="hi"),
        results=[
            CacheResult(
                name="full",
                memory_bytes=1000,
                compression_ratio=1.0,
                perplexity=10.0,
                prefill_ms=1.0,
                decode_ms=2.0,
                tokens_per_second=50.0,
            ),
            CacheResult(
                name="primekv",
                memory_bytes=250,
                compression_ratio=4.0,
                perplexity=10.5,
                prefill_ms=1.2,
                decode_ms=2.0,
                tokens_per_second=50.0,
            ),
        ],
        model_name="gpt2",
    )
    md = report.to_markdown()
    # Header row + separator + 2 data rows.
    assert md.count("\n") == 3
    assert "full" in md
    assert "primekv" in md

    d = report.to_dict()
    # Must be JSON-serializable.
    json.dumps(d)
    assert d["model_name"] == "gpt2"
    assert len(d["results"]) == 2


def test_compare_report_empty_is_safe():
    r = CompareReport(workload=Workload(prompt="x"), results=[])
    assert "no results" in r.to_markdown()
    assert r.as_rows() == []


def test_cache_protocol_is_satisfied_by_baselines():
    from primekv.baselines import FullCache, H2OCache, StreamingLLMCache, UniformQuantCache
    from primekv.cache import PrimeKVCache
    from primekv.classifier import RuleBasedClassifier
    from primekv.eval import CacheProtocol

    candidates = [
        FullCache(1),
        H2OCache(1, capacity=4),
        StreamingLLMCache(1, num_sinks=1, window=4),
        UniformQuantCache(1, bits=8),
        PrimeKVCache(num_layers=1, classifier=RuleBasedClassifier()),
    ]
    for c in candidates:
        assert isinstance(c, CacheProtocol), f"{type(c).__name__} does not satisfy CacheProtocol"
