import time

import torch

from primekv.cache import PrimeKVCache
from primekv.classifier import RuleBasedClassifier
from primekv.metrics import (
    LatencyLog,
    compression_ratio,
    perplexity_from_logits,
    summarize_stats,
    tier_distribution,
    timed,
)


def test_perplexity_from_logits_matches_uniform():
    vocab = 10
    logits = torch.zeros(5, vocab)
    targets = torch.randint(0, vocab, (5,))
    ppl = perplexity_from_logits(logits, targets)
    assert abs(ppl - vocab) < 1e-4


def test_compression_ratio_basic():
    assert compression_ratio(1000, 250) == 4.0


def test_summarize_stats_and_tier_distribution_run():
    cache = PrimeKVCache(
        num_layers=1,
        classifier=RuleBasedClassifier(anchor_prefix_len=2, semantic_stride=2),
    )
    cache.classify_prefill(input_ids=torch.arange(6))
    summary = summarize_stats(cache.stats)
    assert "hit_rate" in summary
    dist = tier_distribution(cache)
    assert set(dist.keys()) == {"ANCHOR", "SEMANTIC", "SUPPORTING", "FILLER"}


def test_latency_log_records_duration():
    log = LatencyLog()
    with timed("sleep", log):
        time.sleep(0.01)
    totals = log.totals()
    assert "sleep" in totals
    assert totals["sleep"] >= 0.005
