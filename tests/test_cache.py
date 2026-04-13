import torch

from primekv.cache import DEFAULT_POLICIES, PrimeKVCache
from primekv.classifier import RuleBasedClassifier, Tier


def _make_cache(num_layers: int = 2) -> PrimeKVCache:
    clf = RuleBasedClassifier(anchor_prefix_len=2, semantic_stride=2)
    return PrimeKVCache(num_layers=num_layers, classifier=clf, device="cpu")


def test_prefill_and_put_get():
    cache = _make_cache()
    ids = torch.arange(8)
    cache.classify_prefill(input_ids=ids)
    k = torch.randn(4, 8)
    v = torch.randn(4, 8)
    cache.put(layer=0, position=0, k=k, v=v)
    got = cache.get(layer=0, position=0)
    assert got is not None
    gk, gv = got
    assert gk.shape == k.shape
    assert gv.shape == v.shape
    assert cache.stats.hits[Tier.ANCHOR] == 1


def test_tier3_folds_into_summary_and_is_not_stored():
    cache = _make_cache()
    ids = torch.arange(4)
    cache.classify_prefill(input_ids=ids)
    k = torch.randn(4, 8)
    v = torch.randn(4, 8)
    # Force a Tier 3 insert directly.
    cache.put(layer=0, position=1, k=k, v=v, tier=Tier.FILLER)
    # Tier 3 uses summary policy in the defaults -> entry should not be
    # stored verbatim; summary_for should return something.
    assert cache.get(layer=0, position=1) is None
    assert cache.summary_for(0) is not None


def test_lru_eviction_respects_cap():
    clf = RuleBasedClassifier(anchor_prefix_len=0, semantic_stride=1000)
    cache = PrimeKVCache(
        num_layers=1,
        classifier=clf,
        max_entries_per_tier={Tier.SUPPORTING: 2},
        device="cpu",
    )
    ids = torch.arange(4)
    cache.classify_prefill(input_ids=ids)
    k = torch.randn(2, 4)
    v = torch.randn(2, 4)
    # Force all four positions into Tier.SUPPORTING so the cap is exercised.
    for pos in range(4):
        cache.put(layer=0, position=pos, k=k, v=v, tier=Tier.SUPPORTING)
    # Only 2 most-recent Tier.SUPPORTING positions should remain.
    assert cache.get(0, 0) is None
    assert cache.get(0, 1) is None
    assert cache.get(0, 2) is not None
    assert cache.get(0, 3) is not None
    assert cache.stats.evictions[Tier.SUPPORTING] == 2


def test_dynamic_reclassification_promotes_tier2_to_tier1():
    clf = RuleBasedClassifier(anchor_prefix_len=0, semantic_stride=1000)
    cache = PrimeKVCache(num_layers=1, classifier=clf, device="cpu")
    ids = torch.arange(3)
    cache.classify_prefill(input_ids=ids)
    # Position 1 starts as SUPPORTING under the default rules.
    k = torch.randn(2, 4)
    v = torch.randn(2, 4)
    cache.put(0, 1, k, v)
    for _ in range(30):
        cache.observe_attention(1, score=1.0)
    # Now it should have been promoted at least once.
    assert cache.stats.promotions[Tier.SEMANTIC] >= 1


def test_memory_bytes_decreases_under_quant():
    clf = RuleBasedClassifier(anchor_prefix_len=0, semantic_stride=1000)
    cache = PrimeKVCache(num_layers=1, classifier=clf, device="cpu")
    ids = torch.arange(1)
    cache.classify_prefill(input_ids=ids)
    k = torch.randn(8, 32)
    v = torch.randn(8, 32)
    # Force tier 2 -> INT4.
    cache.put(0, 0, k, v, tier=Tier.SUPPORTING)
    quant_bytes = cache.memory_bytes()

    cache.reset()
    cache.classify_prefill(input_ids=ids)
    cache.put(0, 0, k, v, tier=Tier.ANCHOR)
    fp_bytes = cache.memory_bytes()
    assert quant_bytes < fp_bytes


def test_default_policies_cover_all_tiers():
    for tier in Tier:
        assert tier in DEFAULT_POLICIES
