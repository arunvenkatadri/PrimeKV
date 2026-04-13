import torch

from primekv.attention import AttentionConfig, tiered_attention
from primekv.cache import PrimeKVCache
from primekv.classifier import RuleBasedClassifier, Tier


def test_tiered_attention_returns_correct_shape():
    clf = RuleBasedClassifier(anchor_prefix_len=2, semantic_stride=2)
    cache = PrimeKVCache(num_layers=1, classifier=clf, device="cpu")
    ids = torch.arange(6)
    cache.classify_prefill(input_ids=ids)

    num_heads, head_dim = 2, 8
    for pos in range(6):
        k = torch.randn(num_heads, head_dim)
        v = torch.randn(num_heads, head_dim)
        cache.put(0, pos, k, v)

    q = torch.randn(num_heads, head_dim)
    out = tiered_attention(q, layer=0, cache=cache)
    assert out.shape == (num_heads, head_dim)


def test_tiered_attention_skip_tier3_default():
    clf = RuleBasedClassifier(anchor_prefix_len=1, semantic_stride=1000)
    cache = PrimeKVCache(num_layers=1, classifier=clf, device="cpu")
    cache.classify_prefill(input_ids=torch.arange(3))
    num_heads, head_dim = 2, 4
    k = torch.randn(num_heads, head_dim)
    v = torch.randn(num_heads, head_dim)
    cache.put(0, 0, k, v)
    cache.put(0, 1, k, v, tier=Tier.FILLER)
    cache.put(0, 2, k, v, tier=Tier.SEMANTIC)

    cfg = AttentionConfig(skip_tier3=True, use_summary=True)
    out = tiered_attention(torch.randn(num_heads, head_dim), layer=0, cache=cache, config=cfg)
    assert out.shape == (num_heads, head_dim)


def test_tiered_attention_empty_cache_returns_zeros():
    clf = RuleBasedClassifier()
    cache = PrimeKVCache(num_layers=1, classifier=clf, device="cpu")
    cache.classify_prefill(input_ids=torch.arange(2))
    q = torch.randn(2, 4)
    out = tiered_attention(q, layer=0, cache=cache)
    assert torch.all(out == 0)
