import torch

from primekv.classifier import MLPClassifier, RuleBasedClassifier, Tier


def test_rule_based_anchor_prefix():
    clf = RuleBasedClassifier(anchor_prefix_len=3, semantic_stride=1000)
    ids = torch.arange(10)
    out = clf.classify(input_ids=ids)
    assert (out.tiers[:3] == int(Tier.ANCHOR)).all()
    # With a huge semantic stride only position 0 would be semantic, but
    # position 0 is already anchor, so the rest fall through to SUPPORTING.
    assert (out.tiers[3:] == int(Tier.SUPPORTING)).all()


def test_rule_based_semantic_stride():
    clf = RuleBasedClassifier(anchor_prefix_len=0, semantic_stride=2)
    ids = torch.arange(6)
    tiers = clf.classify(input_ids=ids).tiers.tolist()
    # Even positions -> semantic, odd -> supporting.
    assert tiers[0] == int(Tier.SEMANTIC)
    assert tiers[1] == int(Tier.SUPPORTING)
    assert tiers[2] == int(Tier.SEMANTIC)


def test_rule_based_filler_tokens():
    clf = RuleBasedClassifier(
        anchor_prefix_len=0,
        semantic_stride=1000,
        filler_token_ids=[42],
    )
    ids = torch.tensor([1, 42, 3])
    tiers = clf.classify(input_ids=ids).tiers.tolist()
    assert tiers[1] == int(Tier.FILLER)


def test_mlp_classifier_shapes():
    clf = MLPClassifier(d_model=16)
    h = torch.randn(5, 16)
    out = clf.classify(hidden_states=h)
    assert out.tiers.shape == (5,)
    assert out.logits.shape == (5, 4)
