"""The file the agent edits.

Exports one function: :func:`build_candidate`. The run loop imports
``build_candidate``, hands it a model/tokenizer, and scores whatever
``PrimeKVCache`` comes back. The agent's job is to edit the body of
``build_candidate`` to produce a better candidate.

Rules (mirrored in ``program.md``):

* The function must return a ``PrimeKVCache`` (or anything with the
  :class:`primekv.eval.CacheProtocol` methods).
* ``num_layers`` is set from the base model. Don't hard-code it.
* Do not change the signature of ``build_candidate``.
* You may import anything from ``primekv.*``.
"""

from __future__ import annotations

from primekv.cache import PrimeKVCache, TierPolicy
from primekv.classifier import RuleBasedClassifier, Tier


def build_candidate(num_layers: int, device: str = "cpu") -> PrimeKVCache:
    """Return a PrimeKVCache configuration to evaluate.

    The starting point matches the default policies. The agent should
    rewrite this function to try different ideas — per-tier
    precisions, capacity caps, stride, dynamic reclassification, etc.
    """
    classifier = RuleBasedClassifier(anchor_prefix_len=4, semantic_stride=3)
    policies = {
        Tier.ANCHOR: TierPolicy(precision="fp16", pinned=True, evictable=False),
        Tier.SEMANTIC: TierPolicy(precision="fp16"),
        Tier.SUPPORTING: TierPolicy(precision="int4"),
        Tier.FILLER: TierPolicy(precision="int4", summary=True),
    }
    return PrimeKVCache(
        num_layers=num_layers,
        classifier=classifier,
        policies=policies,
        max_entries_per_tier=None,
        device=device,
        enable_dynamic_reclassification=True,
    )
