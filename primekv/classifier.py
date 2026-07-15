"""Tier classifier.

Assigns each token in a sequence to one of four structural tiers:

    Tier 0 (Anchor)    : system prompt, attention sinks, format tokens
    Tier 1 (Semantic)  : entities, facts, constraints, reasoning-critical
    Tier 2 (Supporting): elaboration, examples, transitions
    Tier 3 (Filler)    : articles, connectives, boilerplate

Classification happens once at prefill time. The cache may later promote or
demote individual entries based on observed attention — see
``primekv.cache.PrimeKVCache``.

The first-pass classifier is deliberately rule-based so that we can exercise
the rest of the pipeline before training anything. ``MLPClassifier`` is a
thin learned alternative that consumes either token embeddings or an
early-layer hidden state.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Tier(IntEnum):
    """Structural tier for a KV entry.

    We use an ``IntEnum`` so tiers compose cleanly with tensor indexing.
    """

    ANCHOR = 0
    SEMANTIC = 1
    SUPPORTING = 2
    FILLER = 3


NUM_TIERS = 4


@dataclass
class TierAssignment:
    """Result of classifying a sequence.

    Attributes:
        tiers: LongTensor of shape ``(seq_len,)`` with values in ``[0, 3]``.
        logits: Optional FloatTensor of shape ``(seq_len, NUM_TIERS)``. Only
            populated by learned classifiers; rule-based classifiers may set
            this to ``None``.
        meta: Free-form dict for logging/debugging (e.g. which rule fired).
    """

    tiers: torch.Tensor
    logits: Optional[torch.Tensor] = None
    meta: Optional[dict] = None

    def counts(self) -> dict[Tier, int]:
        c: dict[Tier, int] = {t: 0 for t in Tier}
        for t in Tier:
            c[t] = int((self.tiers == int(t)).sum().item())
        return c


class BaseClassifier(nn.Module):
    """Interface every tier classifier must implement.

    The classifier receives one or both of:

    * ``input_ids``: LongTensor ``(seq_len,)`` — raw tokens.
    * ``hidden_states``: FloatTensor ``(seq_len, d_model)`` — embeddings or
      an early-layer hidden state from the base LM.

    It returns a :class:`TierAssignment`. Subclasses should override
    ``classify``; ``forward`` is kept for ``nn.Module`` compatibility but by
    default just delegates.
    """

    def classify(
        self,
        input_ids: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> TierAssignment:
        raise NotImplementedError

    def forward(self, *args, **kwargs) -> TierAssignment:  # noqa: D401
        return self.classify(*args, **kwargs)


class RuleBasedClassifier(BaseClassifier):
    """Cheap heuristic classifier used for bootstrapping.

    Rules (in order):

    1. The first ``anchor_prefix_len`` tokens are Tier 0 (anchor / sink).
    2. Tokens whose id is in ``anchor_token_ids`` are Tier 0 (format / BOS).
    3. Tokens whose id is in ``filler_token_ids`` are Tier 3.
    4. Tokens at positions that are multiples of ``semantic_stride`` are
       Tier 1 (cheap proxy for "this token carries content").
    5. Everything else is Tier 2.

    None of this is principled — it's a placeholder to let the rest of the
    stack run end-to-end while we train a real classifier.
    """

    def __init__(
        self,
        anchor_prefix_len: int = 4,
        anchor_token_ids: Optional[list[int]] = None,
        filler_token_ids: Optional[list[int]] = None,
        semantic_stride: int = 3,
    ) -> None:
        super().__init__()
        self.anchor_prefix_len = anchor_prefix_len
        self.anchor_token_ids = set(anchor_token_ids or [])
        self.filler_token_ids = set(filler_token_ids or [])
        self.semantic_stride = max(1, semantic_stride)

    def classify(
        self,
        input_ids: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> TierAssignment:
        if input_ids is None and hidden_states is None:
            raise ValueError("RuleBasedClassifier needs input_ids or hidden_states")
        if input_ids is not None:
            seq_len = int(input_ids.shape[-1])
            device = input_ids.device
        else:
            seq_len = int(hidden_states.shape[-2])
            device = hidden_states.device

        tiers = torch.full((seq_len,), int(Tier.SUPPORTING), dtype=torch.long, device=device)

        # Rule 1: anchor prefix.
        anchor_end = min(self.anchor_prefix_len, seq_len)
        tiers[:anchor_end] = int(Tier.ANCHOR)

        # Rule 4: semantic stride.
        if self.semantic_stride > 1:
            idx = torch.arange(seq_len, device=device)
            mask = (idx % self.semantic_stride == 0) & (tiers != int(Tier.ANCHOR))
            tiers[mask] = int(Tier.SEMANTIC)

        # Rules 2 and 3 operate on token ids.
        if input_ids is not None and (self.anchor_token_ids or self.filler_token_ids):
            ids = input_ids.view(-1).tolist()
            for pos, tok in enumerate(ids):
                if tok in self.anchor_token_ids:
                    tiers[pos] = int(Tier.ANCHOR)
                elif tok in self.filler_token_ids and tiers[pos] != int(Tier.ANCHOR):
                    tiers[pos] = int(Tier.FILLER)

        return TierAssignment(tiers=tiers, logits=None, meta={"classifier": "rule"})


class MLPClassifier(BaseClassifier):
    """Learned tier head over token embeddings or an early hidden state.

    Deliberately minimal — two-layer MLP. Replace with something smarter
    (attention-pooled, position-aware, etc.) later.
    """

    def __init__(self, d_model: int, hidden: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, NUM_TIERS),
        )

    def classify(
        self,
        input_ids: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> TierAssignment:
        if hidden_states is None:
            raise ValueError("MLPClassifier requires hidden_states")
        h = hidden_states
        if h.dim() == 3:  # (batch, seq, d) — squeeze to single sequence.
            if h.shape[0] != 1:
                raise ValueError("MLPClassifier currently assumes batch size 1")
            h = h[0]
        # fp16/bf16 models hand us Half hidden states; the head runs in its
        # own parameter dtype.
        h = h.to(self.net[0].weight.dtype)
        logits = self.net(h)  # (seq_len, NUM_TIERS)
        tiers = logits.argmax(dim=-1)
        return TierAssignment(tiers=tiers, logits=logits, meta={"classifier": "mlp"})

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Standard cross-entropy loss for supervised pretraining of the head."""
        return F.cross_entropy(logits, targets)
