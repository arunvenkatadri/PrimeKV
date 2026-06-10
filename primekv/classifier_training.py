"""Teacher-label generation + in-prompt training for the MLP classifier.

The rule classifier in :mod:`primekv.classifier` is purely positional
(anchor prefix, semantic stride). The spaCy variant we tested earlier
under-performed the rule version on perplexity. This module provides
the third option: a tiny learned classifier head trained on weak
labels derived from teacher attention.

The training is intentionally minimal:

* :func:`teacher_labels_from_attention` runs one model forward pass
  with ``output_attentions=True`` and bucket the per-position average
  attention received into four tiers (top quartile → ANCHOR, bottom
  quartile → FILLER). This is a cheap, model-derived proxy for "which
  tokens does the model actually pay attention to."
* :func:`train_mlp_classifier_on_prompt` fits :class:`MLPClassifier`
  to those labels using hidden states from an early layer. One prompt,
  small number of epochs, no validation split. Tests the *mechanism*
  — does an attention-supervised classifier outperform the positional
  one — not "is this a deployable classifier." For production use you
  would pre-train on a corpus, not in-prompt.

Scope guardrails:

* This file does not ship a checkpoint. Every call trains from
  scratch on the prompt you hand it.
* The hidden states used for supervision come from a fixed (early)
  layer of the base model; the classifier doesn't need access to
  later layers' representations, which is what makes prefill-time
  classification cheap in principle.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from primekv.classifier import (
    MLPClassifier,
    NUM_TIERS,
    Tier,
    TierAssignment,
)

log = logging.getLogger("primekv.classifier_training")


# ---------------------------------------------------------------------------
# Teacher labels
# ---------------------------------------------------------------------------


@torch.no_grad()
def attention_received_per_position(
    model,
    input_ids: torch.Tensor,
    layers_to_use: Optional[int] = None,
) -> torch.Tensor:
    """Average attention each position receives across heads, layers, and queriers.

    Returns a 1-D ``(seq_len,)`` float tensor. Causal mask handled by
    normalizing each column by the number of queries that could attend
    to it (positions >= column index).
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    out = model(input_ids=input_ids, output_attentions=True, use_cache=False)
    if out.attentions is None or len(out.attentions) == 0:
        raise RuntimeError("model did not return attention weights — pass output_attentions=True-capable model")

    seq_len = int(input_ids.shape[-1])
    layers = out.attentions if layers_to_use is None else out.attentions[:layers_to_use]
    device = input_ids.device
    attn = torch.zeros(seq_len, dtype=torch.float32, device=device)
    counts = torch.arange(seq_len, 0, -1, dtype=torch.float32, device=device)
    for layer_attn in layers:
        a = layer_attn[0].float().mean(dim=0)  # (seq, seq), averaged over heads
        col_sums = a.sum(dim=0)
        attn += col_sums / counts
    return attn / len(layers)


def teacher_labels_from_attention(
    model,
    input_ids: torch.Tensor,
    anchor_prefix_len: int = 16,
) -> torch.Tensor:
    """Per-position tier labels in ``{0,1,2,3}`` derived from attention.

    Procedure:

    1. The first ``anchor_prefix_len`` tokens are forced to ANCHOR (0).
       They're the model's attention sinks regardless of content.
    2. The remaining positions are ranked by average attention received,
       and bucketed into three equal-sized groups → SEMANTIC (1),
       SUPPORTING (2), FILLER (3) from most to least attended.

    Returns a ``(seq_len,)`` LongTensor of tier indices on
    ``input_ids.device``.
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    seq_len = int(input_ids.shape[-1])
    device = input_ids.device

    attn = attention_received_per_position(model, input_ids)
    labels = torch.full((seq_len,), int(Tier.SUPPORTING), dtype=torch.long, device=device)

    anchor_end = min(anchor_prefix_len, seq_len)
    labels[:anchor_end] = int(Tier.ANCHOR)

    non_anchor_idx = torch.arange(anchor_end, seq_len, device=device)
    if len(non_anchor_idx) == 0:
        return labels

    non_anchor_attn = attn[non_anchor_idx]
    # Sort by attention received, descending — most-attended first.
    order = non_anchor_attn.argsort(descending=True)
    bucket = len(order) // 3
    # SEMANTIC (1), SUPPORTING (2), FILLER (3) by attention rank.
    semantic_pos = non_anchor_idx[order[:bucket]]
    filler_pos = non_anchor_idx[order[2 * bucket:]]
    labels[semantic_pos] = int(Tier.SEMANTIC)
    labels[filler_pos] = int(Tier.FILLER)
    return labels


# ---------------------------------------------------------------------------
# Hidden states for supervision
# ---------------------------------------------------------------------------


@torch.no_grad()
def hidden_states_for_classifier(
    model,
    input_ids: torch.Tensor,
    layer_index: int = 2,
) -> torch.Tensor:
    """Return early-layer hidden states ``(seq_len, d_model)`` for ``input_ids``.

    The classifier is supposed to run at prefill time over a cheap
    representation, so by default we grab the third layer's output.
    Caller can override via ``layer_index``; clamped to the model's
    actual depth.
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    hidden = out.hidden_states
    if hidden is None or len(hidden) == 0:
        raise RuntimeError("model did not return hidden_states")
    layer_index = max(0, min(layer_index, len(hidden) - 1))
    return hidden[layer_index][0]  # (seq_len, d_model)


# ---------------------------------------------------------------------------
# In-prompt training
# ---------------------------------------------------------------------------


def train_mlp_classifier_on_prompt(
    classifier: MLPClassifier,
    model,
    input_ids: torch.Tensor,
    epochs: int = 50,
    lr: float = 1e-3,
    anchor_prefix_len: int = 16,
    hidden_layer_index: int = 2,
    verbose: bool = False,
) -> dict:
    """Fit an :class:`MLPClassifier` to teacher attention labels on one prompt.

    Tests the mechanism — does attention-supervised classification beat
    the positional rule — without committing to a corpus-level pre-train.
    Returns a small training-log dict ``{epoch_losses, final_accuracy}``.
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    labels = teacher_labels_from_attention(model, input_ids, anchor_prefix_len=anchor_prefix_len)
    hidden = hidden_states_for_classifier(model, input_ids, layer_index=hidden_layer_index)

    # Make sure dimensions agree.
    if hidden.shape[0] != labels.shape[0]:
        raise RuntimeError(
            f"hidden_states len ({hidden.shape[0]}) != labels len ({labels.shape[0]})"
        )

    classifier.train()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    epoch_losses: list[float] = []
    for epoch in range(int(epochs)):
        assignment = classifier.classify(hidden_states=hidden.unsqueeze(0))
        if assignment.logits is None:
            raise RuntimeError("classifier did not produce logits — wrong classifier type?")
        loss = classifier.loss(assignment.logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(float(loss.item()))
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            log.info("epoch %d/%d  loss=%.4f", epoch, epochs, float(loss.item()))

    classifier.eval()
    with torch.no_grad():
        final_assignment = classifier.classify(hidden_states=hidden.unsqueeze(0))
        preds = final_assignment.tiers
        accuracy = float((preds == labels).float().mean().item())
        per_tier_accuracy = {}
        for t in Tier:
            mask = labels == int(t)
            if mask.any():
                per_tier_accuracy[t.name] = float((preds[mask] == int(t)).float().mean().item())

    return {
        "epoch_losses": epoch_losses,
        "final_loss": epoch_losses[-1] if epoch_losses else float("nan"),
        "final_accuracy": accuracy,
        "per_tier_accuracy": per_tier_accuracy,
        "n_positions": int(labels.shape[0]),
    }
