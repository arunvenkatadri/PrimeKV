"""Tests for the MLP-classifier training scaffold.

These tests use a tiny synthetic model that exposes ``output_attentions``
and ``output_hidden_states``, so we can verify the training loop and
label-generation logic without downloading anything.
"""

import torch

from primekv.classifier import MLPClassifier, Tier
from primekv.classifier_training import (
    attention_received_per_position,
    teacher_labels_from_attention,
    train_mlp_classifier_on_prompt,
)


class _AttnOnlyOutput:
    def __init__(self, attentions=None, hidden_states=None):
        self.attentions = attentions
        self.hidden_states = hidden_states


class _TinyAttnModel:
    """Model that returns synthetic attention + hidden_states tensors.

    Attention pattern: first 4 positions are sinks (high attention),
    rest is roughly uniform. Hidden states are deterministic from
    input_ids so the classifier can fit them.
    """

    def __init__(self, d_model=8, num_heads=2, vocab=32):
        self.d_model = d_model
        self.num_heads = num_heads
        self.vocab = vocab

    def __call__(self, input_ids=None, output_attentions=False,
                 output_hidden_states=False, use_cache=False):
        seq_len = input_ids.shape[-1]
        attentions = None
        if output_attentions:
            # 2 layers, batch=1, heads=2.
            attentions = []
            for _ in range(2):
                a = torch.full((1, self.num_heads, seq_len, seq_len), 1e-3)
                # Strong attention to the first 4 positions from everyone.
                a[:, :, :, :4] = 0.5
                # Apply causal mask: zero above the diagonal.
                mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
                a = a * mask.unsqueeze(0).unsqueeze(0)
                # Renormalize rows so each query's attention sums to 1.
                a = a / a.sum(dim=-1, keepdim=True).clamp_min(1e-9)
                attentions.append(a)
            attentions = tuple(attentions)
        hidden_states = None
        if output_hidden_states:
            torch.manual_seed(0)
            base = torch.randn(seq_len, self.d_model)
            # Make hidden states position-correlated so the classifier
            # actually has signal to fit.
            for p in range(seq_len):
                base[p, 0] += p / 10.0
            hidden_states = tuple(base.unsqueeze(0) for _ in range(4))
        return _AttnOnlyOutput(attentions=attentions, hidden_states=hidden_states)


def test_attention_received_curve_peaks_at_sinks():
    model = _TinyAttnModel()
    ids = torch.arange(20).unsqueeze(0)
    attn = attention_received_per_position(model, ids)
    assert attn.shape == (20,)
    # Synthetic sinks at positions 0-3 should win.
    assert attn[:4].mean() > attn[4:].mean()


def test_teacher_labels_force_anchor_prefix_and_bucket_rest():
    model = _TinyAttnModel()
    ids = torch.arange(18).unsqueeze(0)
    labels = teacher_labels_from_attention(model, ids, anchor_prefix_len=4)
    assert labels.shape == (18,)
    # First four are forced ANCHOR.
    assert (labels[:4] == int(Tier.ANCHOR)).all()
    # Remaining labels are drawn from {SEMANTIC, SUPPORTING, FILLER}.
    remainder = labels[4:].unique().tolist()
    assert set(remainder).issubset({int(Tier.SEMANTIC), int(Tier.SUPPORTING), int(Tier.FILLER)})


def test_train_mlp_classifier_runs_end_to_end_and_loss_decreases():
    model = _TinyAttnModel()
    classifier = MLPClassifier(d_model=8, hidden=32)
    ids = torch.arange(18).unsqueeze(0)
    log = train_mlp_classifier_on_prompt(
        classifier, model, ids,
        epochs=30, lr=1e-2, anchor_prefix_len=4, hidden_layer_index=2,
    )
    assert log["n_positions"] == 18
    assert len(log["epoch_losses"]) == 30
    # Loss should monotonically-ish decrease — first should beat last.
    assert log["epoch_losses"][-1] <= log["epoch_losses"][0]
    # Final accuracy is bounded in [0, 1].
    assert 0.0 <= log["final_accuracy"] <= 1.0
