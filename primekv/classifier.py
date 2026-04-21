"""Tier classifier.

Assigns each token in a sequence to one of four structural tiers:

    Tier 0 (Anchor)    : system prompt, attention sinks, format tokens
    Tier 1 (Semantic)  : entities, facts, constraints, reasoning-critical
    Tier 2 (Supporting): elaboration, examples, transitions
    Tier 3 (Filler)    : articles, connectives, boilerplate

Classification happens once at prefill time. The cache may later promote or
demote individual entries based on observed attention — see
``primekv.cache.PrimeKVCache``.

Three classifiers live here:

* :class:`RuleBasedClassifier` — cheap positional heuristic. The
  bootstrapping classifier; results are honest but unprincipled.
* :class:`MLPClassifier` — two-layer head over hidden states. Placeholder
  for the learned option.
* :class:`SpaCyClassifier` — uses spaCy POS/NER tags to assign tiers
  based on actual linguistic structure. This is the minimum viable test
  of the "structural role > attention magnitude" thesis: tokens that are
  named entities, numbers, or proper nouns go to SEMANTIC; stopwords and
  punctuation go to FILLER; the rest go to SUPPORTING.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Optional

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
        **kwargs,
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
        **kwargs,
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
        **kwargs,
    ) -> TierAssignment:
        if hidden_states is None:
            raise ValueError("MLPClassifier requires hidden_states")
        h = hidden_states
        if h.dim() == 3:  # (batch, seq, d) — squeeze to single sequence.
            if h.shape[0] != 1:
                raise ValueError("MLPClassifier currently assumes batch size 1")
            h = h[0]
        logits = self.net(h)  # (seq_len, NUM_TIERS)
        tiers = logits.argmax(dim=-1)
        return TierAssignment(tiers=tiers, logits=logits, meta={"classifier": "mlp"})

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Standard cross-entropy loss for supervised pretraining of the head."""
        return F.cross_entropy(logits, targets)


class SpaCyClassifier(BaseClassifier):
    """Tier classifier driven by spaCy POS tags and named entities.

    The contract:

    * The caller provides the original ``text`` that was tokenized and
      the HuggingFace ``tokenizer`` used to produce ``input_ids``. That
      lets us map subword tokens to character offsets, then to spaCy
      tokens.
    * Every subword that overlaps a named entity, number, proper noun,
      verb, or noun root goes to ``Tier.SEMANTIC``.
    * Stopwords and punctuation go to ``Tier.FILLER``.
    * The first ``anchor_prefix_len`` tokens remain ``Tier.ANCHOR`` (the
      attention-sink convention is structural, not linguistic).
    * Everything else stays ``Tier.SUPPORTING``.

    When ``text`` and ``tokenizer`` are not supplied to :meth:`classify`,
    falls back to the behavior of :class:`RuleBasedClassifier`. That
    keeps the class usable inside pipelines that only thread
    ``input_ids`` around (e.g. the HF adapter decode loop — there, the
    prefill assignment is what matters).

    Args:
        nlp: a loaded spaCy ``Language`` instance. If ``None``, we try
            to load ``model_name`` at construction time. Raises
            ``ImportError`` if spaCy isn't installed.
        model_name: spaCy model to load if ``nlp`` is not passed.
        anchor_prefix_len: number of leading tokens pinned to ANCHOR.
        semantic_pos: spaCy POS tags that map to SEMANTIC. Defaults
            lean conservative: proper nouns, nouns, numbers, verbs.
        filler_pos: POS tags that map to FILLER.
        entities_are_semantic: when True, any NER span → SEMANTIC,
            overriding POS.
        stopwords_are_filler: when True, spaCy stopwords → FILLER.
    """

    DEFAULT_SEMANTIC_POS = ("PROPN", "NOUN", "NUM", "VERB")
    DEFAULT_FILLER_POS = ("PUNCT", "SPACE", "SYM", "DET", "CCONJ", "SCONJ", "ADP", "PART")

    def __init__(
        self,
        nlp: Optional[Any] = None,
        model_name: str = "en_core_web_sm",
        anchor_prefix_len: int = 4,
        semantic_pos: Optional[tuple[str, ...]] = None,
        filler_pos: Optional[tuple[str, ...]] = None,
        entities_are_semantic: bool = True,
        stopwords_are_filler: bool = True,
    ) -> None:
        super().__init__()
        if nlp is None:
            try:
                import spacy  # type: ignore
            except ImportError as e:
                raise ImportError(
                    "SpaCyClassifier requires spaCy. Install with "
                    "`pip install spacy && python -m spacy download en_core_web_sm`"
                ) from e
            try:
                nlp = spacy.load(model_name)
            except OSError as e:
                raise ImportError(
                    f"spaCy model '{model_name}' is not downloaded. Run: "
                    f"python -m spacy download {model_name}"
                ) from e
        self.nlp = nlp
        self.model_name = model_name
        self.anchor_prefix_len = anchor_prefix_len
        self.semantic_pos = set(semantic_pos or self.DEFAULT_SEMANTIC_POS)
        self.filler_pos = set(filler_pos or self.DEFAULT_FILLER_POS)
        self.entities_are_semantic = entities_are_semantic
        self.stopwords_are_filler = stopwords_are_filler

    def _spacy_char_tiers(self, text: str) -> list[int]:
        """Return per-character tier labels for ``text``.

        Each char index ``i`` maps to ``Tier.value`` of whichever spaCy
        token covers it. Gaps (whitespace between tokens) default to
        SUPPORTING so they don't accidentally become FILLER.
        """
        doc = self.nlp(text)
        char_tiers = [int(Tier.SUPPORTING)] * len(text)

        for tok in doc:
            if tok.pos_ in self.filler_pos or (
                self.stopwords_are_filler and tok.is_stop
            ):
                tier = int(Tier.FILLER)
            elif tok.pos_ in self.semantic_pos:
                tier = int(Tier.SEMANTIC)
            else:
                tier = int(Tier.SUPPORTING)
            start = tok.idx
            end = start + len(tok.text)
            for i in range(start, min(end, len(char_tiers))):
                char_tiers[i] = tier

        if self.entities_are_semantic:
            for ent in doc.ents:
                for i in range(ent.start_char, min(ent.end_char, len(char_tiers))):
                    char_tiers[i] = int(Tier.SEMANTIC)

        return char_tiers

    def classify_text(
        self,
        text: str,
        tokenizer: Any,
        add_special_tokens: bool = False,
        device: Optional[torch.device] = None,
    ) -> TierAssignment:
        """Classify ``text`` using the supplied HF-style tokenizer.

        The tokenizer must support ``return_offsets_mapping=True``. If
        it doesn't (rare — most fast tokenizers do), raise ValueError.
        """
        if not hasattr(tokenizer, "__call__"):
            raise TypeError("tokenizer must be callable (HuggingFace style)")
        enc = tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=add_special_tokens,
            return_tensors=None,
        )
        offsets = enc.get("offset_mapping")
        if offsets is None:
            raise ValueError(
                "tokenizer did not return offset_mapping. SpaCyClassifier "
                "requires a fast tokenizer (use_fast=True)."
            )
        input_ids = enc["input_ids"]
        if isinstance(input_ids[0], list):  # batched
            input_ids = input_ids[0]
            offsets = offsets[0]

        char_tiers = self._spacy_char_tiers(text)
        tiers: list[int] = []
        for start, end in offsets:
            if start == end:
                # Special token or empty span — treat as ANCHOR.
                tiers.append(int(Tier.ANCHOR))
                continue
            span = char_tiers[start:end]
            if not span:
                tiers.append(int(Tier.SUPPORTING))
                continue
            # The strongest signal in the span wins, preferring SEMANTIC
            # > FILLER > SUPPORTING > ANCHOR (anchor is assigned by
            # position, not from text).
            if int(Tier.SEMANTIC) in span:
                tiers.append(int(Tier.SEMANTIC))
            elif int(Tier.FILLER) in span:
                tiers.append(int(Tier.FILLER))
            else:
                tiers.append(int(Tier.SUPPORTING))

        # Apply anchor prefix.
        for i in range(min(self.anchor_prefix_len, len(tiers))):
            tiers[i] = int(Tier.ANCHOR)

        t = torch.tensor(tiers, dtype=torch.long, device=device)
        return TierAssignment(
            tiers=t,
            logits=None,
            meta={"classifier": "spacy", "model": self.model_name, "num_tokens": len(tiers)},
        )

    def classify(
        self,
        input_ids: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        text: Optional[str] = None,
        tokenizer: Optional[Any] = None,
    ) -> TierAssignment:
        """Classify a sequence.

        Preferred path: pass ``text`` and ``tokenizer`` so spaCy sees
        the raw string. If only ``input_ids`` are provided, fall back
        to a rule-based default — we can't do linguistic analysis on
        subword ids alone.
        """
        if text is not None and tokenizer is not None:
            device = None
            if input_ids is not None:
                device = input_ids.device
            elif hidden_states is not None:
                device = hidden_states.device
            return self.classify_text(text=text, tokenizer=tokenizer, device=device)

        # Fallback: no text available.
        if input_ids is None and hidden_states is None:
            raise ValueError(
                "SpaCyClassifier.classify needs either (text, tokenizer) "
                "or input_ids/hidden_states as fallback"
            )
        fallback = RuleBasedClassifier(anchor_prefix_len=self.anchor_prefix_len)
        out = fallback.classify(input_ids=input_ids, hidden_states=hidden_states)
        out.meta = {"classifier": "spacy", "fallback": "rule"}
        return out
