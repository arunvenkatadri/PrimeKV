"""Tests for SpaCyClassifier.

We mock the spaCy ``Language`` object so these tests run without the
real spaCy install. A separate conditional test confirms real spaCy
works if it's available, but is skipped otherwise.
"""

from __future__ import annotations

import pytest
import torch

from primekv.classifier import SpaCyClassifier, Tier


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeToken:
    def __init__(self, text: str, idx: int, pos_: str, is_stop: bool = False):
        self.text = text
        self.idx = idx
        self.pos_ = pos_
        self.is_stop = is_stop


class _FakeSpan:
    def __init__(self, start_char: int, end_char: int):
        self.start_char = start_char
        self.end_char = end_char


class _FakeDoc:
    def __init__(self, tokens: list[_FakeToken], ents: list[_FakeSpan] | None = None):
        self._tokens = tokens
        self.ents = ents or []

    def __iter__(self):
        return iter(self._tokens)


class _FakeNLP:
    """Returns a scripted doc per input text."""

    def __init__(self, docs: dict[str, _FakeDoc]):
        self._docs = docs

    def __call__(self, text: str) -> _FakeDoc:
        return self._docs[text]


class _FakeTokenizer:
    """Whitespace tokenizer that emits (id, offsets) like a HF fast tokenizer.

    Enough to exercise the classifier's offset-mapping path without
    depending on HuggingFace. Each word maps to its (start, end) char
    offset in the input string.
    """

    def __call__(
        self,
        text,
        return_offsets_mapping=False,
        add_special_tokens=False,
        return_tensors=None,
    ):
        offsets: list[tuple[int, int]] = []
        ids: list[int] = []
        i = 0
        while i < len(text):
            if text[i].isspace():
                i += 1
                continue
            j = i
            while j < len(text) and not text[j].isspace():
                j += 1
            offsets.append((i, j))
            ids.append(len(ids))
            i = j
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_spacy_classifier_requires_spacy_if_nlp_none():
    # spaCy isn't installed in the test env, so this path must raise.
    try:
        import spacy  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError):
            SpaCyClassifier(nlp=None)


def test_spacy_classifier_classifies_pos_tags():
    text = "Alice loves cats"
    tokens = [
        _FakeToken("Alice", 0, "PROPN"),
        _FakeToken("loves", 6, "VERB"),
        _FakeToken("cats", 12, "NOUN"),
    ]
    doc = _FakeDoc(tokens, ents=[])
    nlp = _FakeNLP({text: doc})
    clf = SpaCyClassifier(nlp=nlp, anchor_prefix_len=0)

    out = clf.classify_text(text, _FakeTokenizer())
    assert out.tiers.tolist() == [
        int(Tier.SEMANTIC),
        int(Tier.SEMANTIC),
        int(Tier.SEMANTIC),
    ]


def test_spacy_classifier_respects_anchor_prefix():
    text = "Alice loves cats"
    tokens = [
        _FakeToken("Alice", 0, "PROPN"),
        _FakeToken("loves", 6, "VERB"),
        _FakeToken("cats", 12, "NOUN"),
    ]
    nlp = _FakeNLP({text: _FakeDoc(tokens, ents=[])})
    clf = SpaCyClassifier(nlp=nlp, anchor_prefix_len=2)
    out = clf.classify_text(text, _FakeTokenizer())
    assert out.tiers[0].item() == int(Tier.ANCHOR)
    assert out.tiers[1].item() == int(Tier.ANCHOR)
    assert out.tiers[2].item() == int(Tier.SEMANTIC)


def test_spacy_classifier_filler_pos_and_stopwords():
    text = "the big dog"
    tokens = [
        _FakeToken("the", 0, "DET", is_stop=True),
        _FakeToken("big", 4, "ADJ"),  # ADJ is not in semantic_pos → SUPPORTING
        _FakeToken("dog", 8, "NOUN"),
    ]
    nlp = _FakeNLP({text: _FakeDoc(tokens, ents=[])})
    clf = SpaCyClassifier(nlp=nlp, anchor_prefix_len=0)
    out = clf.classify_text(text, _FakeTokenizer())
    tiers = out.tiers.tolist()
    assert tiers[0] == int(Tier.FILLER)       # stopword / DET
    assert tiers[1] == int(Tier.SUPPORTING)   # adjective = neither
    assert tiers[2] == int(Tier.SEMANTIC)     # noun


def test_spacy_classifier_entities_override_pos():
    text = "Paris rocks"
    tokens = [
        _FakeToken("Paris", 0, "PROPN"),
        # Even if we marked this as FILLER, the ent span should rescue it.
        _FakeToken("rocks", 6, "DET", is_stop=True),
    ]
    ents = [_FakeSpan(start_char=0, end_char=11)]
    nlp = _FakeNLP({text: _FakeDoc(tokens, ents=ents)})
    clf = SpaCyClassifier(nlp=nlp, anchor_prefix_len=0)
    out = clf.classify_text(text, _FakeTokenizer())
    assert out.tiers.tolist() == [int(Tier.SEMANTIC), int(Tier.SEMANTIC)]


def test_spacy_classifier_falls_back_without_text():
    nlp = _FakeNLP({})
    clf = SpaCyClassifier(nlp=nlp, anchor_prefix_len=2)
    ids = torch.arange(5)
    out = clf.classify(input_ids=ids)
    # Fallback is rule-based — anchor prefix applied, others are SUPPORTING
    # by default.
    assert out.tiers[0].item() == int(Tier.ANCHOR)
    assert out.tiers[1].item() == int(Tier.ANCHOR)
    assert out.meta == {"classifier": "spacy", "fallback": "rule"}


def test_spacy_classifier_rejects_slow_tokenizer():
    """A tokenizer that doesn't return offset_mapping should error."""

    class NoOffsets:
        def __call__(self, text, return_offsets_mapping=False, add_special_tokens=False, return_tensors=None):
            return {"input_ids": [1, 2, 3]}

    tokens = [_FakeToken("a", 0, "NOUN")]
    nlp = _FakeNLP({"a": _FakeDoc(tokens)})
    clf = SpaCyClassifier(nlp=nlp, anchor_prefix_len=0)
    with pytest.raises(ValueError, match="offset_mapping"):
        clf.classify_text("a", NoOffsets())


# ---------------------------------------------------------------------------
# Real-spaCy smoke test (skipped when the library isn't installed)
# ---------------------------------------------------------------------------


def test_real_spacy_smoke():
    spacy = pytest.importorskip("spacy")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip("en_core_web_sm not installed")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2", use_fast=True)
    clf = SpaCyClassifier(nlp=nlp, anchor_prefix_len=1)
    out = clf.classify_text("Alice went to Paris.", tok)
    assert out.tiers.shape[0] > 0
    assert (out.tiers == int(Tier.SEMANTIC)).any()
