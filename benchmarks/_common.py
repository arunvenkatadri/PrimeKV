"""Shared helpers for benchmark scripts.

Kept deliberately tiny so the individual benchmark files stay readable
from top to bottom.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover
    AutoModelForCausalLM = None
    AutoTokenizer = None


DEFAULT_PROMPTS = [
    "The quick brown fox jumps over the lazy dog. " * 8,
    "In a shocking turn of events, scientists have discovered that " * 8,
    "Once upon a time, in a land far, far away, there lived a wise old king " * 8,
]


def load_model(model_name: str, device: str = "cpu"):
    if AutoModelForCausalLM is None:
        raise RuntimeError("transformers not installed; pip install transformers")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    model.to(device)
    return tok, model


def make_run_dir(name: str, root: str = "runs") -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = Path(root) / f"{ts}-{name}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def dump_json(path: Path, obj: dict) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def pick_device(prefer_cuda: bool = True) -> str:
    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def tokenize_prompts(tok, prompts: list[str], max_length: Optional[int] = None) -> torch.Tensor:
    enc = tok(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return enc["input_ids"]
