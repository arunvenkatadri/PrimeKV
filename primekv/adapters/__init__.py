"""Model adapters.

An adapter knows how to take a HuggingFace model, intercept its
``past_key_values``, route them through a :class:`CacheProtocol`, and
reconstruct a (possibly lossy) cache for decode.

Only GPT-2 is implemented today — Llama / Mistral will live here as
separate modules with the same entry points.
"""

from primekv.adapters.gpt2 import run_with_cache, reconstruct_past_kv

__all__ = ["run_with_cache", "reconstruct_past_kv"]
