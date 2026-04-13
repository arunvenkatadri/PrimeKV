"""PrimeKV: Priority-Managed Inference KV Cache.

A research prototype of a tiered KV cache that classifies entries by
structural role rather than cumulative attention, and applies per-tier
retention, precision, and eviction policies.
"""

from primekv.classifier import (
    Tier,
    TierAssignment,
    BaseClassifier,
    RuleBasedClassifier,
    MLPClassifier,
)
from primekv.cache import PrimeKVCache, CacheEntry, CacheStats
from primekv.quantize import quantize_int8, dequantize_int8, quantize_int4, dequantize_int4
from primekv.eval import (
    CacheProtocol,
    Workload,
    CacheResult,
    CompareReport,
    run_comparison,
)

__all__ = [
    "Tier",
    "TierAssignment",
    "BaseClassifier",
    "RuleBasedClassifier",
    "MLPClassifier",
    "PrimeKVCache",
    "CacheEntry",
    "CacheStats",
    "quantize_int8",
    "dequantize_int8",
    "quantize_int4",
    "dequantize_int4",
    "CacheProtocol",
    "Workload",
    "CacheResult",
    "CompareReport",
    "run_comparison",
]

__version__ = "0.0.1"
