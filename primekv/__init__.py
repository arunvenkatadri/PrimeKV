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
    SpaCyClassifier,
)
from primekv.cache import PrimeKVCache, CacheEntry, CacheStats, TierPolicy, DEFAULT_POLICIES
from primekv.quantize import quantize_int8, dequantize_int8, quantize_int4, dequantize_int4
from primekv.eval import (
    CacheProtocol,
    Workload,
    CacheResult,
    CompareReport,
    run_comparison,
)
from primekv.tuning import (
    TuningProfile,
    auto_tune,
    build_cache_from_profile,
    describe_profile,
    estimate_bytes,
)
from primekv.reasoning_eval import (
    ReasoningTest,
    ReasoningResult,
    ReasoningReport,
    default_tests,
    run_reasoning_eval,
    grade,
)

__all__ = [
    "Tier",
    "TierAssignment",
    "BaseClassifier",
    "RuleBasedClassifier",
    "MLPClassifier",
    "SpaCyClassifier",
    "PrimeKVCache",
    "CacheEntry",
    "CacheStats",
    "TierPolicy",
    "DEFAULT_POLICIES",
    "quantize_int8",
    "dequantize_int8",
    "quantize_int4",
    "dequantize_int4",
    "CacheProtocol",
    "Workload",
    "CacheResult",
    "CompareReport",
    "run_comparison",
    "TuningProfile",
    "auto_tune",
    "build_cache_from_profile",
    "describe_profile",
    "estimate_bytes",
    "ReasoningTest",
    "ReasoningResult",
    "ReasoningReport",
    "default_tests",
    "run_reasoning_eval",
    "grade",
]

__version__ = "0.0.1"
