# © Artur Czarnecki. All rights reserved.

"""Token Optimization runtime package (Phase TOKEN-1A/TOKEN-1B)."""

from __future__ import annotations

from intergrax.runtime.token_optimization.contracts import (
    CompressionLevel,
    CompressionReceiptRef,
    OutputPolicy,
    OutputProfile,
    ProtectedRegion,
    ProtectedRegionKind,
    ProtectedRegionValidationResult,
    ProtectedRegionValidationStatus,
    StrategySafetyClass,
    TokenCategory,
    TokenOptimizationAttribution,
    TokenOptimizationBypassReason,
    TokenOptimizationDecision,
    TokenOptimizationMechanism,
    TokenOptimizationPluginCapability,
    TokenOptimizationPluginDescriptor,
    TokenOptimizationPolicy,
    TokenOptimizationProfile,
    TokenOptimizationRequest,
    TokenOptimizationResult,
    TokenOptimizationSourceType,
    TokenOptimizationStrategyKind,
    TokenOptimizationStrategyRef,
    TokenSavingsClaimConfidence,
    TokenSavingsMeasurement,
    TokenUsageMeasurement,
)
from intergrax.runtime.token_optimization.protected_regions import (
    detect_protected_regions,
    validate_protected_regions,
)

__all__ = [
    "CompressionLevel",
    "CompressionReceiptRef",
    "detect_protected_regions",
    "OutputPolicy",
    "OutputProfile",
    "ProtectedRegion",
    "ProtectedRegionKind",
    "ProtectedRegionValidationResult",
    "ProtectedRegionValidationStatus",
    "StrategySafetyClass",
    "TokenCategory",
    "TokenOptimizationAttribution",
    "TokenOptimizationBypassReason",
    "TokenOptimizationDecision",
    "TokenOptimizationMechanism",
    "TokenOptimizationPluginCapability",
    "TokenOptimizationPluginDescriptor",
    "TokenOptimizationPolicy",
    "TokenOptimizationProfile",
    "TokenOptimizationRequest",
    "TokenOptimizationResult",
    "TokenOptimizationSourceType",
    "TokenOptimizationStrategyKind",
    "TokenOptimizationStrategyRef",
    "TokenSavingsClaimConfidence",
    "TokenSavingsMeasurement",
    "TokenUsageMeasurement",
    "validate_protected_regions",
]
