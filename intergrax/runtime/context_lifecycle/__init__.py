# © Artur Czarnecki. All rights reserved.

"""Public Unified Context Lifecycle runtime contracts and serialization."""

from __future__ import annotations

from intergrax.runtime.context_lifecycle.contracts import (
    ArtifactCompatibilityReason,
    ArtifactCompatibilityResult,
    ArtifactCompatibilityStatus,
    ArtifactCompressionTarget,
    ArtifactCreationCoordinationStatus,
    ArtifactCreationReservation,
    ArtifactLookupKey,
    ArtifactSourceRange,
    ArtifactValidationStatus,
    ArtifactValidationSummary,
    ContextOptimizationDecision,
    ContextOptimizationMode,
    ContextOptimizationPolicy,
    ContextOptimizationReasonCode,
    EphemeralArtifactPersistencePolicy,
    ModelCallExecutionScope,
    OptimizationArtifactType,
    OptimizationExecutionGuard,
    ReusableArtifactStatus,
    ReusableOptimizationArtifact,
)
from intergrax.runtime.context_lifecycle.serialization import (
    artifact_compatibility_result_to_safe_dict,
    artifact_creation_reservation_to_safe_dict,
    artifact_lookup_key_to_canonical_dict,
    compute_artifact_lookup_key_hash,
    context_optimization_policy_to_safe_dict,
    optimization_execution_guard_to_safe_dict,
    reusable_optimization_artifact_to_safe_dict,
)

__all__ = [
    "ArtifactCompatibilityReason",
    "ArtifactCompatibilityResult",
    "ArtifactCompatibilityStatus",
    "ArtifactCompressionTarget",
    "ArtifactCreationCoordinationStatus",
    "ArtifactCreationReservation",
    "ArtifactLookupKey",
    "ArtifactSourceRange",
    "ArtifactValidationStatus",
    "ArtifactValidationSummary",
    "ContextOptimizationDecision",
    "ContextOptimizationMode",
    "ContextOptimizationPolicy",
    "ContextOptimizationReasonCode",
    "EphemeralArtifactPersistencePolicy",
    "ModelCallExecutionScope",
    "OptimizationArtifactType",
    "OptimizationExecutionGuard",
    "ReusableArtifactStatus",
    "ReusableOptimizationArtifact",
    "artifact_compatibility_result_to_safe_dict",
    "artifact_creation_reservation_to_safe_dict",
    "artifact_lookup_key_to_canonical_dict",
    "compute_artifact_lookup_key_hash",
    "context_optimization_policy_to_safe_dict",
    "optimization_execution_guard_to_safe_dict",
    "reusable_optimization_artifact_to_safe_dict",
]
