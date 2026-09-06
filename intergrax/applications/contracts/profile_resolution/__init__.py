# © Artur Czarnecki. All rights reserved.

"""Tier-3 profile resolution contracts (P1.1/P1.2)."""

from intergrax.applications.contracts.profile_resolution.activation import (
    ActivateEffectiveProfileRevisionRequest,
    ActiveEffectiveProfileRevisionBinding,
    ActiveEffectiveProfileRevisionCasOutcome,
    ActiveEffectiveProfileRevisionCasResult,
    ActiveEffectiveProfileRevisionStore,
    EffectiveProfileActivationResult,
)
from intergrax.applications.contracts.profile_resolution.decision import (
    DegradedCapability,
    ProfileDependencyFailure,
    ProfileLayerResolution,
    ProfileResolutionDecision,
    ProfileResolutionDecisionKind,
    ProfileResolutionWarning,
)
from intergrax.applications.contracts.profile_resolution.delta import (
    ProfileDelta,
    ProfileFieldUpdate,
    ProfileLayerInput,
)
from intergrax.applications.contracts.profile_resolution.diff import (
    EffectiveProfileDiff,
    ProfileDiffChangeKind,
    ProfileDiffEntry,
    ProfileDiffProvenanceRef,
)
from intergrax.applications.contracts.profile_resolution.errors import (
    EffectiveProfileActivationConflictError,
    EffectiveProfileActivationError,
    EffectiveProfileActivationPersistenceError,
    EffectiveProfileActivationRejectedError,
    EffectiveProfileActivationRevisionNotFoundError,
    EffectiveProfileActivationScopeMismatchError,
    EffectiveProfileRevisionConflictError,
    EffectiveProfileRevisionError,
    MissingActiveEffectiveProfileRevisionError,
    MissingPinnedEffectiveProfileRevisionError,
    ProfileLayerConflictError,
    ProfileOverrideRejectedError,
    ProfileResolutionError,
)
from intergrax.applications.contracts.profile_resolution.execution_binding import (
    EFFECTIVE_PROFILE_REVISION_METADATA_KEY,
    EffectiveProfileExecutionBinding,
    EffectiveProfileExecutionPinningStore,
    EffectiveProfileRevisionCheckpointEvidence,
)
from intergrax.applications.contracts.profile_resolution.layer import (
    CANONICAL_LAYER_ORDER,
    ProfileLayer,
    profile_layer_sort_key,
)
from intergrax.applications.contracts.profile_resolution.resolution import ProfileResolution
from intergrax.applications.contracts.profile_resolution.revision import (
    EffectiveProfileRevision,
    EffectiveProfileRevisionScope,
)
from intergrax.applications.contracts.profile_resolution.revision_id import (
    EffectiveProfileRevisionId,
    mint_effective_profile_revision_id,
    validate_effective_profile_revision_id,
)
from intergrax.applications.contracts.profile_resolution.store import (
    EffectiveProfileRevisionStore,
)

__all__ = [
    "ActivateEffectiveProfileRevisionRequest",
    "ActiveEffectiveProfileRevisionBinding",
    "ActiveEffectiveProfileRevisionCasOutcome",
    "ActiveEffectiveProfileRevisionCasResult",
    "ActiveEffectiveProfileRevisionStore",
    "CANONICAL_LAYER_ORDER",
    "DegradedCapability",
    "EFFECTIVE_PROFILE_REVISION_METADATA_KEY",
    "EffectiveProfileActivationConflictError",
    "EffectiveProfileActivationError",
    "EffectiveProfileActivationPersistenceError",
    "EffectiveProfileActivationRejectedError",
    "EffectiveProfileActivationResult",
    "EffectiveProfileActivationRevisionNotFoundError",
    "EffectiveProfileActivationScopeMismatchError",
    "EffectiveProfileDiff",
    "EffectiveProfileExecutionBinding",
    "EffectiveProfileExecutionPinningStore",
    "EffectiveProfileRevision",
    "EffectiveProfileRevisionCheckpointEvidence",
    "EffectiveProfileRevisionConflictError",
    "EffectiveProfileRevisionError",
    "EffectiveProfileRevisionId",
    "EffectiveProfileRevisionScope",
    "EffectiveProfileRevisionStore",
    "MissingActiveEffectiveProfileRevisionError",
    "MissingPinnedEffectiveProfileRevisionError",
    "ProfileDelta",
    "ProfileDependencyFailure",
    "ProfileDiffChangeKind",
    "ProfileDiffEntry",
    "ProfileDiffProvenanceRef",
    "ProfileFieldUpdate",
    "ProfileLayer",
    "ProfileLayerConflictError",
    "ProfileLayerInput",
    "ProfileLayerResolution",
    "ProfileOverrideRejectedError",
    "ProfileResolution",
    "ProfileResolutionDecision",
    "ProfileResolutionDecisionKind",
    "ProfileResolutionError",
    "ProfileResolutionWarning",
    "mint_effective_profile_revision_id",
    "profile_layer_sort_key",
    "validate_effective_profile_revision_id",
]
