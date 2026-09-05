# © Artur Czarnecki. All rights reserved.

"""Tier-3 profile resolution contracts (P1.1)."""

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
from intergrax.applications.contracts.profile_resolution.errors import (
    ProfileLayerConflictError,
    ProfileOverrideRejectedError,
    ProfileResolutionError,
)
from intergrax.applications.contracts.profile_resolution.layer import (
    CANONICAL_LAYER_ORDER,
    ProfileLayer,
    profile_layer_sort_key,
)
from intergrax.applications.contracts.profile_resolution.resolution import ProfileResolution

__all__ = [
    "CANONICAL_LAYER_ORDER",
    "DegradedCapability",
    "ProfileDelta",
    "ProfileDependencyFailure",
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
    "profile_layer_sort_key",
]
