# © Artur Czarnecki. All rights reserved.

"""Tier-3 capability dependency validation contracts (P1.3)."""

from intergrax.applications.contracts.capability_dependency.dependency import (
    CapabilityDependency,
    CapabilityDependencyAvailabilityStatus,
    CapabilityDependencyKind,
    CapabilityDependencyRequirement,
    CapabilityRef,
)
from intergrax.applications.contracts.capability_dependency.errors import (
    CapabilityDependencyDeclarationConflictError,
    CapabilityDependencyProviderConflictError,
    CapabilityDependencyValidationError,
    RequiredCapabilityDependencyUnavailableError,
)
from intergrax.applications.contracts.capability_dependency.provider import (
    CapabilityDependencyProvider,
    CapabilityDependencyValidationContext,
)
from intergrax.applications.contracts.capability_dependency.validation import (
    CapabilityDependencyDegradationEvidence,
    CapabilityDependencyEvaluation,
    CapabilityDependencyFailureEvidence,
    CapabilityDependencyOutcome,
    CapabilityDependencyValidationResult,
)

__all__ = [
    "CapabilityDependency",
    "CapabilityDependencyAvailabilityStatus",
    "CapabilityDependencyDeclarationConflictError",
    "CapabilityDependencyProviderConflictError",
    "CapabilityDependencyDegradationEvidence",
    "CapabilityDependencyEvaluation",
    "CapabilityDependencyFailureEvidence",
    "CapabilityDependencyKind",
    "CapabilityDependencyOutcome",
    "CapabilityDependencyProvider",
    "CapabilityDependencyRequirement",
    "CapabilityDependencyValidationContext",
    "CapabilityDependencyValidationError",
    "CapabilityDependencyValidationResult",
    "CapabilityRef",
    "RequiredCapabilityDependencyUnavailableError",
]
