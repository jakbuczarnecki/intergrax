# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed domain conflicts for Agent Distribution services (AP-4)."""

from __future__ import annotations


class AgentDistributionError(Exception):
    """Base error for Agent Distribution domain services."""


class AgentPackageTrustError(AgentDistributionError):
    """Malformed trust invocation or unacceptable installation trust evidence."""


class AgentDistributionNotFoundError(AgentDistributionError):
    """Requested durable record does not exist."""


class InstallationLifecycleError(AgentDistributionError):
    """Illegal installation lifecycle transition."""


class InstallationSlotConflict(AgentDistributionError):
    """Concurrent or stale installation slot mutation."""


class BindingRevisionConflict(AgentDistributionError):
    """Optimistic binding revision mismatch."""


class BindingLifecycleError(AgentDistributionError):
    """Illegal binding lifecycle transition."""


class RuntimeRevisionConflict(AgentDistributionError):
    """Concurrent or stale runtime revision mutation."""


class RuntimeRevisionLifecycleError(AgentDistributionError):
    """Illegal runtime revision lifecycle transition."""


class EffectiveRosterConflict(AgentDistributionError):
    """Ambiguous or invalid effective roster merge inputs."""


class DependencySpecificationError(AgentDistributionError):
    """Invalid candidate dependency specification assembly."""


class DependencyResolutionError(AgentDistributionError):
    """Resolver output failed validation or conflict detection."""


class MaterializedRuntimeLockError(AgentDistributionError):
    """Invalid materialized runtime lock assembly."""


class MaterializedRuntimeLockConflict(AgentDistributionError):
    """Lock identity collision with different semantic content."""


class CandidateRuntimeGraphError(AgentDistributionError):
    """Candidate runtime graph failed structural validation gates."""


class MaterializationError(AgentDistributionError):
    """Materialization failed or returned invalid output."""


class MaterializationInputConflict(MaterializationError):
    """Materialization inputs are mutually inconsistent."""


class MaterializationUnsupportedTopology(MaterializationError):
    """Requested materialization topology has no production adapter."""

class MaterializationLockArtifactLocationBlocked(MaterializationError):
    """Lock lacks physical artifact location required for deterministic wheel install."""

    BLOCKER_CODE = "AP-8_BLOCKED_BY_MISSING_LOCK_ARTIFACT_LOCATION_CONTRACT"


class MaterializationLockArtifactIdentityBlocked(MaterializationError):
    """Lock package lacks cryptographic identity required for production install."""

    BLOCKER_CODE = "AP-8_BLOCKED_BY_MISSING_LOCK_ARTIFACT_IDENTITY"
