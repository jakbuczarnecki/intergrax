# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed domain conflicts for Agent Distribution services (AP-4)."""

from __future__ import annotations


class AgentDistributionError(Exception):
    """Base error for Agent Distribution domain services."""


class AgentPackageTrustError(AgentDistributionError):
    """Malformed trust invocation or unacceptable installation trust evidence."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code


class AgentPackageAttestationError(AgentDistributionError):
    """Malformed attestation verification inputs at the package authenticity boundary."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code


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


class RuntimeMaterializationConflict(AgentDistributionError):
    """Immutable runtime materialization authority conflict for one revision."""


class EffectiveRosterSnapshotConflict(AgentDistributionError):
    """Immutable effective roster snapshot authority conflict for one revision."""


class EffectiveRosterAuthorityError(AgentDistributionError):
    """Effective roster historical authority resolution failure."""


class EffectiveRosterAuthorityNotFound(EffectiveRosterAuthorityError):
    """Canonical effective roster snapshot is missing for one runtime revision."""


class EffectiveRosterAuthorityConflict(EffectiveRosterAuthorityError):
    """Effective roster snapshot fails runtime revision authority validation."""


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


class RuntimeActivationError(AgentDistributionError):
    """Activation orchestration precondition or persistence failure."""


class RuntimeActivationConflict(RuntimeActivationError):
    """Concurrent or stale activation / serving pointer mutation."""


class RuntimeReadinessError(RuntimeActivationError):
    """Candidate deployment failed readiness validation."""


class RuntimeRollbackError(AgentDistributionError):
    """Rollback orchestration precondition or persistence failure."""


class RuntimeDrainError(AgentDistributionError):
    """Drain orchestration failure or timeout outcome."""
