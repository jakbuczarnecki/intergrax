# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Provider-neutral capability acquisition ports for AW-7A."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.autonomous_work.capability_acquisition import (
    ResolvedWorkerCapabilityPolicy,
    WorkerCapabilityAuthorityCompatibility,
    WorkerCapabilityCandidate,
    WorkerCapabilityDiscoveryLayerOutcome,
    WorkerCapabilityDiscoveryRequest,
)
from intergrax.contracts.autonomous_work.ids import WorkerInstanceId
from intergrax.contracts.autonomous_work.profile_reference import (
    CapabilityProfileRef,
    CodecraftProfileRef,
)


class WorkerCapabilityProfileResolutionError(Exception):
    """Capability profile reference could not be resolved fail-closed."""


class WorkerCodecraftProfileResolutionError(Exception):
    """CodeCraft profile reference could not be resolved fail-closed."""


@runtime_checkable
class WorkerCapabilityProfileResolver(Protocol):
    """Resolve ``CapabilityProfileRef`` into immutable acquisition policy."""

    def resolve(self, profile_ref: CapabilityProfileRef) -> ResolvedWorkerCapabilityPolicy:
        """Return policy for ``profile_ref`` or raise resolution error."""
        ...


@runtime_checkable
class WorkerCodecraftProfileResolver(Protocol):
    """Resolve whether CodeCraft profile permits candidate consideration."""

    def is_candidate_consideration_allowed(
        self,
        profile_ref: CodecraftProfileRef,
    ) -> bool:
        """Return True when ephemeral/adaptive generation may be considered."""
        ...


@runtime_checkable
class WorkerToolCapabilityDiscoveryPort(Protocol):
    """Discover Tool candidates via canonical ToolRegistry adapter."""

    def discover(
        self,
        request: WorkerCapabilityDiscoveryRequest,
    ) -> WorkerCapabilityDiscoveryLayerOutcome:
        ...


@runtime_checkable
class WorkerSkillCapabilityDiscoveryPort(Protocol):
    """Discover Skill candidates via canonical SkillRegistry adapter."""

    def discover(
        self,
        request: WorkerCapabilityDiscoveryRequest,
    ) -> WorkerCapabilityDiscoveryLayerOutcome:
        ...


@runtime_checkable
class WorkerIntegrationCapabilityDiscoveryPort(Protocol):
    """Discover Integration candidates via canonical catalog/manifest adapter."""

    def discover(
        self,
        request: WorkerCapabilityDiscoveryRequest,
    ) -> WorkerCapabilityDiscoveryLayerOutcome:
        ...


@runtime_checkable
class WorkerApprovedAlternateDiscoveryPort(Protocol):
    """Discover approved alternate workflow/capability candidates."""

    def discover(
        self,
        request: WorkerCapabilityDiscoveryRequest,
    ) -> WorkerCapabilityDiscoveryLayerOutcome:
        ...


@runtime_checkable
class WorkerConfigurationOpportunityDiscoveryPort(Protocol):
    """Discover existing approved capabilities requiring non-authority configuration."""

    def discover(
        self,
        request: WorkerCapabilityDiscoveryRequest,
    ) -> WorkerCapabilityDiscoveryLayerOutcome:
        ...


@runtime_checkable
class WorkerCapabilityAuthorityCompatibilityPort(Protocol):
    """Assess authority envelope compatibility — never grants authority."""

    def assess(
        self,
        *,
        worker_instance_id: WorkerInstanceId,
        candidate: WorkerCapabilityCandidate,
    ) -> WorkerCapabilityAuthorityCompatibility:
        ...


class UnavailableToolCapabilityDiscovery:
    """Fail-closed tool discovery when domain is unavailable."""

    def discover(
        self,
        request: WorkerCapabilityDiscoveryRequest,
    ) -> WorkerCapabilityDiscoveryLayerOutcome:
        del request
        from intergrax.contracts.autonomous_work.capability_acquisition import (
            CapabilityDiscoveryDisposition,
        )

        return WorkerCapabilityDiscoveryLayerOutcome(
            disposition=CapabilityDiscoveryDisposition.UNAVAILABLE,
        )


class UnavailableSkillCapabilityDiscovery:
    """Fail-closed skill discovery when domain is unavailable."""

    def discover(
        self,
        request: WorkerCapabilityDiscoveryRequest,
    ) -> WorkerCapabilityDiscoveryLayerOutcome:
        del request
        from intergrax.contracts.autonomous_work.capability_acquisition import (
            CapabilityDiscoveryDisposition,
        )

        return WorkerCapabilityDiscoveryLayerOutcome(
            disposition=CapabilityDiscoveryDisposition.UNAVAILABLE,
        )


class UnavailableIntegrationCapabilityDiscovery:
    """Fail-closed integration discovery when domain is unavailable."""

    def discover(
        self,
        request: WorkerCapabilityDiscoveryRequest,
    ) -> WorkerCapabilityDiscoveryLayerOutcome:
        del request
        from intergrax.contracts.autonomous_work.capability_acquisition import (
            CapabilityDiscoveryDisposition,
        )

        return WorkerCapabilityDiscoveryLayerOutcome(
            disposition=CapabilityDiscoveryDisposition.UNAVAILABLE,
        )


class UnavailableApprovedAlternateDiscovery:
    """Fail-closed approved alternate discovery when catalog is absent."""

    def discover(
        self,
        request: WorkerCapabilityDiscoveryRequest,
    ) -> WorkerCapabilityDiscoveryLayerOutcome:
        del request
        from intergrax.contracts.autonomous_work.capability_acquisition import (
            CapabilityDiscoveryDisposition,
        )

        return WorkerCapabilityDiscoveryLayerOutcome(
            disposition=CapabilityDiscoveryDisposition.UNAVAILABLE,
        )


class UnavailableConfigurationOpportunityDiscovery:
    """Fail-closed configuration opportunity discovery when domain is absent."""

    def discover(
        self,
        request: WorkerCapabilityDiscoveryRequest,
    ) -> WorkerCapabilityDiscoveryLayerOutcome:
        del request
        from intergrax.contracts.autonomous_work.capability_acquisition import (
            CapabilityDiscoveryDisposition,
        )

        return WorkerCapabilityDiscoveryLayerOutcome(
            disposition=CapabilityDiscoveryDisposition.UNAVAILABLE,
        )


class StaticWorkerCapabilityProfileResolver:
    """Always return one configured policy — explicit platform default."""

    def __init__(self, policy: ResolvedWorkerCapabilityPolicy) -> None:
        self._policy = policy

    def resolve(self, profile_ref: CapabilityProfileRef) -> ResolvedWorkerCapabilityPolicy:
        if profile_ref != self._policy.profile_ref:
            raise WorkerCapabilityProfileResolutionError(
                f"capability profile unavailable: {profile_ref.profile_id}@"
                f"{profile_ref.version.value}",
            )
        return self._policy


class MappingWorkerCapabilityProfileResolver:
    """Resolve ``CapabilityProfileRef`` from an explicit in-memory mapping."""

    def __init__(
        self,
        policies: dict[tuple[str, int], ResolvedWorkerCapabilityPolicy],
    ) -> None:
        self._policies = dict(policies)

    def resolve(self, profile_ref: CapabilityProfileRef) -> ResolvedWorkerCapabilityPolicy:
        key = (profile_ref.profile_id, profile_ref.version.value)
        policy = self._policies.get(key)
        if policy is None:
            raise WorkerCapabilityProfileResolutionError(
                f"capability profile unavailable: {profile_ref.profile_id}@"
                f"{profile_ref.version.value}",
            )
        return policy


class StaticCodecraftProfileResolver:
    """Deterministic CodeCraft profile gate for tests and wiring."""

    def __init__(self, *, allowed: bool) -> None:
        self._allowed = allowed

    def is_candidate_consideration_allowed(
        self,
        profile_ref: CodecraftProfileRef,
    ) -> bool:
        del profile_ref
        return self._allowed


def permissive_capability_policy(
    profile_ref: CapabilityProfileRef,
) -> ResolvedWorkerCapabilityPolicy:
    """Explicit permissive policy for tests — not a production default."""

    from intergrax.contracts.autonomous_work.capability_acquisition import (
        WorkerAutonomyLevel,
        WorkerCapabilityCandidateKind,
    )

    return ResolvedWorkerCapabilityPolicy(
        profile_ref=profile_ref,
        allowed_candidate_kinds=frozenset(WorkerCapabilityCandidateKind),
        allowed_autonomy_levels=frozenset(WorkerAutonomyLevel),
        allowed_operation_patterns=(),
        generated_capability_allowed=True,
        adaptive_integration_allowed=True,
        durable_change_allowed=True,
    )


class AllowAllAuthorityCompatibilityPort:
    """Deterministic authority compatibility for tests — never grants authority."""

    def assess(
        self,
        *,
        worker_instance_id: WorkerInstanceId,
        candidate: WorkerCapabilityCandidate,
    ) -> WorkerCapabilityAuthorityCompatibility:
        del worker_instance_id, candidate
        return WorkerCapabilityAuthorityCompatibility.COMPATIBLE
