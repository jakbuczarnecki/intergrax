# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog-backed canonical DiscoveryCompletion for worker recovery (UCA-6B)."""

from __future__ import annotations

from datetime import datetime

from intergrax.autonomous_work.capability_catalog_discovery_adapters import (
    CapabilityCatalogDiscoveryDependencies,
    SkillRegistryManifestLookup,
    identity_key_from_entry_identity,
    _run_skill_discovery_layer,
    _run_tool_discovery_layer,
)
from intergrax.autonomous_work.worker_capability_recovery_ports import (
    CanonicalCapabilityDiscoveryPort,
    CanonicalCapabilityDiscoveryRequest,
)
from intergrax.capability_catalog.governed_candidate import GovernedCapabilityCandidate
from intergrax.capability_catalog.work_stage_effective import (
    select_effective_executable_candidates,
)
from intergrax.contracts.autonomous_work.capability_acquisition import (
    WorkerCapabilityDiscoveryRequest,
)
from intergrax.contracts.capability_catalog.discovery_completion import (
    DiscoveryCompletion,
    build_discovery_completion,
)
from intergrax.contracts.capability_catalog.federation import (
    CapabilityCatalogFederationCompleteness,
)
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.capability_catalog.availability import AvailabilityDisposition
from intergrax.skills.registry.runtime import SkillRegistry


def _catalog_only_keys(
    allowed: tuple[GovernedCapabilityCandidate, ...],
    executable: tuple[GovernedCapabilityCandidate, ...],
) -> tuple[CapabilityIdentityKey, ...]:
    executable_keys = frozenset(item.ranked.identity.sort_key for item in executable)
    catalog_only = [
        identity_key_from_entry_identity(candidate.ranked.identity)
        for candidate in allowed
        if candidate.availability is AvailabilityDisposition.CATALOG_AVAILABLE
        and candidate.ranked.identity.sort_key not in executable_keys
    ]
    return tuple(sorted(catalog_only, key=lambda item: item.sort_key))


def aggregate_layer_outcomes_to_discovery_completion(
    *,
    need_id: str,
    discovery_correlation_id: str,
    federation_completeness: CapabilityCatalogFederationCompleteness,
    created_at: datetime,
    layer_outcomes: tuple,
    governed_allowed: tuple[GovernedCapabilityCandidate, ...],
    governed_executable: tuple[GovernedCapabilityCandidate, ...],
) -> DiscoveryCompletion:
    """Map AW catalog layer dispositions to canonical DiscoveryCompletion facts."""
    from intergrax.contracts.autonomous_work.capability_acquisition import (
        CapabilityDiscoveryDisposition,
    )

    conflict = False
    unavailable = False
    governance_blocked = False

    for outcome in layer_outcomes:
        disposition = outcome.disposition
        if disposition is CapabilityDiscoveryDisposition.CONFLICT:
            conflict = True
        elif disposition is CapabilityDiscoveryDisposition.UNAVAILABLE:
            unavailable = True
        elif disposition is CapabilityDiscoveryDisposition.POLICY_BLOCKED:
            governance_blocked = True

    host_keys = tuple(
        sorted(
            (
                identity_key_from_entry_identity(candidate.ranked.identity)
                for candidate in governed_executable
            ),
            key=lambda item: item.sort_key,
        )
    )
    catalog_keys = _catalog_only_keys(governed_allowed, governed_executable)

    return build_discovery_completion(
        need_id=need_id,
        discovery_correlation_id=discovery_correlation_id,
        federation_completeness=federation_completeness,
        created_at=created_at,
        suitable_host_allowed_keys=host_keys,
        suitable_catalog_allowed_keys=catalog_keys,
        governance_blocked=governance_blocked,
        availability_blocked=False,
        scope_unavailable=False,
        unavailable=unavailable,
        conflict=conflict,
    )


class CatalogCanonicalCapabilityDiscoveryService(CanonicalCapabilityDiscoveryPort):
    """Run governed catalog Tool/Skill discovery and emit DiscoveryCompletion."""

    def __init__(
        self,
        *,
        dependencies: CapabilityCatalogDiscoveryDependencies,
        skill_registry: SkillRegistry,
    ) -> None:
        self._dependencies = dependencies
        self._manifest_lookup = SkillRegistryManifestLookup(skill_registry)

    def complete_discovery(
        self,
        request: CanonicalCapabilityDiscoveryRequest,
    ) -> DiscoveryCompletion:
        worker_discovery = WorkerCapabilityDiscoveryRequest(
            need=request.worker_need,
            profile_ref=request.worker_need.capability_profile_ref,
            worker_instance_id=request.worker_need.worker_instance_id,
        )
        tool_layer = _run_tool_discovery_layer(worker_discovery, self._dependencies)
        skill_layer = _run_skill_discovery_layer(
            worker_discovery,
            self._dependencies,
            self._manifest_lookup,
        )
        layer_outcomes = (tool_layer.outcome, skill_layer.outcome)
        merged_allowed = list(tool_layer.governed_allowed)
        merged_allowed.extend(skill_layer.governed_allowed)
        governed_allowed = tuple(merged_allowed)
        governed_executable = select_effective_executable_candidates(governed_allowed)
        need_id = request.capability_need.need_id or need_id_fallback(request)
        return aggregate_layer_outcomes_to_discovery_completion(
            need_id=need_id,
            discovery_correlation_id=request.discovery_correlation_id,
            federation_completeness=self._dependencies.snapshot.federation_completeness,
            created_at=request.requested_at,
            layer_outcomes=layer_outcomes,
            governed_allowed=governed_allowed,
            governed_executable=governed_executable,
        )


def need_id_fallback(request: CanonicalCapabilityDiscoveryRequest) -> str:
    return request.discovery_correlation_id


__all__ = [
    "CatalogCanonicalCapabilityDiscoveryService",
    "aggregate_layer_outcomes_to_discovery_completion",
]
