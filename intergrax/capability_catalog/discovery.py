# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Deterministic discovery filtering over federated snapshots (Stage 3)."""

from __future__ import annotations

from intergrax.capability_catalog.candidate import CapabilityDiscoveryCandidate
from intergrax.capability_catalog.entry import CapabilityCatalogEntry
from intergrax.capability_catalog.errors import CapabilityCatalogDiscoveryError
from intergrax.capability_catalog.snapshot import CapabilityCatalogSnapshot
from intergrax.contracts.capability_catalog.availability import AvailabilityDisposition
from intergrax.contracts.capability_catalog.evidence import (
    CapabilityDiscoveryAvailabilityEvidence,
)
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.capability_catalog.query import (
    CapabilityDiscoveryQuery,
    LogicalIdentityFilter,
    SourceFilter,
)
from intergrax.contracts.capability_catalog.scope import CapabilityDiscoveryScopeMode


def discover_capability_candidates(
    snapshot: CapabilityCatalogSnapshot,
    query: CapabilityDiscoveryQuery,
    *,
    availability_evidence: CapabilityDiscoveryAvailabilityEvidence | None = None,
) -> tuple[CapabilityDiscoveryCandidate, ...]:
    """Filter a Stage-2 snapshot with a typed query — pure and deterministic."""
    evidence = availability_evidence or CapabilityDiscoveryAvailabilityEvidence()
    _validate_enterprise_evidence(query, evidence)
    blocked_keys = _key_set(evidence.blocked_keys)
    unavailable_keys = _key_set(evidence.unavailable_keys)
    host_available_keys = _key_set(evidence.host_available_keys)
    scope_visible_keys = (
        _key_set(evidence.scope_visible_keys)
        if evidence.scope_visible_keys is not None
        else None
    )
    kind_filter = frozenset(query.kinds) if query.kinds else None
    availability_constraints = (
        frozenset(query.availability_constraints)
        if query.availability_constraints
        else None
    )
    candidates: list[CapabilityDiscoveryCandidate] = []
    for entry in snapshot.entries:
        if kind_filter is not None and entry.identity.kind not in kind_filter:
            continue
        if not _matches_logical_identity(entry, query.logical_identity):
            continue
        if not _matches_source(entry, query.source):
            continue
        identity_key = CapabilityIdentityKey.from_discovery_identity(entry.identity)
        disposition = _resolve_availability_disposition(
            identity_key=identity_key,
            scope_mode=query.scope.mode,
            blocked_keys=blocked_keys,
            unavailable_keys=unavailable_keys,
            host_available_keys=host_available_keys,
            scope_visible_keys=scope_visible_keys,
        )
        if availability_constraints is not None and disposition not in availability_constraints:
            continue
        candidates.append(
            CapabilityDiscoveryCandidate(
                catalog_entry=entry,
                availability=disposition,
            ),
        )
    return tuple(candidates)


def _validate_enterprise_evidence(
    query: CapabilityDiscoveryQuery,
    evidence: CapabilityDiscoveryAvailabilityEvidence,
) -> None:
    if query.scope.mode is not CapabilityDiscoveryScopeMode.ENTERPRISE:
        return
    if evidence.scope_visible_keys is None:
        raise CapabilityCatalogDiscoveryError(
            "enterprise discovery requires scope_visible_keys availability evidence",
        )


def _key_set(keys: tuple[CapabilityIdentityKey, ...]) -> frozenset[tuple[str, str, str, str]]:
    return frozenset(key.sort_key for key in keys)


def _matches_logical_identity(
    entry: CapabilityCatalogEntry,
    filter: LogicalIdentityFilter | None,
) -> bool:
    if filter is None:
        return True
    logical_id = entry.identity.logical.logical_id
    if filter.exact_logical_ids and logical_id in filter.exact_logical_ids:
        return True
    if filter.logical_id_prefixes:
        return any(logical_id.startswith(prefix) for prefix in filter.logical_id_prefixes)
    return False


def _matches_source(
    entry: CapabilityCatalogEntry,
    filter: SourceFilter | None,
) -> bool:
    if filter is None:
        return True
    source = entry.identity.source
    if filter.source_ids and source.source_id not in filter.source_ids:
        return False
    if filter.source_kinds and source.source_kind not in filter.source_kinds:
        return False
    return True


def _resolve_availability_disposition(
    *,
    identity_key: CapabilityIdentityKey,
    scope_mode: CapabilityDiscoveryScopeMode,
    blocked_keys: frozenset[tuple[str, str, str, str]],
    unavailable_keys: frozenset[tuple[str, str, str, str]],
    host_available_keys: frozenset[tuple[str, str, str, str]],
    scope_visible_keys: frozenset[tuple[str, str, str, str]] | None,
) -> AvailabilityDisposition:
    # Competing disposition evidence is contractually disjoint before resolution.
    sort_key = identity_key.sort_key
    if sort_key in blocked_keys:
        return AvailabilityDisposition.BLOCKED
    if sort_key in unavailable_keys:
        return AvailabilityDisposition.UNAVAILABLE
    if scope_mode is CapabilityDiscoveryScopeMode.ENTERPRISE:
        if scope_visible_keys is not None and sort_key not in scope_visible_keys:
            return AvailabilityDisposition.SCOPE_UNAVAILABLE
    if sort_key in host_available_keys:
        return AvailabilityDisposition.HOST_AVAILABLE
    return AvailabilityDisposition.CATALOG_AVAILABLE
