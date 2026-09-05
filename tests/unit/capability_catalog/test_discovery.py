# © Artur Czarnecki. All rights reserved.

"""CAPABILITY-CATALOG-1 Stage 3 discovery filtering tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from intergrax.capability_catalog import (
    CapabilityCatalogDiscoveryError,
    CapabilityCatalogEntry,
    CapabilityCatalogSnapshot,
    CapabilityDiscoveryCandidate,
    discover_capability_candidates,
)
from intergrax.contracts.capability_catalog import (
    AvailabilityDisposition,
    CapabilityDiscoveryAvailabilityEvidence,
    CapabilityDiscoveryQuery,
    CapabilityDiscoveryScope,
    CapabilityDiscoveryScopeMode,
    CapabilityDiscoveryIdentity,
    CapabilityIdentityKey,
    CapabilityKind,
    CapabilityLogicalIdentity,
    CapabilityProvenance,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
    LogicalIdentityFilter,
    SourceFilter,
)

pytestmark = pytest.mark.unit


def _source(source_id: str = "official.catalog") -> CapabilitySourceIdentity:
    return CapabilitySourceIdentity(
        source_id=source_id,
        source_kind=CapabilitySourceKind.OFFICIAL,
    )


def _entry(
    *,
    kind: CapabilityKind = CapabilityKind.TOOL,
    source_id: str = "official.catalog",
    logical_id: str = "tools.echo.ping",
) -> CapabilityCatalogEntry:
    source = _source(source_id)
    return CapabilityCatalogEntry(
        identity=CapabilityDiscoveryIdentity(
            kind=kind,
            source=source,
            logical=CapabilityLogicalIdentity(kind=kind, logical_id=logical_id),
        ),
        provenance=CapabilityProvenance(source=source, version_label="1.0.0"),
        display_label=logical_id,
    )


def _snapshot(*entries: CapabilityCatalogEntry) -> CapabilityCatalogSnapshot:
    ordered = tuple(sorted(entries, key=lambda entry: entry.identity.sort_key))
    return CapabilityCatalogSnapshot(
        source_ids=("official.catalog",),
        entries=ordered,
    )


def _enterprise_scope() -> CapabilityDiscoveryScope:
    return CapabilityDiscoveryScope(
        organization_id="org-acme",
        tenant_id="tenant-a",
        application_id="app-research",
    )


def _identity_key(entry: CapabilityCatalogEntry) -> CapabilityIdentityKey:
    return CapabilityIdentityKey.from_discovery_identity(entry.identity)


def _enterprise_evidence(
    *,
    visible: tuple[CapabilityCatalogEntry, ...],
    host: tuple[CapabilityCatalogEntry, ...] = (),
    blocked: tuple[CapabilityCatalogEntry, ...] = (),
    unavailable: tuple[CapabilityCatalogEntry, ...] = (),
) -> CapabilityDiscoveryAvailabilityEvidence:
    return CapabilityDiscoveryAvailabilityEvidence(
        scope_visible_keys=tuple(_identity_key(entry) for entry in visible),
        host_available_keys=tuple(_identity_key(entry) for entry in host),
        blocked_keys=tuple(_identity_key(entry) for entry in blocked),
        unavailable_keys=tuple(_identity_key(entry) for entry in unavailable),
    )


def test_filter_agent_skill_tool_kinds() -> None:
    agent = _entry(kind=CapabilityKind.AGENT, logical_id="agents.researcher")
    skill = _entry(kind=CapabilityKind.SKILL, logical_id="skills.browser")
    tool = _entry(kind=CapabilityKind.TOOL, logical_id="tools.echo.ping")
    snapshot = _snapshot(agent, skill, tool)
    evidence = _enterprise_evidence(visible=(agent, skill, tool))

    agent_only = discover_capability_candidates(
        snapshot,
        CapabilityDiscoveryQuery(scope=_enterprise_scope(), kinds=(CapabilityKind.AGENT,)),
        availability_evidence=evidence,
    )
    assert [candidate.identity.kind for candidate in agent_only] == [CapabilityKind.AGENT]

    skill_only = discover_capability_candidates(
        snapshot,
        CapabilityDiscoveryQuery(scope=_enterprise_scope(), kinds=(CapabilityKind.SKILL,)),
        availability_evidence=evidence,
    )
    assert [candidate.identity.kind for candidate in skill_only] == [CapabilityKind.SKILL]

    tool_only = discover_capability_candidates(
        snapshot,
        CapabilityDiscoveryQuery(scope=_enterprise_scope(), kinds=(CapabilityKind.TOOL,)),
        availability_evidence=evidence,
    )
    assert [candidate.identity.kind for candidate in tool_only] == [CapabilityKind.TOOL]


def test_filter_multiple_kinds_preserves_snapshot_order() -> None:
    agent = _entry(kind=CapabilityKind.AGENT, logical_id="agents.researcher")
    skill = _entry(kind=CapabilityKind.SKILL, logical_id="skills.browser")
    tool = _entry(kind=CapabilityKind.TOOL, logical_id="tools.echo.ping")
    snapshot = _snapshot(tool, agent, skill)
    evidence = _enterprise_evidence(visible=(agent, skill, tool))

    candidates = discover_capability_candidates(
        snapshot,
        CapabilityDiscoveryQuery(
            scope=_enterprise_scope(),
            kinds=(CapabilityKind.AGENT, CapabilityKind.SKILL, CapabilityKind.TOOL),
        ),
        availability_evidence=evidence,
    )
    assert [candidate.identity.logical.logical_id for candidate in candidates] == [
        "agents.researcher",
        "skills.browser",
        "tools.echo.ping",
    ]


def test_source_filter_by_source_id_and_kind() -> None:
    official = _entry(source_id="official.catalog", logical_id="tools.official")
    private_source = CapabilitySourceIdentity(
        source_id="enterprise.private",
        source_kind=CapabilitySourceKind.ENTERPRISE_PRIVATE,
    )
    private = CapabilityCatalogEntry(
        identity=CapabilityDiscoveryIdentity(
            kind=CapabilityKind.TOOL,
            source=private_source,
            logical=CapabilityLogicalIdentity(
                kind=CapabilityKind.TOOL,
                logical_id="tools.private",
            ),
        ),
        provenance=CapabilityProvenance(source=private_source, version_label="1.0.0"),
        display_label="tools.private",
    )
    snapshot = _snapshot(official, private)
    evidence = _enterprise_evidence(visible=(official, private))

    by_source = discover_capability_candidates(
        snapshot,
        CapabilityDiscoveryQuery(
            scope=_enterprise_scope(),
            source=SourceFilter(source_ids=("enterprise.private",)),
        ),
        availability_evidence=evidence,
    )
    assert len(by_source) == 1
    assert by_source[0].identity.source.source_id == "enterprise.private"

    by_kind = discover_capability_candidates(
        snapshot,
        CapabilityDiscoveryQuery(
            scope=_enterprise_scope(),
            source=SourceFilter(source_kinds=(CapabilitySourceKind.OFFICIAL,)),
        ),
        availability_evidence=evidence,
    )
    assert len(by_kind) == 1
    assert by_kind[0].identity.source.source_kind is CapabilitySourceKind.OFFICIAL


def test_logical_identity_exact_and_prefix_filters() -> None:
    ping = _entry(logical_id="tools.echo.ping")
    search = _entry(logical_id="tools.rag.search")
    snapshot = _snapshot(ping, search)
    evidence = _enterprise_evidence(visible=(ping, search))

    exact = discover_capability_candidates(
        snapshot,
        CapabilityDiscoveryQuery(
            scope=_enterprise_scope(),
            logical_identity=LogicalIdentityFilter(exact_logical_ids=("tools.echo.ping",)),
        ),
        availability_evidence=evidence,
    )
    assert len(exact) == 1
    assert exact[0].identity.logical.logical_id == "tools.echo.ping"

    prefix = discover_capability_candidates(
        snapshot,
        CapabilityDiscoveryQuery(
            scope=_enterprise_scope(),
            logical_identity=LogicalIdentityFilter(logical_id_prefixes=("tools.rag.",)),
        ),
        availability_evidence=evidence,
    )
    assert len(prefix) == 1
    assert prefix[0].identity.logical.logical_id == "tools.rag.search"


def test_no_match_returns_empty_tuple() -> None:
    entry = _entry(logical_id="tools.echo.ping")
    snapshot = _snapshot(entry)
    evidence = _enterprise_evidence(visible=(entry,))

    candidates = discover_capability_candidates(
        snapshot,
        CapabilityDiscoveryQuery(
            scope=_enterprise_scope(),
            kinds=(CapabilityKind.AGENT,),
        ),
        availability_evidence=evidence,
    )
    assert candidates == ()


def test_discovery_is_deterministic_for_same_inputs() -> None:
    entries = (
        _entry(kind=CapabilityKind.AGENT, logical_id="agents.a"),
        _entry(kind=CapabilityKind.SKILL, logical_id="skills.b"),
        _entry(kind=CapabilityKind.TOOL, logical_id="tools.c"),
    )
    snapshot = _snapshot(*entries)
    query = CapabilityDiscoveryQuery(scope=_enterprise_scope())
    evidence = _enterprise_evidence(visible=entries)

    first = discover_capability_candidates(snapshot, query, availability_evidence=evidence)
    second = discover_capability_candidates(snapshot, query, availability_evidence=evidence)
    assert first == second


def test_snapshot_entry_order_independent_of_input_order() -> None:
    a = _entry(kind=CapabilityKind.AGENT, logical_id="agents.a")
    b = _entry(kind=CapabilityKind.SKILL, logical_id="skills.b")
    c = _entry(kind=CapabilityKind.TOOL, logical_id="tools.c")
    snapshot_ab = _snapshot(a, b, c)
    snapshot_cb = _snapshot(c, b, a)
    query = CapabilityDiscoveryQuery(scope=_enterprise_scope())
    evidence = _enterprise_evidence(visible=(a, b, c))

    assert discover_capability_candidates(
        snapshot_ab,
        query,
        availability_evidence=evidence,
    ) == discover_capability_candidates(
        snapshot_cb,
        query,
        availability_evidence=evidence,
    )


def test_enterprise_scope_requires_scope_visible_evidence() -> None:
    snapshot = _snapshot(_entry())
    query = CapabilityDiscoveryQuery(scope=_enterprise_scope())
    with pytest.raises(
        CapabilityCatalogDiscoveryError,
        match="enterprise discovery requires scope_visible_keys",
    ):
        discover_capability_candidates(snapshot, query)


def test_scope_visibility_differs_between_tenants_via_evidence() -> None:
    tenant_a_tool = _entry(logical_id="tools.tenant-a")
    tenant_b_tool = _entry(logical_id="tools.tenant-b")
    snapshot = _snapshot(tenant_a_tool, tenant_b_tool)

    tenant_a_scope = CapabilityDiscoveryScope(
        organization_id="org-acme",
        tenant_id="tenant-a",
        application_id="app-research",
    )
    tenant_b_scope = CapabilityDiscoveryScope(
        organization_id="org-acme",
        tenant_id="tenant-b",
        application_id="app-research",
    )

    tenant_a_candidates = discover_capability_candidates(
        snapshot,
        CapabilityDiscoveryQuery(scope=tenant_a_scope),
        availability_evidence=_enterprise_evidence(visible=(tenant_a_tool,)),
    )
    tenant_b_candidates = discover_capability_candidates(
        snapshot,
        CapabilityDiscoveryQuery(scope=tenant_b_scope),
        availability_evidence=_enterprise_evidence(visible=(tenant_b_tool,)),
    )

    assert len(tenant_a_candidates) == 2
    assert tenant_a_candidates[0].availability is AvailabilityDisposition.CATALOG_AVAILABLE
    assert tenant_a_candidates[0].identity.logical.logical_id == "tools.tenant-a"
    assert tenant_a_candidates[1].availability is AvailabilityDisposition.SCOPE_UNAVAILABLE

    assert len(tenant_b_candidates) == 2
    assert tenant_b_candidates[1].availability is AvailabilityDisposition.CATALOG_AVAILABLE
    assert tenant_b_candidates[1].identity.logical.logical_id == "tools.tenant-b"


def test_global_scope_does_not_require_scope_visible_evidence() -> None:
    entry = _entry()
    snapshot = _snapshot(entry)
    candidates = discover_capability_candidates(
        snapshot,
        CapabilityDiscoveryQuery(scope=CapabilityDiscoveryScope(mode=CapabilityDiscoveryScopeMode.GLOBAL)),
    )
    assert len(candidates) == 1
    assert candidates[0].availability is AvailabilityDisposition.CATALOG_AVAILABLE


def test_availability_dispositions_are_surfaced() -> None:
    catalog = _entry(logical_id="tools.catalog")
    host = _entry(logical_id="tools.host")
    blocked = _entry(logical_id="tools.blocked")
    unavailable = _entry(logical_id="tools.unavailable")
    out_of_scope = _entry(logical_id="tools.out-of-scope")
    snapshot = _snapshot(catalog, host, blocked, unavailable, out_of_scope)
    evidence = CapabilityDiscoveryAvailabilityEvidence(
        scope_visible_keys=tuple(
            _identity_key(entry)
            for entry in (catalog, host, blocked, unavailable)
        ),
        host_available_keys=(_identity_key(host),),
        blocked_keys=(_identity_key(blocked),),
        unavailable_keys=(_identity_key(unavailable),),
    )

    candidates = discover_capability_candidates(
        snapshot,
        CapabilityDiscoveryQuery(scope=_enterprise_scope()),
        availability_evidence=evidence,
    )
    by_id = {candidate.identity.logical.logical_id: candidate.availability for candidate in candidates}

    assert by_id["tools.catalog"] is AvailabilityDisposition.CATALOG_AVAILABLE
    assert by_id["tools.host"] is AvailabilityDisposition.HOST_AVAILABLE
    assert by_id["tools.blocked"] is AvailabilityDisposition.BLOCKED
    assert by_id["tools.unavailable"] is AvailabilityDisposition.UNAVAILABLE
    assert by_id["tools.out-of-scope"] is AvailabilityDisposition.SCOPE_UNAVAILABLE


def test_empty_result_differs_from_blocked_candidate() -> None:
    blocked = _entry(logical_id="tools.blocked")
    snapshot = _snapshot(blocked)
    evidence = _enterprise_evidence(visible=(blocked,), blocked=(blocked,))

    all_candidates = discover_capability_candidates(
        snapshot,
        CapabilityDiscoveryQuery(scope=_enterprise_scope()),
        availability_evidence=evidence,
    )
    assert len(all_candidates) == 1
    assert all_candidates[0].availability is AvailabilityDisposition.BLOCKED

    available_only = discover_capability_candidates(
        snapshot,
        CapabilityDiscoveryQuery(
            scope=_enterprise_scope(),
            availability_constraints=(
                AvailabilityDisposition.CATALOG_AVAILABLE,
                AvailabilityDisposition.HOST_AVAILABLE,
            ),
        ),
        availability_evidence=evidence,
    )
    assert available_only == ()


def test_availability_constraint_filter() -> None:
    host = _entry(logical_id="tools.host")
    catalog = _entry(logical_id="tools.catalog")
    snapshot = _snapshot(catalog, host)
    evidence = _enterprise_evidence(visible=(catalog, host), host=(host,))

    host_only = discover_capability_candidates(
        snapshot,
        CapabilityDiscoveryQuery(
            scope=_enterprise_scope(),
            availability_constraints=(AvailabilityDisposition.HOST_AVAILABLE,),
        ),
        availability_evidence=evidence,
    )
    assert len(host_only) == 1
    assert host_only[0].identity.logical.logical_id == "tools.host"


def test_candidate_preserves_identity_and_provenance() -> None:
    entry = _entry(logical_id="tools.echo.ping")
    snapshot = _snapshot(entry)
    evidence = _enterprise_evidence(visible=(entry,), host=(entry,))

    candidate = discover_capability_candidates(
        snapshot,
        CapabilityDiscoveryQuery(scope=_enterprise_scope()),
        availability_evidence=evidence,
    )[0]

    assert candidate.identity == entry.identity
    assert candidate.provenance == entry.provenance
    assert candidate.catalog_entry is entry
    assert isinstance(candidate, CapabilityDiscoveryCandidate)


def test_candidate_and_snapshot_are_immutable() -> None:
    entry = _entry()
    snapshot = _snapshot(entry)
    evidence = _enterprise_evidence(visible=(entry,))
    candidate = discover_capability_candidates(
        snapshot,
        CapabilityDiscoveryQuery(scope=_enterprise_scope()),
        availability_evidence=evidence,
    )[0]

    with pytest.raises(ValidationError):
        candidate.availability = AvailabilityDisposition.BLOCKED
    with pytest.raises(ValidationError):
        snapshot.entries = ()
