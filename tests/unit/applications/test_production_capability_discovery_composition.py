# © Artur Czarnecki. All rights reserved.

"""Production STRICT governed-discovery composition tests (CAPABILITY-CATALOG-1 Stage 5)."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.production_capability_discovery_composition import (
    build_production_capability_governance_context,
    consume_governed_discovery_for_downstream,
    discover_rank_and_govern_capabilities,
    resolve_capability_governance_posture,
)
from intergrax.applications._shared.production_capability_governance_evidence import (
    ProductionCapabilityGovernanceEvidenceBundle,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.capability_catalog import (
    CapabilityCatalogEntry,
    CapabilityCatalogSnapshot,
    CapabilityGovernanceError,
    StableIdentityRanker,
)
from intergrax.contracts.capability_catalog import (
    CapabilityAgentGovernanceEvidence,
    CapabilityDiscoveryAvailabilityEvidence,
    CapabilityDiscoveryIdentity,
    CapabilityDiscoveryQuery,
    CapabilityDiscoveryScope,
    CapabilityDiscoveryScopeMode,
    CapabilityGovernancePosture,
    CapabilityGovernanceReasonCode,
    CapabilityIdentityKey,
    CapabilityKind,
    CapabilityLogicalIdentity,
    CapabilityProvenance,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
    CapabilityToolGovernanceEvidence,
    GovernanceDisposition,
)

pytestmark = pytest.mark.unit

_TOOL_SOURCE = CapabilitySourceIdentity(
    source_id="tools.catalog.builtin",
    source_kind=CapabilitySourceKind.BUILTIN,
)
_ALLOWED_TOOL_ID = "tool.echo.ping"
_DENIED_TOOL_ID = "tool.echo.denied"


def _tool_entry(logical_id: str) -> CapabilityCatalogEntry:
    return CapabilityCatalogEntry(
        identity=CapabilityDiscoveryIdentity(
            kind=CapabilityKind.TOOL,
            source=_TOOL_SOURCE,
            logical=CapabilityLogicalIdentity(
                kind=CapabilityKind.TOOL,
                logical_id=logical_id,
            ),
        ),
        provenance=CapabilityProvenance(source=_TOOL_SOURCE),
        display_label=logical_id,
    )


def _identity_key(logical_id: str) -> CapabilityIdentityKey:
    return CapabilityIdentityKey(
        kind=CapabilityKind.TOOL,
        source_id=_TOOL_SOURCE.source_id,
        source_kind=_TOOL_SOURCE.source_kind,
        logical_id=logical_id,
    )


def _strict_environment(*, tool_ids: tuple[str, ...]) -> ApplicationEnvironmentProfile:
    return ApplicationEnvironmentProfile.product_defaults(
        profile_id="production.capability.discovery.strict",
        tool_ids=list(tool_ids),
    )


def _discovery_query() -> CapabilityDiscoveryQuery:
    return CapabilityDiscoveryQuery(
        scope=CapabilityDiscoveryScope(
            organization_id="org.prod",
            tenant_id="tenant.prod",
            application_id="app.prod",
            mode=CapabilityDiscoveryScopeMode.ENTERPRISE,
        ),
    )


def _availability_evidence(
    *entries: CapabilityCatalogEntry,
) -> CapabilityDiscoveryAvailabilityEvidence:
    return CapabilityDiscoveryAvailabilityEvidence(
        scope_visible_keys=tuple(
            CapabilityIdentityKey.from_discovery_identity(entry.identity)
            for entry in entries
        ),
    )


def _snapshot(*entries: CapabilityCatalogEntry) -> CapabilityCatalogSnapshot:
    ordered = tuple(sorted(entries, key=lambda entry: entry.identity.sort_key))
    return CapabilityCatalogSnapshot(
        source_ids=(_TOOL_SOURCE.source_id,),
        entries=ordered,
    )


def _blocked_reason_codes(
    blocked: tuple[object, ...],
) -> set[CapabilityGovernanceReasonCode]:
    codes: set[CapabilityGovernanceReasonCode] = set()
    for item in blocked:
        for evidence in item.evidence:
            codes.add(evidence.reason_code)
    return codes


def test_strict_host_maps_to_strict_governance_posture() -> None:
    strict_env = _strict_environment(tool_ids=(_ALLOWED_TOOL_ID,))
    assert resolve_capability_governance_posture(strict_env) is (
        CapabilityGovernancePosture.STRICT
    )

    lab_env = ApplicationEnvironmentProfile.lab_defaults(
        profile_id="production.capability.discovery.lab",
    )
    assert resolve_capability_governance_posture(lab_env) is (
        CapabilityGovernancePosture.NON_STRICT
    )
    assert lab_env.execution_mode is not ExecutionMode.STRICT


def test_strict_production_empty_evaluators_fail_closed_at_entry_point() -> None:
    allowed_entry = _tool_entry(_ALLOWED_TOOL_ID)
    environment = _strict_environment(tool_ids=(_ALLOWED_TOOL_ID,))
    with pytest.raises(CapabilityGovernanceError, match="requires at least one evaluator"):
        discover_rank_and_govern_capabilities(
            snapshot=_snapshot(allowed_entry),
            query=_discovery_query(),
            availability_evidence=_availability_evidence(allowed_entry),
            environment=environment,
            ranker=StableIdentityRanker(),
            governance_evaluators=(),
        )


def test_strict_production_happy_path_returns_governed_result() -> None:
    allowed_entry = _tool_entry(_ALLOWED_TOOL_ID)
    denied_entry = _tool_entry(_DENIED_TOOL_ID)
    environment = _strict_environment(tool_ids=(_ALLOWED_TOOL_ID,))
    result = discover_rank_and_govern_capabilities(
        snapshot=_snapshot(allowed_entry, denied_entry),
        query=_discovery_query(),
        availability_evidence=_availability_evidence(allowed_entry, denied_entry),
        environment=environment,
        ranker=StableIdentityRanker(),
    )

    assert len(result.allowed) == 1
    assert len(result.blocked) == 1
    assert result.allowed[0].identity.logical.logical_id == _ALLOWED_TOOL_ID
    assert result.blocked[0].identity.logical.logical_id == _DENIED_TOOL_ID
    assert result.allowed[0].ranking_evidence.rank_position > (
        result.blocked[0].ranking_evidence.rank_position
    )
    assert CapabilityGovernanceReasonCode.NOT_ENTITLED in _blocked_reason_codes(
        result.blocked,
    )


def test_blocked_candidates_do_not_reach_downstream_consumer() -> None:
    allowed_entry = _tool_entry(_ALLOWED_TOOL_ID)
    denied_entry = _tool_entry(_DENIED_TOOL_ID)
    environment = _strict_environment(tool_ids=(_ALLOWED_TOOL_ID,))
    result = discover_rank_and_govern_capabilities(
        snapshot=_snapshot(allowed_entry, denied_entry),
        query=_discovery_query(),
        availability_evidence=_availability_evidence(allowed_entry, denied_entry),
        environment=environment,
        ranker=StableIdentityRanker(),
    )

    downstream = consume_governed_discovery_for_downstream(result)
    assert [item.identity.logical.logical_id for item in downstream] == [_ALLOWED_TOOL_ID]
    assert all(
        evidence.disposition is GovernanceDisposition.ALLOWED
        for item in downstream
        for evidence in item.evidence
    )


def test_strict_production_missing_tool_evidence_blocks_tool_candidate() -> None:
    allowed_entry = _tool_entry(_ALLOWED_TOOL_ID)
    environment = _strict_environment(tool_ids=(_ALLOWED_TOOL_ID,))
    result = discover_rank_and_govern_capabilities(
        snapshot=_snapshot(allowed_entry),
        query=_discovery_query(),
        availability_evidence=_availability_evidence(allowed_entry),
        environment=environment,
        ranker=StableIdentityRanker(),
        governance_evidence=ProductionCapabilityGovernanceEvidenceBundle(
            tool_evidence=None,
        ),
    )
    assert not result.allowed
    assert len(result.blocked) == 1
    assert CapabilityGovernanceReasonCode.MISSING_REQUIRED_EVIDENCE in _blocked_reason_codes(
        result.blocked,
    )


def test_strict_production_tool_policy_denial_blocks_candidate() -> None:
    allowed_entry = _tool_entry(_ALLOWED_TOOL_ID)
    environment = _strict_environment(tool_ids=(_ALLOWED_TOOL_ID,))
    result = discover_rank_and_govern_capabilities(
        snapshot=_snapshot(allowed_entry),
        query=_discovery_query(),
        availability_evidence=_availability_evidence(allowed_entry),
        environment=environment,
        ranker=StableIdentityRanker(),
        governance_evidence=ProductionCapabilityGovernanceEvidenceBundle(
            tool_evidence=CapabilityToolGovernanceEvidence(
                denied_keys=(_identity_key(_ALLOWED_TOOL_ID),),
            ),
        ),
    )
    assert not result.allowed
    assert CapabilityGovernanceReasonCode.POLICY_DENIED in _blocked_reason_codes(
        result.blocked,
    )


def test_strict_production_agent_missing_evidence_blocks_agent_candidate() -> None:
    agent_source = CapabilitySourceIdentity(
        source_id="agents.catalog.official",
        source_kind=CapabilitySourceKind.OFFICIAL,
    )
    agent_entry = CapabilityCatalogEntry(
        identity=CapabilityDiscoveryIdentity(
            kind=CapabilityKind.AGENT,
            source=agent_source,
            logical=CapabilityLogicalIdentity(
                kind=CapabilityKind.AGENT,
                logical_id="agent.search.v1",
            ),
        ),
        provenance=CapabilityProvenance(source=agent_source),
        display_label="agent.search.v1",
    )
    environment = _strict_environment(tool_ids=())
    keys = (CapabilityIdentityKey.from_discovery_identity(agent_entry.identity),)
    result = discover_rank_and_govern_capabilities(
        snapshot=CapabilityCatalogSnapshot(
            source_ids=(agent_source.source_id,),
            entries=(agent_entry,),
        ),
        query=_discovery_query(),
        availability_evidence=CapabilityDiscoveryAvailabilityEvidence(
            scope_visible_keys=keys,
        ),
        environment=environment,
        ranker=StableIdentityRanker(),
        governance_evidence=ProductionCapabilityGovernanceEvidenceBundle(
            agent_evidence=None,
        ),
    )
    assert not result.allowed
    assert CapabilityGovernanceReasonCode.MISSING_REQUIRED_EVIDENCE in _blocked_reason_codes(
        result.blocked,
    )


def test_strict_production_skill_missing_evidence_blocks_skill_candidate() -> None:
    skill_source = CapabilitySourceIdentity(
        source_id="skills.catalog.builtin",
        source_kind=CapabilitySourceKind.BUILTIN,
    )
    skill_entry = CapabilityCatalogEntry(
        identity=CapabilityDiscoveryIdentity(
            kind=CapabilityKind.SKILL,
            source=skill_source,
            logical=CapabilityLogicalIdentity(
                kind=CapabilityKind.SKILL,
                logical_id="skills.rag.retrieve",
            ),
        ),
        provenance=CapabilityProvenance(source=skill_source),
        display_label="skills.rag.retrieve",
    )
    environment = _strict_environment(tool_ids=())
    keys = (CapabilityIdentityKey.from_discovery_identity(skill_entry.identity),)
    result = discover_rank_and_govern_capabilities(
        snapshot=CapabilityCatalogSnapshot(
            source_ids=(skill_source.source_id,),
            entries=(skill_entry,),
        ),
        query=_discovery_query(),
        availability_evidence=CapabilityDiscoveryAvailabilityEvidence(
            scope_visible_keys=keys,
        ),
        environment=environment,
        ranker=StableIdentityRanker(),
        governance_evidence=ProductionCapabilityGovernanceEvidenceBundle(
            skill_evidence=None,
        ),
    )
    assert not result.allowed
    assert CapabilityGovernanceReasonCode.MISSING_REQUIRED_EVIDENCE in _blocked_reason_codes(
        result.blocked,
    )


def test_strict_production_agent_trusted_evidence_allows_agent_candidate() -> None:
    agent_source = CapabilitySourceIdentity(
        source_id="agents.catalog.official",
        source_kind=CapabilitySourceKind.OFFICIAL,
    )
    agent_entry = CapabilityCatalogEntry(
        identity=CapabilityDiscoveryIdentity(
            kind=CapabilityKind.AGENT,
            source=agent_source,
            logical=CapabilityLogicalIdentity(
                kind=CapabilityKind.AGENT,
                logical_id="agent.search.v1",
            ),
        ),
        provenance=CapabilityProvenance(source=agent_source),
        display_label="agent.search.v1",
    )
    agent_key = CapabilityIdentityKey.from_discovery_identity(agent_entry.identity)
    environment = _strict_environment(tool_ids=())
    keys = (agent_key,)
    result = discover_rank_and_govern_capabilities(
        snapshot=CapabilityCatalogSnapshot(
            source_ids=(agent_source.source_id,),
            entries=(agent_entry,),
        ),
        query=_discovery_query(),
        availability_evidence=CapabilityDiscoveryAvailabilityEvidence(
            scope_visible_keys=keys,
        ),
        environment=environment,
        ranker=StableIdentityRanker(),
        agent_evidence=CapabilityAgentGovernanceEvidence(trusted_keys=(agent_key,)),
    )
    assert len(result.allowed) == 1
    assert not result.blocked


def test_build_production_capability_governance_context_never_downgrades_strict() -> None:
    environment = _strict_environment(tool_ids=(_ALLOWED_TOOL_ID,))
    context = build_production_capability_governance_context(environment)
    assert context.posture is CapabilityGovernancePosture.STRICT
