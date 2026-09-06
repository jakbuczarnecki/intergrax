# © Artur Czarnecki. All rights reserved.

"""CAPABILITY-CATALOG-1 Stage 8 adaptive work-stage discovery tests."""

from __future__ import annotations

import pytest

from intergrax.capability_catalog import (
    AvailabilityPreservingGovernanceEvaluator,
    CapabilityCatalogDiscoveryError,
    CapabilityCatalogEntry,
    CapabilityCatalogSnapshot,
    CapabilityGovernanceError,
    StableIdentityRanker,
    WorkStageCapabilityDiscoveryService,
    compare_work_stage_effective_capabilities,
    discover_effective_capabilities_for_work_stage,
)
from intergrax.capability_catalog.adapters.skill_governance import SkillProfileGovernanceEvaluator
from intergrax.capability_catalog.adapters.tool_governance import ToolPolicyGovernanceEvaluator
from intergrax.contracts.capability_catalog import (
    AvailabilityDisposition,
    CapabilityDiscoveryAvailabilityEvidence,
    CapabilityDiscoveryIdentity,
    CapabilityDiscoveryQuery,
    CapabilityDiscoveryScope,
    CapabilityDiscoveryScopeMode,
    CapabilityGovernanceContext,
    CapabilityGovernancePosture,
    CapabilityIdentityKey,
    CapabilityKind,
    CapabilityLogicalIdentity,
    CapabilityProvenance,
    CapabilitySetConstraintMode,
    CapabilitySkillGovernanceEvidence,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
    CapabilityToolGovernanceEvidence,
    LogicalIdentityFilter,
    WorkStageCapabilityNeed,
)
from intergrax.skills.registry.profile import SkillProfile
from intergrax.skills.registry.runtime import SkillRegistry
from intergrax.tools.registry.profile import ToolProfile
from intergrax.tools.registry.runtime import ToolRegistry

pytestmark = pytest.mark.unit

_TOOL_SOURCE = CapabilitySourceIdentity(
    source_id="tools.catalog.builtin",
    source_kind=CapabilitySourceKind.BUILTIN,
)
_SKILL_SOURCE = CapabilitySourceIdentity(
    source_id="skills.catalog.builtin",
    source_kind=CapabilitySourceKind.BUILTIN,
)
_WORK_REF = "work.incident-42"
_GOAL = "resolve customer incident"
_TOOL_SEARCH = "tool.search.logs"
_TOOL_DENIED = "tool.search.denied"
_SKILL_SUMMARY = "skill.incident.summary"
_SCOPE_EXCLUDED = "tool.search.scope_excluded"


def _enterprise_scope() -> CapabilityDiscoveryScope:
    return CapabilityDiscoveryScope(
        organization_id="org.prod",
        tenant_id="tenant.prod",
        application_id="app.prod",
        mode=CapabilityDiscoveryScopeMode.ENTERPRISE,
    )


def _entry(
    *,
    kind: CapabilityKind,
    logical_id: str,
    source: CapabilitySourceIdentity,
) -> CapabilityCatalogEntry:
    return CapabilityCatalogEntry(
        identity=CapabilityDiscoveryIdentity(
            kind=kind,
            source=source,
            logical=CapabilityLogicalIdentity(kind=kind, logical_id=logical_id),
        ),
        provenance=CapabilityProvenance(source=source),
        display_label=logical_id,
    )


def _identity_key(
    *,
    kind: CapabilityKind,
    logical_id: str,
    source: CapabilitySourceIdentity,
) -> CapabilityIdentityKey:
    return CapabilityIdentityKey(
        kind=kind,
        source_id=source.source_id,
        source_kind=source.source_kind,
        logical_id=logical_id,
    )


def _snapshot(*entries: CapabilityCatalogEntry) -> CapabilityCatalogSnapshot:
    ordered = tuple(sorted(entries, key=lambda entry: entry.identity.sort_key))
    return CapabilityCatalogSnapshot(
        source_ids=tuple(sorted({entry.identity.source.source_id for entry in ordered})),
        entries=ordered,
    )


def _evaluators() -> tuple[
    AvailabilityPreservingGovernanceEvaluator,
    ToolPolicyGovernanceEvaluator,
    SkillProfileGovernanceEvaluator,
]:
    return (
        AvailabilityPreservingGovernanceEvaluator(),
        ToolPolicyGovernanceEvaluator(),
        SkillProfileGovernanceEvaluator(),
    )


def _governance_context(
    *,
    allowed_tool_ids: tuple[str, ...] = (),
    denied_tool_ids: tuple[str, ...] = (),
    allowed_skill_ids: tuple[str, ...] = (),
) -> CapabilityGovernanceContext:
    tool_allowed = tuple(
        _identity_key(kind=CapabilityKind.TOOL, logical_id=tool_id, source=_TOOL_SOURCE)
        for tool_id in allowed_tool_ids
    )
    tool_denied = tuple(
        _identity_key(kind=CapabilityKind.TOOL, logical_id=tool_id, source=_TOOL_SOURCE)
        for tool_id in denied_tool_ids
    )
    skill_allowed = tuple(
        _identity_key(kind=CapabilityKind.SKILL, logical_id=skill_id, source=_SKILL_SOURCE)
        for skill_id in allowed_skill_ids
    )
    return CapabilityGovernanceContext(
        posture=CapabilityGovernancePosture.STRICT,
        tool_evidence=CapabilityToolGovernanceEvidence(
            allowed_keys=tool_allowed,
            denied_keys=tool_denied,
            allowed_constraint_mode=CapabilitySetConstraintMode.EXPLICIT_SET,
        ),
        skill_evidence=CapabilitySkillGovernanceEvidence(
            enabled_keys=skill_allowed,
            enabled_constraint_mode=CapabilitySetConstraintMode.EXPLICIT_SET,
        ),
    )


def _availability_evidence(
    *,
    host_available: tuple[CapabilityCatalogEntry, ...],
    scope_visible: tuple[CapabilityCatalogEntry, ...] | None = None,
    blocked: tuple[CapabilityCatalogEntry, ...] = (),
) -> CapabilityDiscoveryAvailabilityEvidence:
    visible = scope_visible if scope_visible is not None else host_available
    return CapabilityDiscoveryAvailabilityEvidence(
        host_available_keys=tuple(
            CapabilityIdentityKey.from_discovery_identity(entry.identity)
            for entry in host_available
        ),
        scope_visible_keys=tuple(
            CapabilityIdentityKey.from_discovery_identity(entry.identity)
            for entry in visible
        ),
        blocked_keys=tuple(
            CapabilityIdentityKey.from_discovery_identity(entry.identity)
            for entry in blocked
        ),
    )


def _need(
    *,
    stage_reference: str,
    stage_objective: str,
    query: CapabilityDiscoveryQuery | None,
) -> WorkStageCapabilityNeed:
    return WorkStageCapabilityNeed(
        work_reference=_WORK_REF,
        stage_reference=stage_reference,
        goal_objective=_GOAL,
        stage_objective=stage_objective,
        discovery_query=query,
    )


def _service() -> WorkStageCapabilityDiscoveryService:
    return WorkStageCapabilityDiscoveryService(governance_evaluators=_evaluators())


def test_stage_transition_rediscovery_differs_between_steps() -> None:
    tool_entry = _entry(
        kind=CapabilityKind.TOOL,
        logical_id=_TOOL_SEARCH,
        source=_TOOL_SOURCE,
    )
    skill_entry = _entry(
        kind=CapabilityKind.SKILL,
        logical_id=_SKILL_SUMMARY,
        source=_SKILL_SOURCE,
    )
    snapshot = _snapshot(tool_entry, skill_entry)
    availability = _availability_evidence(
        host_available=(tool_entry, skill_entry),
        scope_visible=(tool_entry, skill_entry),
    )
    context = _governance_context(
        allowed_tool_ids=(_TOOL_SEARCH,),
        allowed_skill_ids=(_SKILL_SUMMARY,),
    )
    service = _service()

    stage_one = service.resolve(
        _need(
            stage_reference="stage.collect",
            stage_objective="collect evidence",
            query=CapabilityDiscoveryQuery(
                scope=_enterprise_scope(),
                kinds=(CapabilityKind.TOOL,),
                logical_identity=LogicalIdentityFilter(
                    exact_logical_ids=(_TOOL_SEARCH,),
                ),
            ),
        ),
        snapshot=snapshot,
        availability_evidence=availability,
        governance_context=context,
    )
    stage_two = service.resolve(
        _need(
            stage_reference="stage.summarize",
            stage_objective="summarize evidence",
            query=CapabilityDiscoveryQuery(
                scope=_enterprise_scope(),
                kinds=(CapabilityKind.SKILL,),
                logical_identity=LogicalIdentityFilter(
                    exact_logical_ids=(_SKILL_SUMMARY,),
                ),
            ),
        ),
        snapshot=snapshot,
        availability_evidence=availability,
        governance_context=context,
    )

    assert stage_one.effective_set != stage_two.effective_set
    assert [
        item.identity.logical.logical_id
        for item in stage_one.effective_set.effective_candidates
    ] == [_TOOL_SEARCH]
    assert [
        item.identity.logical.logical_id
        for item in stage_two.effective_set.effective_candidates
    ] == [_SKILL_SUMMARY]

    transition = compare_work_stage_effective_capabilities(
        stage_one.effective_set,
        stage_two.effective_set,
    )
    assert transition.added_identity_keys[0].logical_id == _SKILL_SUMMARY
    assert transition.removed_identity_keys[0].logical_id == _TOOL_SEARCH


def test_effective_set_respects_policy_profile_and_scope_intersection() -> None:
    allowed_entry = _entry(
        kind=CapabilityKind.TOOL,
        logical_id=_TOOL_SEARCH,
        source=_TOOL_SOURCE,
    )
    profile_excluded = _entry(
        kind=CapabilityKind.TOOL,
        logical_id="tool.search.profile_excluded",
        source=_TOOL_SOURCE,
    )
    scope_excluded = _entry(
        kind=CapabilityKind.TOOL,
        logical_id=_SCOPE_EXCLUDED,
        source=_TOOL_SOURCE,
    )
    governance_blocked = _entry(
        kind=CapabilityKind.TOOL,
        logical_id=_TOOL_DENIED,
        source=_TOOL_SOURCE,
    )
    snapshot = _snapshot(
        allowed_entry,
        profile_excluded,
        scope_excluded,
        governance_blocked,
    )
    availability = _availability_evidence(
        host_available=(allowed_entry, governance_blocked),
        scope_visible=(allowed_entry, governance_blocked, scope_excluded),
    )
    context = _governance_context(
        allowed_tool_ids=(_TOOL_SEARCH,),
        denied_tool_ids=(_TOOL_DENIED,),
    )
    evidence = _service().resolve(
        _need(
            stage_reference="stage.collect",
            stage_objective="collect evidence",
            query=CapabilityDiscoveryQuery(
                scope=_enterprise_scope(),
                kinds=(CapabilityKind.TOOL,),
                logical_identity=LogicalIdentityFilter(
                    logical_id_prefixes=("tool.search.",),
                ),
            ),
        ),
        snapshot=snapshot,
        availability_evidence=availability,
        governance_context=context,
    )

    effective_ids = [
        item.identity.logical.logical_id
        for item in evidence.effective_set.effective_candidates
    ]
    assert effective_ids == [_TOOL_SEARCH]
    assert evidence.effective_set.governed_result.blocked
    assert profile_excluded.identity.logical.logical_id not in effective_ids
    assert scope_excluded.identity.logical.logical_id not in effective_ids
    assert governance_blocked.identity.logical.logical_id not in effective_ids


def test_catalog_only_candidate_is_discoverable_but_not_effective_executable() -> None:
    catalog_only = _entry(
        kind=CapabilityKind.TOOL,
        logical_id="tool.private.catalog_only",
        source=_TOOL_SOURCE,
    )
    snapshot = _snapshot(catalog_only)
    availability = _availability_evidence(
        host_available=(),
        scope_visible=(catalog_only,),
    )
    context = _governance_context(allowed_tool_ids=("tool.private.catalog_only",))
    evidence = _service().resolve(
        _need(
            stage_reference="stage.collect",
            stage_objective="collect evidence",
            query=CapabilityDiscoveryQuery(
                scope=_enterprise_scope(),
                kinds=(CapabilityKind.TOOL,),
                logical_identity=LogicalIdentityFilter(
                    exact_logical_ids=("tool.private.catalog_only",),
                ),
            ),
        ),
        snapshot=snapshot,
        availability_evidence=availability,
        governance_context=context,
    )

    assert evidence.effective_set.effective_candidates == ()
    assert len(evidence.catalog_only_identity_keys) == 1
    assert (
        evidence.catalog_only_identity_keys[0].logical_id == "tool.private.catalog_only"
    )
    allowed = evidence.effective_set.governed_result.allowed
    assert len(allowed) == 1
    assert allowed[0].availability is AvailabilityDisposition.CATALOG_AVAILABLE


def test_empty_need_returns_empty_effective_set_with_evidence() -> None:
    evidence = _service().resolve(
        _need(
            stage_reference="stage.wait",
            stage_objective="await approval",
            query=None,
        ),
        snapshot=_snapshot(),
        availability_evidence=CapabilityDiscoveryAvailabilityEvidence(),
        governance_context=_governance_context(),
    )
    assert evidence.effective_set.effective_candidates == ()
    assert evidence.effective_set.governed_result.allowed == ()
    assert evidence.effective_set.governed_result.blocked == ()


def test_valid_need_with_no_authorized_candidate_returns_empty_effective_set() -> None:
    tool_entry = _entry(
        kind=CapabilityKind.TOOL,
        logical_id=_TOOL_DENIED,
        source=_TOOL_SOURCE,
    )
    snapshot = _snapshot(tool_entry)
    availability = _availability_evidence(
        host_available=(tool_entry,),
        scope_visible=(tool_entry,),
    )
    context = _governance_context(denied_tool_ids=(_TOOL_DENIED,))
    evidence = _service().resolve(
        _need(
            stage_reference="stage.collect",
            stage_objective="collect evidence",
            query=CapabilityDiscoveryQuery(
                scope=_enterprise_scope(),
                kinds=(CapabilityKind.TOOL,),
                logical_identity=LogicalIdentityFilter(
                    exact_logical_ids=(_TOOL_DENIED,),
                ),
            ),
        ),
        snapshot=snapshot,
        availability_evidence=availability,
        governance_context=context,
    )
    assert evidence.effective_set.effective_candidates == ()
    assert len(evidence.effective_set.governed_result.blocked) == 1


def test_same_stage_inputs_are_deterministic() -> None:
    tool_entry = _entry(
        kind=CapabilityKind.TOOL,
        logical_id=_TOOL_SEARCH,
        source=_TOOL_SOURCE,
    )
    snapshot = _snapshot(tool_entry)
    availability = _availability_evidence(
        host_available=(tool_entry,),
        scope_visible=(tool_entry,),
    )
    context = _governance_context(allowed_tool_ids=(_TOOL_SEARCH,))
    need = _need(
        stage_reference="stage.collect",
        stage_objective="collect evidence",
        query=CapabilityDiscoveryQuery(
            scope=_enterprise_scope(),
            kinds=(CapabilityKind.TOOL,),
            logical_identity=LogicalIdentityFilter(exact_logical_ids=(_TOOL_SEARCH,)),
        ),
    )
    kwargs = {
        "snapshot": snapshot,
        "availability_evidence": availability,
        "governance_context": context,
    }
    first = _service().resolve(need, **kwargs)
    second = _service().resolve(need, **kwargs)
    assert first == second


def test_enterprise_scope_without_scope_evidence_fails_closed() -> None:
    tool_entry = _entry(
        kind=CapabilityKind.TOOL,
        logical_id=_TOOL_SEARCH,
        source=_TOOL_SOURCE,
    )
    with pytest.raises(CapabilityCatalogDiscoveryError, match="scope_visible_keys"):
        _service().resolve(
            _need(
                stage_reference="stage.collect",
                stage_objective="collect evidence",
                query=CapabilityDiscoveryQuery(scope=_enterprise_scope()),
            ),
            snapshot=_snapshot(tool_entry),
            availability_evidence=CapabilityDiscoveryAvailabilityEvidence(
                host_available_keys=(
                    CapabilityIdentityKey.from_discovery_identity(tool_entry.identity),
                ),
            ),
            governance_context=_governance_context(),
        )


def test_strict_empty_evaluators_fail_closed() -> None:
    tool_entry = _entry(
        kind=CapabilityKind.TOOL,
        logical_id=_TOOL_SEARCH,
        source=_TOOL_SOURCE,
    )
    service = WorkStageCapabilityDiscoveryService(governance_evaluators=())
    with pytest.raises(CapabilityGovernanceError, match="requires at least one evaluator"):
        service.resolve(
            _need(
                stage_reference="stage.collect",
                stage_objective="collect evidence",
                query=CapabilityDiscoveryQuery(scope=_enterprise_scope()),
            ),
            snapshot=_snapshot(tool_entry),
            availability_evidence=_availability_evidence(host_available=(tool_entry,)),
            governance_context=CapabilityGovernanceContext(
                posture=CapabilityGovernancePosture.STRICT,
            ),
        )


def test_module_entry_point_matches_service() -> None:
    tool_entry = _entry(
        kind=CapabilityKind.TOOL,
        logical_id=_TOOL_SEARCH,
        source=_TOOL_SOURCE,
    )
    snapshot = _snapshot(tool_entry)
    availability = _availability_evidence(
        host_available=(tool_entry,),
        scope_visible=(tool_entry,),
    )
    context = _governance_context(allowed_tool_ids=(_TOOL_SEARCH,))
    need = _need(
        stage_reference="stage.collect",
        stage_objective="collect evidence",
        query=CapabilityDiscoveryQuery(
            scope=_enterprise_scope(),
            kinds=(CapabilityKind.TOOL,),
            logical_identity=LogicalIdentityFilter(exact_logical_ids=(_TOOL_SEARCH,)),
        ),
    )
    service_result = _service().resolve(
        need,
        snapshot=snapshot,
        availability_evidence=availability,
        governance_context=context,
    )
    module_result = discover_effective_capabilities_for_work_stage(
        need,
        snapshot=snapshot,
        availability_evidence=availability,
        governance_context=context,
        governance_evaluators=_evaluators(),
        ranker=StableIdentityRanker(),
    )
    assert service_result == module_result


def test_stage8_resolution_does_not_mutate_registries_or_profiles() -> None:
    tool_registry = ToolRegistry()
    skill_registry = SkillRegistry()
    tool_profile = ToolProfile(enabled=[_TOOL_SEARCH])
    skill_profile = SkillProfile(enabled=[_SKILL_SUMMARY])
    tool_ids_before = tool_registry.tool_ids()
    skill_ids_before = skill_registry.skill_ids()

    tool_entry = _entry(
        kind=CapabilityKind.TOOL,
        logical_id=_TOOL_SEARCH,
        source=_TOOL_SOURCE,
    )
    _service().resolve(
        _need(
            stage_reference="stage.collect",
            stage_objective="collect evidence",
            query=CapabilityDiscoveryQuery(
                scope=_enterprise_scope(),
                kinds=(CapabilityKind.TOOL,),
                logical_identity=LogicalIdentityFilter(exact_logical_ids=(_TOOL_SEARCH,)),
            ),
        ),
        snapshot=_snapshot(tool_entry),
        availability_evidence=_availability_evidence(
            host_available=(tool_entry,),
            scope_visible=(tool_entry,),
        ),
        governance_context=_governance_context(allowed_tool_ids=(_TOOL_SEARCH,)),
    )

    assert tool_registry.tool_ids() == tool_ids_before
    assert skill_registry.skill_ids() == skill_ids_before
    assert tool_profile.enabled == [_TOOL_SEARCH]
    assert skill_profile.enabled == [_SKILL_SUMMARY]
