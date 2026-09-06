# © Artur Czarnecki. All rights reserved.

"""CAPABILITY-CATALOG-1 Stage 8 work-stage contract tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from intergrax.capability_catalog import (
    AvailabilityPreservingGovernanceEvaluator,
    CapabilityCatalogEntry,
    CapabilityDiscoveryCandidate,
    GovernedCapabilityCandidate,
    GovernedDiscoveryResult,
    RankedCapabilityCandidate,
    compare_work_stage_effective_capabilities,
    govern_capability_candidates,
)
from intergrax.capability_catalog.work_stage_effective import (
    EffectiveCapabilitySet,
    WorkStageCapabilityDiscoveryEvidence,
    WorkStageCapabilityTransitionEvidence,
)
from intergrax.contracts.capability_catalog import (
    AvailabilityDisposition,
    CapabilityDiscoveryIdentity,
    CapabilityDiscoveryQuery,
    CapabilityDiscoveryScope,
    CapabilityDiscoveryScopeMode,
    CapabilityGovernanceReasonCode,
    CapabilityIdentityKey,
    CapabilityKind,
    CapabilityLogicalIdentity,
    CapabilityProvenance,
    CapabilityRankingEvidence,
    CapabilityRankingSignal,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
    GovernanceDecisionEvidence,
    GovernanceDisposition,
    WorkStageCapabilityNeed,
)

pytestmark = pytest.mark.unit

_BASELINE = AvailabilityPreservingGovernanceEvaluator()
_TOOL_SOURCE = CapabilitySourceIdentity(
    source_id="tools.catalog.builtin",
    source_kind=CapabilitySourceKind.BUILTIN,
)
_SKILL_SOURCE = CapabilitySourceIdentity(
    source_id="skills.catalog.builtin",
    source_kind=CapabilitySourceKind.BUILTIN,
)


def _scope() -> CapabilityDiscoveryScope:
    return CapabilityDiscoveryScope(
        organization_id="org-acme",
        tenant_id="tenant-a",
        application_id="app-research",
        mode=CapabilityDiscoveryScopeMode.ENTERPRISE,
    )


def _need(stage_reference: str = "stage.collect") -> WorkStageCapabilityNeed:
    return WorkStageCapabilityNeed(
        work_reference="work.incident-42",
        stage_reference=stage_reference,
        goal_objective="resolve customer incident",
        stage_objective="collect evidence",
        discovery_query=CapabilityDiscoveryQuery(scope=_scope()),
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


def _ranked(
    entry: CapabilityCatalogEntry,
    *,
    availability: AvailabilityDisposition,
    position: int = 1,
) -> RankedCapabilityCandidate:
    candidate = CapabilityDiscoveryCandidate(
        catalog_entry=entry,
        availability=availability,
    )
    return RankedCapabilityCandidate(
        candidate=candidate,
        evidence=CapabilityRankingEvidence(
            ranker_id="stable.identity",
            rank_position=position,
            signal=CapabilityRankingSignal.STABLE_IDENTITY_ORDER,
            original_stage3_position=position,
        ),
    )


def _governed(
    entry: CapabilityCatalogEntry,
    *,
    availability: AvailabilityDisposition,
    position: int = 1,
) -> GovernedCapabilityCandidate:
    ranked = _ranked(entry, availability=availability, position=position)
    result = govern_capability_candidates((ranked,), evaluators=(_BASELINE,))
    return result.allowed[0]


def _effective_set(
    *,
    allowed: tuple[GovernedCapabilityCandidate, ...],
    effective: tuple[GovernedCapabilityCandidate, ...],
    need: WorkStageCapabilityNeed | None = None,
) -> EffectiveCapabilitySet:
    return EffectiveCapabilitySet(
        need=need or _need(),
        governed_result=GovernedDiscoveryResult(allowed=allowed, blocked=()),
        effective_candidates=effective,
    )


def test_work_stage_capability_need_requires_non_empty_references() -> None:
    with pytest.raises(ValidationError, match="work_reference"):
        WorkStageCapabilityNeed(
            work_reference="",
            stage_reference="stage.collect",
            goal_objective="resolve customer incident",
            stage_objective="collect evidence",
            discovery_query=CapabilityDiscoveryQuery(scope=_scope()),
        )


def test_work_stage_capability_need_distinguishes_goal_and_stage_objectives() -> None:
    need = WorkStageCapabilityNeed(
        work_reference="work.incident-42",
        stage_reference="stage.collect",
        goal_objective="resolve customer incident",
        stage_objective="collect evidence",
        discovery_query=CapabilityDiscoveryQuery(scope=_scope()),
    )
    assert need.goal_objective != need.stage_objective
    assert need.requests_capabilities is True


def test_work_stage_capability_need_empty_query_means_no_capability_request() -> None:
    need = WorkStageCapabilityNeed(
        work_reference="work.incident-42",
        stage_reference="stage.wait",
        goal_objective="resolve customer incident",
        stage_objective="await human approval",
    )
    assert need.discovery_query is None
    assert need.requests_capabilities is False


def test_work_stage_capability_need_is_immutable() -> None:
    need = WorkStageCapabilityNeed(
        work_reference="work.incident-42",
        stage_reference="stage.collect",
        goal_objective="resolve customer incident",
        stage_objective="collect evidence",
    )
    with pytest.raises(ValidationError):
        need.stage_reference = "stage.other"  # type: ignore[misc]


def test_effective_set_rejects_candidate_not_in_governed_allowed() -> None:
    allowed_entry = _entry(
        kind=CapabilityKind.TOOL,
        logical_id="tool.alpha",
        source=_TOOL_SOURCE,
    )
    forged_entry = _entry(
        kind=CapabilityKind.TOOL,
        logical_id="tool.beta",
        source=_TOOL_SOURCE,
    )
    allowed = _governed(
        allowed_entry,
        availability=AvailabilityDisposition.HOST_AVAILABLE,
    )
    forged = _governed(
        forged_entry,
        availability=AvailabilityDisposition.HOST_AVAILABLE,
    )
    with pytest.raises(
        ValidationError,
        match="effective candidates must be members of governed_result.allowed",
    ):
        _effective_set(allowed=(allowed,), effective=(forged,))


def test_effective_set_rejects_blocked_identity_not_in_allowed() -> None:
    allowed_entry = _entry(
        kind=CapabilityKind.TOOL,
        logical_id="tool.alpha",
        source=_TOOL_SOURCE,
    )
    blocked_entry = _entry(
        kind=CapabilityKind.TOOL,
        logical_id="tool.blocked",
        source=_TOOL_SOURCE,
    )
    allowed = _governed(
        allowed_entry,
        availability=AvailabilityDisposition.HOST_AVAILABLE,
    )
    blocked_ranked = _ranked(
        blocked_entry,
        availability=AvailabilityDisposition.HOST_AVAILABLE,
    )
    forged_evidence = (
        GovernanceDecisionEvidence(
            evaluator_id="test.forge",
            disposition=GovernanceDisposition.ALLOWED,
            reason_code=CapabilityGovernanceReasonCode.GOVERNANCE_ALLOWED,
        ),
    )
    forged = GovernedCapabilityCandidate(
        ranked=blocked_ranked,
        evidence=forged_evidence,
    )
    with pytest.raises(
        ValidationError,
        match="effective candidates must be members of governed_result.allowed",
    ):
        _effective_set(allowed=(allowed,), effective=(forged,))


def test_effective_set_accepts_valid_host_available_subset() -> None:
    host_entry = _entry(
        kind=CapabilityKind.TOOL,
        logical_id="tool.host",
        source=_TOOL_SOURCE,
    )
    catalog_entry = _entry(
        kind=CapabilityKind.TOOL,
        logical_id="tool.catalog",
        source=_TOOL_SOURCE,
    )
    host_allowed = _governed(
        host_entry,
        availability=AvailabilityDisposition.HOST_AVAILABLE,
        position=1,
    )
    catalog_allowed = _governed(
        catalog_entry,
        availability=AvailabilityDisposition.CATALOG_AVAILABLE,
        position=2,
    )
    effective_set = _effective_set(
        allowed=(host_allowed, catalog_allowed),
        effective=(host_allowed,),
    )
    assert effective_set.effective_candidates == (host_allowed,)


def test_effective_set_rejects_catalog_available_candidate() -> None:
    catalog_entry = _entry(
        kind=CapabilityKind.TOOL,
        logical_id="tool.catalog",
        source=_TOOL_SOURCE,
    )
    catalog_allowed = _governed(
        catalog_entry,
        availability=AvailabilityDisposition.CATALOG_AVAILABLE,
    )
    with pytest.raises(
        ValidationError,
        match="effective candidates must be HOST_AVAILABLE executable members",
    ):
        _effective_set(allowed=(catalog_allowed,), effective=(catalog_allowed,))


def test_effective_set_rejects_duplicate_effective_candidates() -> None:
    host_entry = _entry(
        kind=CapabilityKind.TOOL,
        logical_id="tool.host",
        source=_TOOL_SOURCE,
    )
    host_allowed = _governed(
        host_entry,
        availability=AvailabilityDisposition.HOST_AVAILABLE,
    )
    with pytest.raises(
        ValidationError,
        match="effective candidates must not contain duplicate capability identities",
    ):
        _effective_set(
            allowed=(host_allowed,),
            effective=(host_allowed, host_allowed),
        )


def test_discovery_evidence_derives_catalog_only_identity_keys() -> None:
    host_entry = _entry(
        kind=CapabilityKind.TOOL,
        logical_id="tool.host",
        source=_TOOL_SOURCE,
    )
    catalog_b = _entry(
        kind=CapabilityKind.TOOL,
        logical_id="tool.catalog.b",
        source=_TOOL_SOURCE,
    )
    catalog_c = _entry(
        kind=CapabilityKind.TOOL,
        logical_id="tool.catalog.c",
        source=_TOOL_SOURCE,
    )
    host_allowed = _governed(
        host_entry,
        availability=AvailabilityDisposition.HOST_AVAILABLE,
        position=1,
    )
    catalog_b_allowed = _governed(
        catalog_b,
        availability=AvailabilityDisposition.CATALOG_AVAILABLE,
        position=2,
    )
    catalog_c_allowed = _governed(
        catalog_c,
        availability=AvailabilityDisposition.CATALOG_AVAILABLE,
        position=3,
    )
    effective_set = _effective_set(
        allowed=(host_allowed, catalog_b_allowed, catalog_c_allowed),
        effective=(host_allowed,),
    )
    evidence = WorkStageCapabilityDiscoveryEvidence(effective_set=effective_set)
    assert evidence.need == effective_set.need
    assert [key.logical_id for key in evidence.catalog_only_identity_keys] == [
        "tool.catalog.b",
        "tool.catalog.c",
    ]


def test_discovery_evidence_rejects_forged_catalog_only_field() -> None:
    host_entry = _entry(
        kind=CapabilityKind.TOOL,
        logical_id="tool.host",
        source=_TOOL_SOURCE,
    )
    host_allowed = _governed(
        host_entry,
        availability=AvailabilityDisposition.HOST_AVAILABLE,
    )
    effective_set = _effective_set(allowed=(host_allowed,), effective=(host_allowed,))
    with pytest.raises(ValidationError):
        WorkStageCapabilityDiscoveryEvidence(
            effective_set=effective_set,
            catalog_only_identity_keys=(
                CapabilityIdentityKey.from_discovery_identity(host_entry.identity),
            ),  # type: ignore[call-arg]
        )


def test_discovery_evidence_derived_fields_are_not_serialized() -> None:
    host_entry = _entry(
        kind=CapabilityKind.TOOL,
        logical_id="tool.host",
        source=_TOOL_SOURCE,
    )
    host_allowed = _governed(
        host_entry,
        availability=AvailabilityDisposition.HOST_AVAILABLE,
    )
    effective_set = _effective_set(allowed=(host_allowed,), effective=(host_allowed,))
    evidence = WorkStageCapabilityDiscoveryEvidence(effective_set=effective_set)
    dumped = evidence.model_dump()
    assert "need" not in dumped
    assert "catalog_only_identity_keys" not in dumped
    assert evidence.need == effective_set.need


def test_transition_evidence_derives_added_and_removed_keys() -> None:
    entry_a = _entry(
        kind=CapabilityKind.TOOL,
        logical_id="tool.a",
        source=_TOOL_SOURCE,
    )
    entry_b = _entry(
        kind=CapabilityKind.TOOL,
        logical_id="tool.b",
        source=_TOOL_SOURCE,
    )
    entry_c = _entry(
        kind=CapabilityKind.TOOL,
        logical_id="tool.c",
        source=_TOOL_SOURCE,
    )
    allowed_a = _governed(entry_a, availability=AvailabilityDisposition.HOST_AVAILABLE, position=1)
    allowed_b = _governed(entry_b, availability=AvailabilityDisposition.HOST_AVAILABLE, position=2)
    allowed_c = _governed(entry_c, availability=AvailabilityDisposition.HOST_AVAILABLE, position=3)
    previous = _effective_set(allowed=(allowed_a, allowed_b), effective=(allowed_a, allowed_b))
    current = _effective_set(
        allowed=(allowed_b, allowed_c),
        effective=(allowed_b, allowed_c),
        need=previous.need,
    )
    transition = compare_work_stage_effective_capabilities(previous, current)
    assert [key.logical_id for key in transition.added_identity_keys] == ["tool.c"]
    assert [key.logical_id for key in transition.removed_identity_keys] == ["tool.a"]


def test_transition_evidence_rejects_forged_diff_fields() -> None:
    entry_a = _entry(
        kind=CapabilityKind.TOOL,
        logical_id="tool.a",
        source=_TOOL_SOURCE,
    )
    entry_b = _entry(
        kind=CapabilityKind.TOOL,
        logical_id="tool.b",
        source=_TOOL_SOURCE,
    )
    allowed_a = _governed(entry_a, availability=AvailabilityDisposition.HOST_AVAILABLE)
    allowed_b = _governed(entry_b, availability=AvailabilityDisposition.HOST_AVAILABLE)
    previous = _effective_set(allowed=(allowed_a,), effective=(allowed_a,))
    current = _effective_set(
        allowed=(allowed_b,),
        effective=(allowed_b,),
        need=previous.need,
    )
    with pytest.raises(ValidationError):
        WorkStageCapabilityTransitionEvidence(
            previous=previous,
            current=current,
            added_identity_keys=(),
            removed_identity_keys=(),
        )


def test_transition_evidence_requires_same_work_reference() -> None:
    entry = _entry(
        kind=CapabilityKind.TOOL,
        logical_id="tool.a",
        source=_TOOL_SOURCE,
    )
    allowed = _governed(entry, availability=AvailabilityDisposition.HOST_AVAILABLE)
    previous = _effective_set(
        allowed=(allowed,),
        effective=(allowed,),
        need=_need(stage_reference="stage.one"),
    )
    other_work = WorkStageCapabilityNeed(
        work_reference="work.other",
        stage_reference="stage.two",
        goal_objective="resolve customer incident",
        stage_objective="collect evidence",
        discovery_query=CapabilityDiscoveryQuery(scope=_scope()),
    )
    current = _effective_set(
        allowed=(allowed,),
        effective=(allowed,),
        need=other_work,
    )
    with pytest.raises(ValidationError, match="same work_reference"):
        compare_work_stage_effective_capabilities(previous, current)
