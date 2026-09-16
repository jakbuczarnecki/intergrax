# © Artur Czarnecki. All rights reserved.

"""MEM-ENT-10C: specialized recall & disclosure governance."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pytest

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.data_classification import DataClassification
from intergrax.memory.contracts.enterprise_memory_record import (
    MemoryProvenance,
    MemoryRecordGovernance,
    MemoryRecordSourceType,
    MemoryRecordTrust,
    MemoryTrustClass,
)
from intergrax.memory.contracts.entity_temporal_memory import (
    EntityMemoryScope,
    EntityRecord,
    EntityRelationDirection,
    EntityRelationQuery,
    EntityRelationRecord,
    EntityTypeRef,
    RelationTypeRef,
)
from intergrax.memory.contracts.long_horizon_memory import (
    CanonicalMemorySourceSnapshot,
    ChildSummaryRef,
    LineageTraversalRequest,
    LongHorizonMemoryScope,
    LongHorizonMemoryViolation,
    LongHorizonRecallQuery,
    LongHorizonSummaryRecord,
    MemorySourceRef,
    SummaryNodeKind,
    SummaryStatus,
)
from intergrax.memory.contracts.procedural_memory import ProcedureOutcomeEvidence
from intergrax.memory.contracts.memory_control import MemoryControlPlaneScope, MemoryControlScopeRef
from intergrax.memory.contracts.memory_models import MemoryKind
from intergrax.memory.contracts.memory_security_governance import (
    MemoryGovernanceDecision,
    MemoryGovernanceEvaluationRequest,
    MemoryGovernanceOperation,
    MemoryGovernanceOutcome,
    MemoryGovernanceReasonCode,
    MemoryGovernanceRecordSnapshot,
    MemorySecurityContext,
    MemorySecurityStrategySet,
)
from intergrax.memory.contracts.procedural_memory import (
    DefaultProcedureRankingStrategy,
    ProcedureApplicabilityStrategy,
    ProcedureQuery,
    ProcedureRecallContext,
    ProcedureRecord,
    ProcedureTypeRef,
)
from intergrax.memory.entity_temporal_memory_service import EntityTemporalMemoryService
from intergrax.memory.long_horizon_memory_service import (
    LongHorizonMemoryService,
    LongHorizonMemoryStrategySet,
    build_default_long_horizon_strategies,
)
from intergrax.memory.memory_security_governance_service import (
    MemorySecurityGovernanceService,
    build_default_memory_security_governance_service,
)
from intergrax.memory.memory_specialized_disclosure_governance import (
    filter_memory_disclosure_candidates,
    memory_security_context_for_recall,
)
from intergrax.memory.procedural_memory_service import (
    ProceduralMemoryService,
    ProceduralMemoryStrategySet,
    build_default_procedural_memory_strategies,
)
from intergrax.memory.stores.in_memory_entity_temporal_memory_store import (
    InMemoryEntityTemporalMemoryStore,
)
from intergrax.memory.stores.in_memory_long_horizon_memory_store import (
    InMemoryLongHorizonMemoryStore,
)
from intergrax.memory.stores.in_memory_procedural_memory_store import (
    InMemoryProceduralMemoryStore,
)
from intergrax.memory.strategies.defaults.memory_security_governance import (
    build_default_memory_security_strategy_set,
)
from tests.unit.memory.governance_source_fixtures import PermissiveCanonicalGovernanceSourceAuthority

pytestmark = pytest.mark.gate

_TENANT = "tenant-10c"
_USER = "user-10c"


def _identity(tenant: str = _TENANT, user: str = _USER) -> RequestIdentity:
    return RequestIdentity(tenant_id=tenant, user_id=user)


def _scope() -> EntityMemoryScope:
    return EntityMemoryScope(tenant_id=_TENANT, user_id=_USER)


class _ScopedSourceAuthority:
    def __init__(self, scope: LongHorizonMemoryScope) -> None:
        self._scope = scope

    def resolve_canonical_source(
        self,
        scope: LongHorizonMemoryScope,
        memory_id: str,
        revision: int,
    ) -> CanonicalMemorySourceSnapshot:
        if scope != self._scope:
            raise LongHorizonMemoryViolation("canonical source scope mismatch")
        return CanonicalMemorySourceSnapshot(
            memory_id=memory_id,
            revision=revision,
            content=f"content-{memory_id}",
            observed_at="2025-01-01T00:00:00+00:00",
        )


def _snapshot(memory_id: str, *, classification: DataClassification) -> MemoryGovernanceRecordSnapshot:
    return MemoryGovernanceRecordSnapshot(
        memory_id=memory_id,
        revision=1,
        kind=MemoryKind.USER_FACT,
        provenance=MemoryProvenance(source_type=MemoryRecordSourceType.USER_EXPLICIT),
        trust=MemoryRecordTrust(trust_class=MemoryTrustClass.USER_EXPLICIT),
        governance=MemoryRecordGovernance(data_classification=classification),
    )


def _recall_context() -> MemorySecurityContext:
    return memory_security_context_for_recall(_identity(), _scope())


@dataclass(frozen=True, slots=True)
class _FixedGovernancePolicy:
    outcome: MemoryGovernanceOutcome
    reason_code: MemoryGovernanceReasonCode = MemoryGovernanceReasonCode.ALLOWED
    policy_id: str = "test.governance"
    policy_version: str = "1"

    def evaluate(self, request: MemoryGovernanceEvaluationRequest) -> MemoryGovernanceDecision:
        record = request.existing_record or request.proposed_record
        return MemoryGovernanceDecision(
            outcome=self.outcome,
            reason_code=self.reason_code,
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            operation=request.context.operation,
            subject_memory_id=record.memory_id if record else None,
        )


@dataclass(frozen=True, slots=True)
class _ExplodingPolicy:
    policy_id: str = "test.boom"
    policy_version: str = "1"

    def evaluate(self, request: MemoryGovernanceEvaluationRequest) -> MemoryGovernanceDecision:
        raise RuntimeError("policy failure")


def _governance_with_governance_policy(policy: object) -> MemorySecurityGovernanceService:
    base = build_default_memory_security_strategy_set()
    strategies = MemorySecurityStrategySet(
        authorization=base.authorization,
        trust=base.trust,
        admission=base.admission,
        governance=policy,
        retention=base.retention,
    )
    return MemorySecurityGovernanceService(strategies=strategies)


def _governance_missing() -> MemorySecurityGovernanceService:
    return MemorySecurityGovernanceService(strategies=None)


def test_filter_disclosure_allow_included() -> None:
    governance = build_default_memory_security_governance_service()
    ctx = _recall_context()
    snap = _snapshot("a", classification=DataClassification.INTERNAL)
    kept = filter_memory_disclosure_candidates(
        governance, ctx, (snap,), to_snapshot=lambda item: item
    )
    assert kept == (snap,)


def test_filter_disclosure_deny_excluded() -> None:
    governance = _governance_with_governance_policy(
        _FixedGovernancePolicy(
            outcome=MemoryGovernanceOutcome.DENY,
            reason_code=MemoryGovernanceReasonCode.GOVERNANCE_DENY,
        )
    )
    ctx = _recall_context()
    snap = _snapshot("a", classification=DataClassification.INTERNAL)
    assert filter_memory_disclosure_candidates(
        governance, ctx, (snap,), to_snapshot=lambda item: item
    ) == ()


def test_filter_disclosure_require_review_excluded() -> None:
    governance = _governance_with_governance_policy(
        _FixedGovernancePolicy(
            outcome=MemoryGovernanceOutcome.REQUIRE_REVIEW,
            reason_code=MemoryGovernanceReasonCode.REVIEW_REQUIRED,
        )
    )
    ctx = _recall_context()
    snap = _snapshot("a", classification=DataClassification.INTERNAL)
    assert filter_memory_disclosure_candidates(
        governance, ctx, (snap,), to_snapshot=lambda item: item
    ) == ()


def test_filter_disclosure_policy_failure_excluded() -> None:
    governance = _governance_with_governance_policy(_ExplodingPolicy())
    ctx = _recall_context()
    snap = _snapshot("a", classification=DataClassification.INTERNAL)
    assert filter_memory_disclosure_candidates(
        governance, ctx, (snap,), to_snapshot=lambda item: item
    ) == ()


def test_filter_disclosure_policy_missing_excluded() -> None:
    governance = _governance_missing()
    ctx = _recall_context()
    snap = _snapshot("a", classification=DataClassification.INTERNAL)
    assert filter_memory_disclosure_candidates(
        governance, ctx, (snap,), to_snapshot=lambda item: item
    ) == ()


def _entity_record(entity_id: str, *, restricted: bool = False) -> EntityRecord:
    classification = (
        DataClassification.RESTRICTED if restricted else DataClassification.INTERNAL
    )
    return EntityRecord(
        entity_id=entity_id,
        entity_type=EntityTypeRef("person"),
        canonical_name=f"name-{entity_id}",
        governance=MemoryRecordGovernance(data_classification=classification),
    )


def test_entity_get_entity_restricted_hidden() -> None:
    store = InMemoryEntityTemporalMemoryStore()
    scope = _scope()
    store.upsert_entity(scope, _entity_record("allowed"))
    store.upsert_entity(scope, _entity_record("restricted", restricted=True))
    service = EntityTemporalMemoryService(
        _store=store,
        _security_governance=build_default_memory_security_governance_service(),
    )
    listed = service.list_entities(_identity(), scope)
    ids = {record.entity_id for record in listed}
    assert "allowed" in ids
    assert "restricted" not in ids
    assert service.get_entity(_identity(), scope, "restricted") is None


def test_entity_cross_scope_no_disclosure() -> None:
    store = InMemoryEntityTemporalMemoryStore()
    scope = _scope()
    store.upsert_entity(scope, _entity_record("e1"))
    service = EntityTemporalMemoryService(
        _store=store,
        _security_governance=build_default_memory_security_governance_service(),
    )
    wrong = RequestIdentity(tenant_id="other-tenant", user_id=_USER)
    assert service.get_entity(wrong, scope, "e1") is None


def test_entity_relation_restricted_excluded_from_query() -> None:
    store = InMemoryEntityTemporalMemoryStore()
    scope = _scope()
    store.upsert_entity(scope, _entity_record("a"))
    store.upsert_entity(scope, _entity_record("b"))
    store.upsert_relation(
        scope,
        EntityRelationRecord(
            relation_id="rel-ok",
            source_entity_id="a",
            target_entity_id="b",
            relation_type=RelationTypeRef("knows"),
            governance=MemoryRecordGovernance(data_classification=DataClassification.INTERNAL),
        ),
    )
    store.upsert_relation(
        scope,
        EntityRelationRecord(
            relation_id="rel-deny",
            source_entity_id="a",
            target_entity_id="b",
            relation_type=RelationTypeRef("secret"),
            governance=MemoryRecordGovernance(data_classification=DataClassification.RESTRICTED),
        ),
    )
    service = EntityTemporalMemoryService(
        _store=store,
        _security_governance=build_default_memory_security_governance_service(),
    )
    result = service.query_relations(
        _identity(),
        scope,
        EntityRelationQuery(
            entity_id="a",
            direction=EntityRelationDirection.OUTBOUND,
            limit=10,
            as_of=datetime(2025, 6, 1, tzinfo=timezone.utc),
        ),
    )
    assert {rel.relation_id for rel in result.relations} == {"rel-ok"}


def _procedure(procedure_id: str, *, quality: float | None = None) -> ProcedureRecord:
    return ProcedureRecord(
        procedure_id=procedure_id,
        procedure_type=ProcedureTypeRef("runbook"),
        title=procedure_id,
        source_memory_id=f"src-{procedure_id}",
        source_memory_revision=1,
        provenance=MemoryProvenance(source_type=MemoryRecordSourceType.USER_EXPLICIT),
        trust=MemoryRecordTrust(trust_class=MemoryTrustClass.USER_EXPLICIT),
        outcome_evidence=ProcedureOutcomeEvidence(quality_score=quality),
    )


@dataclass
class _RecordingApplicability:
    seen: list[ProcedureRecord] = field(default_factory=list)

    def is_applicable(
        self,
        record: ProcedureRecord,
        context: ProcedureRecallContext,
        *,
        query: ProcedureQuery | None = None,
    ) -> bool:
        self.seen.append(record)
        return True


@dataclass
class _RecordingRanking:
    seen: list[ProcedureRecord] = field(default_factory=list)

    def rank(
        self,
        candidates: tuple[ProcedureRecord, ...],
        context: ProcedureRecallContext,
    ) -> tuple[ProcedureRecord, ...]:
        self.seen.extend(candidates)
        return DefaultProcedureRankingStrategy().rank(candidates, context)


def test_procedural_denied_not_passed_to_applicability_or_ranking() -> None:
    scope = _scope()
    store = InMemoryProceduralMemoryStore()
    store.upsert_procedure(
        scope,
        ProcedureRecord(
            procedure_id="denied",
            procedure_type=ProcedureTypeRef("runbook"),
            title="denied",
            source_memory_id="src-denied",
            source_memory_revision=1,
            governance=MemoryRecordGovernance(data_classification=DataClassification.RESTRICTED),
            provenance=MemoryProvenance(source_type=MemoryRecordSourceType.USER_EXPLICIT),
            trust=MemoryRecordTrust(trust_class=MemoryTrustClass.USER_EXPLICIT),
        ),
    )
    store.upsert_procedure(scope, _procedure("allowed", quality=0.1))
    applicability = _RecordingApplicability()
    ranking = _RecordingRanking()
    service = ProceduralMemoryService(
        _store=store,
        _strategies=ProceduralMemoryStrategySet(
            applicability=applicability,
            ranking=ranking,
        ),
        _security_governance=build_default_memory_security_governance_service(),
        _governance_source_authority=PermissiveCanonicalGovernanceSourceAuthority(),
    )
    result = service.recall_procedures(
        _identity(),
        scope,
        ProcedureQuery(limit=10),
        ProcedureRecallContext(),
    )
    assert {p.procedure_id for p in result.procedures} == {"allowed"}
    assert {p.procedure_id for p in applicability.seen} == {"allowed"}
    assert {p.procedure_id for p in ranking.seen} == {"allowed"}


def test_procedural_cross_scope_no_disclosure() -> None:
    scope = _scope()
    store = InMemoryProceduralMemoryStore()
    store.upsert_procedure(scope, _procedure("p1"))
    service = ProceduralMemoryService(
        _store=store,
        _strategies=build_default_procedural_memory_strategies(),
        _security_governance=build_default_memory_security_governance_service(),
        _governance_source_authority=PermissiveCanonicalGovernanceSourceAuthority(),
    )
    wrong = RequestIdentity(tenant_id=_TENANT, user_id="other-user")
    result = service.recall_procedures(
        wrong,
        scope,
        ProcedureQuery(limit=5),
        ProcedureRecallContext(),
    )
    assert result.procedures == ()


@dataclass(frozen=True, slots=True)
class _DenyProcedureTypePolicy:
    policy_id: str = "test.procedure.kind"
    policy_version: str = "1"

    def evaluate(self, request: MemoryGovernanceEvaluationRequest) -> MemoryGovernanceDecision:
        record = request.existing_record
        if record is not None and record.kind is MemoryKind.PROCEDURAL:
            preview = (record.content_preview or "").lower()
            if "deny-me" in preview:
                return MemoryGovernanceDecision(
                    outcome=MemoryGovernanceOutcome.DENY,
                    reason_code=MemoryGovernanceReasonCode.GOVERNANCE_DENY,
                    policy_id=self.policy_id,
                    policy_version=self.policy_version,
                    operation=request.context.operation,
                    subject_memory_id=record.memory_id,
                )
        return MemoryGovernanceDecision(
            outcome=MemoryGovernanceOutcome.ALLOW,
            reason_code=MemoryGovernanceReasonCode.ALLOWED,
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            operation=request.context.operation,
        )


def test_external_policy_changes_procedural_disclosure_without_service_change() -> None:
    scope = _scope()
    store = InMemoryProceduralMemoryStore()
    store.upsert_procedure(
        scope,
        ProcedureRecord(
            procedure_id="deny-me",
            procedure_type=ProcedureTypeRef("runbook"),
            title="deny-me",
            source_memory_id="src-1",
            source_memory_revision=1,
            provenance=MemoryProvenance(source_type=MemoryRecordSourceType.USER_EXPLICIT),
            trust=MemoryRecordTrust(trust_class=MemoryTrustClass.USER_EXPLICIT),
        ),
    )
    store.upsert_procedure(scope, _procedure("keep"))
    service = ProceduralMemoryService(
        _store=store,
        _strategies=build_default_procedural_memory_strategies(),
        _security_governance=_governance_with_governance_policy(_DenyProcedureTypePolicy()),
        _governance_source_authority=PermissiveCanonicalGovernanceSourceAuthority(),
    )
    result = service.recall_procedures(
        _identity(),
        scope,
        ProcedureQuery(limit=10),
        ProcedureRecallContext(),
    )
    assert {p.procedure_id for p in result.procedures} == {"keep"}


def _leaf(summary_id: str, *, restricted: bool = False) -> LongHorizonSummaryRecord:
    classification = (
        DataClassification.RESTRICTED if restricted else DataClassification.INTERNAL
    )
    return LongHorizonSummaryRecord(
        summary_id=summary_id,
        summary_level=1,
        node_kind=SummaryNodeKind.LEAF,
        revision=1,
        content=f"content-{summary_id}",
        source_memory_refs=(MemorySourceRef("mem-1", 1),),
        covered_from="2025-01-01T00:00:00+00:00",
        covered_until="2025-01-02T00:00:00+00:00",
        source_count=1,
        created_at="2025-01-01T00:00:00+00:00",
        governance=MemoryRecordGovernance(data_classification=classification),
    )


@dataclass
class _RecordingLhRecall:
    seen: list[LongHorizonSummaryRecord] = field(default_factory=list)

    def rank(
        self,
        candidates: tuple[LongHorizonSummaryRecord, ...],
        query: LongHorizonRecallQuery,
    ) -> tuple[LongHorizonSummaryRecord, ...]:
        self.seen.extend(candidates)
        return candidates


def test_long_horizon_restricted_summary_not_ranked() -> None:
    scope = _scope()
    store = InMemoryLongHorizonMemoryStore()
    store.upsert_summary(scope, _leaf("allowed"))
    store.upsert_summary(scope, _leaf("restricted", restricted=True))
    recall_strategy = _RecordingLhRecall()
    strategies = build_default_long_horizon_strategies()
    service = LongHorizonMemoryService(
        _store=store,
        _strategies=LongHorizonMemoryStrategySet(
            grouping=strategies.grouping,
            summary=strategies.summary,
            compaction=strategies.compaction,
            recall=recall_strategy,
        ),
        _source_authority=_ScopedSourceAuthority(scope),
        _governance_source_authority=PermissiveCanonicalGovernanceSourceAuthority(),
        _security_governance=build_default_memory_security_governance_service(),
    )
    result = service.recall(
        _identity(),
        scope,
        LongHorizonRecallQuery(limit=10, statuses=(SummaryStatus.ACTIVE,)),
    )
    assert {s.summary_id for s in result.summaries} == {"allowed"}
    assert {s.summary_id for s in recall_strategy.seen} == {"allowed"}


def test_long_horizon_traversal_denied_child_absent() -> None:
    scope = _scope()
    store = InMemoryLongHorizonMemoryStore()
    denied_child = _leaf("child-denied", restricted=True)
    allowed_child = _leaf("child-ok")
    parent = LongHorizonSummaryRecord(
        summary_id="parent",
        summary_level=2,
        node_kind=SummaryNodeKind.AGGREGATE,
        revision=1,
        content="parent",
        child_summary_refs=(
            ChildSummaryRef("child-denied", 1),
            ChildSummaryRef("child-ok", 1),
        ),
        source_count=2,
        created_at="2025-01-01T00:00:00+00:00",
    )
    store.upsert_summary(scope, denied_child)
    store.upsert_summary(scope, allowed_child)
    store.upsert_summary(scope, parent)
    strategies = build_default_long_horizon_strategies()
    service = LongHorizonMemoryService(
        _store=store,
        _strategies=strategies,
        _source_authority=_ScopedSourceAuthority(scope),
        _governance_source_authority=PermissiveCanonicalGovernanceSourceAuthority(),
        _security_governance=build_default_memory_security_governance_service(),
    )
    result = service.traverse_lineage(
        _identity(),
        scope,
        LineageTraversalRequest(summary_id="parent", max_depth=4, max_nodes=16),
    )
    visited_ids = {s.summary_id for s in result.visited_summaries}
    assert "child-denied" not in visited_ids
    assert "child-ok" in visited_ids


def test_specialized_services_do_not_import_default_policies() -> None:
    paths = (
        "intergrax/memory/procedural_memory_service.py",
        "intergrax/memory/long_horizon_memory_service.py",
        "intergrax/memory/entity_temporal_memory_service.py",
        "intergrax/memory/memory_specialized_disclosure_governance.py",
    )
    forbidden = {
        "DefaultMemoryGovernancePolicy",
        "DefaultMemoryAuthorizationPolicy",
    }
    for path in paths:
        source = ast.parse(open(path, encoding="utf-8").read())
        names = {node.id for node in ast.walk(source) if isinstance(node, ast.Name)}
        assert not forbidden.intersection(names), path
