# © Artur Czarnecki. All rights reserved.

"""MEM-ENT-10C-R: entity graph disclosure integrity and capability contract closure."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.data_classification import DataClassification
from intergrax.memory.contracts.enterprise_memory_record import MemoryRecordGovernance
from intergrax.memory.contracts.entity_temporal_memory import (
    EntityGraphDisclosureResult,
    EntityMemoryScope,
    EntityRecord,
    EntityRelationDirection,
    EntityRelationQuery,
    EntityRelationRecord,
    EntityTemporalMemoryCapability,
    EntityTypeRef,
    RelationTypeRef,
)
from intergrax.memory.contracts.memory_security_governance import (
    MemoryGovernanceDecision,
    MemoryGovernanceEvaluationRequest,
    MemoryGovernanceOutcome,
    MemoryGovernanceReasonCode,
)
from intergrax.memory.entity_temporal_memory_service import EntityTemporalMemoryService
from intergrax.memory.memory_security_governance_service import (
    MemorySecurityGovernanceService,
    build_default_memory_security_governance_service,
)
from intergrax.memory.stores.in_memory_entity_temporal_memory_store import (
    InMemoryEntityTemporalMemoryStore,
)
from intergrax.memory.strategies.defaults.memory_security_governance import (
    build_default_memory_security_strategy_set,
)
from intergrax.memory.contracts.memory_security_governance import MemorySecurityStrategySet

pytestmark = pytest.mark.gate

_TENANT = "tenant-10c-r"
_USER = "user-10c-r"
_AS_OF = datetime(2025, 6, 1, tzinfo=timezone.utc)


def _identity(tenant: str = _TENANT, user: str = _USER) -> RequestIdentity:
    return RequestIdentity(tenant_id=tenant, user_id=user)


def _scope() -> EntityMemoryScope:
    return EntityMemoryScope(tenant_id=_TENANT, user_id=_USER)


def _entity(entity_id: str, *, restricted: bool = False) -> EntityRecord:
    classification = (
        DataClassification.RESTRICTED if restricted else DataClassification.INTERNAL
    )
    return EntityRecord(
        entity_id=entity_id,
        entity_type=EntityTypeRef("person"),
        canonical_name=f"name-{entity_id}",
        governance=MemoryRecordGovernance(data_classification=classification),
    )


def _relation(
    relation_id: str,
    source: str,
    target: str,
    *,
    restricted: bool = False,
) -> EntityRelationRecord:
    classification = (
        DataClassification.RESTRICTED if restricted else DataClassification.INTERNAL
    )
    return EntityRelationRecord(
        relation_id=relation_id,
        source_entity_id=source,
        target_entity_id=target,
        relation_type=RelationTypeRef("knows"),
        governance=MemoryRecordGovernance(data_classification=classification),
    )


def _service(store: InMemoryEntityTemporalMemoryStore | None = None) -> EntityTemporalMemoryService:
    return EntityTemporalMemoryService(
        _store=store or InMemoryEntityTemporalMemoryStore(),
        _security_governance=build_default_memory_security_governance_service(),
    )


def _outbound_query(entity_id: str) -> EntityRelationQuery:
    return EntityRelationQuery(
        entity_id=entity_id,
        direction=EntityRelationDirection.OUTBOUND,
        limit=10,
        as_of=_AS_OF,
    )


def _inbound_query(entity_id: str) -> EntityRelationQuery:
    return EntityRelationQuery(
        entity_id=entity_id,
        direction=EntityRelationDirection.INBOUND,
        limit=10,
        as_of=_AS_OF,
    )


def test_allowed_relation_and_endpoints_visible() -> None:
    store = InMemoryEntityTemporalMemoryStore()
    scope = _scope()
    store.upsert_entity(scope, _entity("a"))
    store.upsert_entity(scope, _entity("b"))
    store.upsert_relation(scope, _relation("rel-ab", "a", "b"))
    service = _service(store)
    query = _outbound_query("a")
    rels = service.query_relations(_identity(), scope, query)
    assert {r.relation_id for r in rels.relations} == {"rel-ab"}
    neighbors = service.disclose_entity_neighbors(_identity(), scope, "a", query=query)
    assert {e.entity_id for e in neighbors.entities} == {"b"}
    assert {r.relation_id for r in neighbors.relations} == {"rel-ab"}


def test_allowed_relation_denied_target_hidden() -> None:
    store = InMemoryEntityTemporalMemoryStore()
    scope = _scope()
    store.upsert_entity(scope, _entity("a"))
    store.upsert_entity(scope, _entity("b", restricted=True))
    store.upsert_relation(scope, _relation("rel-ab", "a", "b"))
    service = _service(store)
    query = _outbound_query("a")
    assert service.query_relations(_identity(), scope, query).relations == ()
    neighbors = service.disclose_entity_neighbors(_identity(), scope, "a", query=query)
    assert neighbors == EntityGraphDisclosureResult(entities=(), relations=())


def test_denied_source_hides_relation() -> None:
    store = InMemoryEntityTemporalMemoryStore()
    scope = _scope()
    store.upsert_entity(scope, _entity("a", restricted=True))
    store.upsert_entity(scope, _entity("b"))
    store.upsert_relation(scope, _relation("rel-ab", "a", "b"))
    service = _service(store)
    query = _outbound_query("a")
    assert service.query_relations(_identity(), scope, query).relations == ()


def test_denied_relation_hidden_endpoints_allowed() -> None:
    store = InMemoryEntityTemporalMemoryStore()
    scope = _scope()
    store.upsert_entity(scope, _entity("a"))
    store.upsert_entity(scope, _entity("b"))
    store.upsert_relation(scope, _relation("rel-ab", "a", "b", restricted=True))
    service = _service(store)
    query = _outbound_query("a")
    assert service.query_relations(_identity(), scope, query).relations == ()


def test_all_denied_empty_neighbor_result() -> None:
    store = InMemoryEntityTemporalMemoryStore()
    scope = _scope()
    store.upsert_entity(scope, _entity("a", restricted=True))
    store.upsert_entity(scope, _entity("b", restricted=True))
    store.upsert_relation(scope, _relation("rel-ab", "a", "b", restricted=True))
    service = _service(store)
    query = _outbound_query("a")
    neighbors = service.disclose_entity_neighbors(_identity(), scope, "a", query=query)
    assert neighbors == EntityGraphDisclosureResult(entities=(), relations=())


def test_dangling_endpoint_excludes_relation() -> None:
    class _DanglingRelationStore(InMemoryEntityTemporalMemoryStore):
        def query_relations(
            self,
            scope: EntityMemoryScope,
            query: EntityRelationQuery,
        ):
            from intergrax.memory.contracts.entity_temporal_memory import EntityRelationResult

            dangling = _relation("rel-missing", "a", "missing-b")
            return EntityRelationResult(relations=(dangling,))

    store = _DanglingRelationStore()
    scope = _scope()
    store.upsert_entity(scope, _entity("a"))
    service = _service(store)
    query = _outbound_query("a")
    assert service.query_relations(_identity(), scope, query).relations == ()


def test_inbound_denied_source_hidden() -> None:
    store = InMemoryEntityTemporalMemoryStore()
    scope = _scope()
    store.upsert_entity(scope, _entity("a", restricted=True))
    store.upsert_entity(scope, _entity("b"))
    store.upsert_relation(scope, _relation("rel-ab", "a", "b"))
    service = _service(store)
    query = _inbound_query("b")
    assert service.query_relations(_identity(), scope, query).relations == ()


def test_outbound_denied_target_hidden() -> None:
    store = InMemoryEntityTemporalMemoryStore()
    scope = _scope()
    store.upsert_entity(scope, _entity("a"))
    store.upsert_entity(scope, _entity("b", restricted=True))
    store.upsert_relation(scope, _relation("rel-ab", "a", "b"))
    service = _service(store)
    rels = service.query_relations(_identity(), scope, _outbound_query("a"))
    assert rels.relations == ()
    for rel in rels.relations:
        assert "b" not in (rel.source_entity_id, rel.target_entity_id)


def test_query_relations_no_denied_endpoint_leak() -> None:
    store = InMemoryEntityTemporalMemoryStore()
    scope = _scope()
    store.upsert_entity(scope, _entity("a"))
    store.upsert_entity(scope, _entity("secret-b", restricted=True))
    store.upsert_relation(scope, _relation("rel-leak", "a", "secret-b"))
    service = _service(store)
    result = service.query_relations(_identity(), scope, _outbound_query("a"))
    assert result.relations == ()
    leaked = {
        endpoint
        for rel in result.relations
        for endpoint in (rel.source_entity_id, rel.target_entity_id)
    }
    assert "secret-b" not in leaked


def test_denied_root_no_relation_disclosure() -> None:
    store = InMemoryEntityTemporalMemoryStore()
    scope = _scope()
    store.upsert_entity(scope, _entity("a", restricted=True))
    store.upsert_entity(scope, _entity("b"))
    store.upsert_relation(scope, _relation("rel-ab", "a", "b"))
    service = _service(store)
    query = _outbound_query("a")
    assert service.query_relations(_identity(), scope, query).relations == ()


def test_entity_temporal_memory_capability_surface() -> None:
    assert issubclass(EntityTemporalMemoryService, object)
    service = _service()
    assert isinstance(service, EntityTemporalMemoryCapability)


@dataclass(frozen=True, slots=True)
class _DenyEntityIdPolicy:
    denied_suffix: str
    policy_id: str = "test.entity.id"
    policy_version: str = "1"

    def evaluate(self, request: MemoryGovernanceEvaluationRequest) -> MemoryGovernanceDecision:
        record = request.existing_record or request.proposed_record
        memory_id = (record.memory_id if record else "") or ""
        if memory_id.endswith(self.denied_suffix):
            return MemoryGovernanceDecision(
                outcome=MemoryGovernanceOutcome.DENY,
                reason_code=MemoryGovernanceReasonCode.GOVERNANCE_DENY,
                policy_id=self.policy_id,
                policy_version=self.policy_version,
                operation=request.context.operation,
                subject_memory_id=memory_id or None,
            )
        return MemoryGovernanceDecision(
            outcome=MemoryGovernanceOutcome.ALLOW,
            reason_code=MemoryGovernanceReasonCode.ALLOWED,
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            operation=request.context.operation,
        )


def _governance_with_policy(policy: object) -> MemorySecurityGovernanceService:
    base = build_default_memory_security_strategy_set()
    strategies = MemorySecurityStrategySet(
        authorization=base.authorization,
        trust=base.trust,
        admission=base.admission,
        governance=policy,
        retention=base.retention,
    )
    return MemorySecurityGovernanceService(strategies=strategies)


def test_external_policy_hides_endpoint_relation_without_service_change() -> None:
    store = InMemoryEntityTemporalMemoryStore()
    scope = _scope()
    store.upsert_entity(scope, _entity("a"))
    store.upsert_entity(scope, _entity("deny-me"))
    store.upsert_relation(scope, _relation("rel-hide", "a", "deny-me"))
    store.upsert_entity(scope, _entity("keep"))
    store.upsert_relation(scope, _relation("rel-keep", "a", "keep"))
    service = EntityTemporalMemoryService(
        _store=store,
        _security_governance=_governance_with_policy(_DenyEntityIdPolicy("deny-me")),
    )
    query = _outbound_query("a")
    assert {r.relation_id for r in service.query_relations(_identity(), scope, query).relations} == {
        "rel-keep"
    }


@dataclass(frozen=True, slots=True)
class _ExplodingPolicy:
    policy_id: str = "test.boom"
    policy_version: str = "1"

    def evaluate(self, request: MemoryGovernanceEvaluationRequest) -> MemoryGovernanceDecision:
        raise RuntimeError("policy failure")


def test_endpoint_policy_failure_hides_relation() -> None:
    store = InMemoryEntityTemporalMemoryStore()
    scope = _scope()
    store.upsert_entity(scope, _entity("a"))
    store.upsert_entity(scope, _entity("b"))
    store.upsert_relation(scope, _relation("rel-ab", "a", "b"))
    service = EntityTemporalMemoryService(
        _store=store,
        _security_governance=_governance_with_policy(_ExplodingPolicy()),
    )
    assert service.query_relations(_identity(), scope, _outbound_query("a")).relations == ()


@dataclass(frozen=True, slots=True)
class _RequireReviewPolicy:
    policy_id: str = "test.review"
    policy_version: str = "1"

    def evaluate(self, request: MemoryGovernanceEvaluationRequest) -> MemoryGovernanceDecision:
        return MemoryGovernanceDecision(
            outcome=MemoryGovernanceOutcome.REQUIRE_REVIEW,
            reason_code=MemoryGovernanceReasonCode.REVIEW_REQUIRED,
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            operation=request.context.operation,
        )


def test_require_review_endpoint_hides_relation() -> None:
    store = InMemoryEntityTemporalMemoryStore()
    scope = _scope()
    store.upsert_entity(scope, _entity("a"))
    store.upsert_entity(scope, _entity("b"))
    store.upsert_relation(scope, _relation("rel-ab", "a", "b"))
    service = EntityTemporalMemoryService(
        _store=store,
        _security_governance=_governance_with_policy(_RequireReviewPolicy()),
    )
    assert service.query_relations(_identity(), scope, _outbound_query("a")).relations == ()


def test_cross_scope_workspace_mismatch_hides_graph() -> None:
    store = InMemoryEntityTemporalMemoryStore()
    scope = _scope()
    store.upsert_entity(scope, _entity("a"))
    store.upsert_entity(scope, _entity("b"))
    store.upsert_relation(scope, _relation("rel-ab", "a", "b"))
    service = _service(store)
    wrong = RequestIdentity(tenant_id="other-tenant", user_id=_USER)
    assert service.query_relations(wrong, scope, _outbound_query("a")).relations == ()
