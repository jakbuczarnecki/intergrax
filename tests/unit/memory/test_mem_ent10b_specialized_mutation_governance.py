# © Artur Czarnecki. All rights reserved.

"""MEM-ENT-10B: specialized memory mutation governance."""

from __future__ import annotations

import ast
from dataclasses import dataclass

import pytest

from intergrax.applications._shared.long_horizon_memory_wiring import (
    resolve_long_horizon_memory_capability,
)
from intergrax.applications._shared.memory_security_governance_wiring import (
    resolve_memory_security_governance_service,
)
from intergrax.applications._shared.procedural_memory_wiring import (
    resolve_procedural_memory_capability,
)
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    MemoryProfile,
)
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
    entity_memory_entity_id_for_entry,
)
from intergrax.memory.contracts.long_horizon_memory import (
    CanonicalMemorySourceSnapshot,
    ChildSummaryRef,
    LongHorizonCompactionRequest,
    LongHorizonCompactionSource,
    LongHorizonMemoryScope,
    LongHorizonSummaryRecord,
    MemorySourceRef,
    SummaryNodeKind,
    SummaryStatus,
)
from intergrax.memory.contracts.memory_models import MemoryKind, UserProfileMemoryEntry
from intergrax.memory.contracts.memory_security_governance import (
    MemoryGovernanceDecision,
    MemoryGovernanceDenied,
    MemoryGovernanceEvaluationRequest,
    MemoryGovernanceOperation,
    MemoryGovernanceOutcome,
    MemoryGovernanceReasonCode,
    MemorySecurityStrategySet,
)
from intergrax.memory.contracts.procedural_memory import (
    ProcedureRecord,
    ProcedureStatus,
    ProcedureSupersessionRequest,
    ProcedureTypeRef,
)
from intergrax.memory.default_memory_control_plane import DefaultMemoryControlPlane
from intergrax.memory.entity_memory_indexing import DefaultEntityMemoryIndexer
from intergrax.memory.long_horizon_memory_service import (
    LongHorizonMemoryService,
    build_default_long_horizon_strategies,
)
from intergrax.memory.memory_security_governance_service import (
    MemorySecurityGovernanceService,
    build_default_memory_security_governance_service,
)
from intergrax.memory.procedural_memory_service import (
    ProceduralMemoryService,
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

pytestmark = pytest.mark.gate

_TENANT = "tenant-10b"
_USER = "user-10b"


def _identity(tenant: str = _TENANT, user: str = _USER) -> RequestIdentity:
    return RequestIdentity(tenant_id=tenant, user_id=user)


def _entity_scope() -> EntityMemoryScope:
    return EntityMemoryScope(tenant_id=_TENANT, user_id=_USER)


def _procedure(procedure_id: str = "proc-1") -> ProcedureRecord:
    return ProcedureRecord(
        procedure_id=procedure_id,
        procedure_type=ProcedureTypeRef("test"),
        title="title",
        source_memory_id="mem-src",
        source_memory_revision=1,
        provenance=MemoryProvenance(source_type=MemoryRecordSourceType.USER_EXPLICIT),
        trust=MemoryRecordTrust(trust_class=MemoryTrustClass.USER_EXPLICIT),
    )


@dataclass(frozen=True, slots=True)
class _DenyAllAuthorization:
    policy_id: str = "test.deny"
    policy_version: str = "1"

    def evaluate(self, request: MemoryGovernanceEvaluationRequest) -> MemoryGovernanceDecision:
        return MemoryGovernanceDecision(
            outcome=MemoryGovernanceOutcome.DENY,
            reason_code=MemoryGovernanceReasonCode.GOVERNANCE_DENY,
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            operation=request.context.operation,
        )


def _deny_governance() -> MemorySecurityGovernanceService:
    base = build_default_memory_security_strategy_set()
    strategies = MemorySecurityStrategySet(
        authorization=_DenyAllAuthorization(),
        trust=base.trust,
        admission=base.admission,
        governance=base.governance,
        retention=base.retention,
    )
    return MemorySecurityGovernanceService(strategies=strategies)


def test_entity_projection_allow_mutation() -> None:
    store = InMemoryEntityTemporalMemoryStore()
    indexer = DefaultEntityMemoryIndexer(
        store, security_governance=build_default_memory_security_governance_service()
    )
    scope = _entity_scope()
    entry = UserProfileMemoryEntry(content="hello entity", kind=MemoryKind.USER_FACT)
    indexer.index_memory_entry(_identity(), scope, entry)
    memory_entity_id = entity_memory_entity_id_for_entry(scope, entry.entry_id)
    assert store.get_entity(scope, memory_entity_id) is not None


def test_entity_projection_deny_zero_mutation() -> None:
    store = InMemoryEntityTemporalMemoryStore()
    indexer = DefaultEntityMemoryIndexer(store, security_governance=_deny_governance())
    scope = _entity_scope()
    entry = UserProfileMemoryEntry(content="blocked", kind=MemoryKind.USER_FACT)
    with pytest.raises(MemoryGovernanceDenied):
        indexer.index_memory_entry(_identity(), scope, entry)
    assert store.get_entity(scope, entity_memory_entity_id_for_entry(scope, entry.entry_id)) is None


def test_entity_cross_scope_before_mutation() -> None:
    store = InMemoryEntityTemporalMemoryStore()
    indexer = DefaultEntityMemoryIndexer(
        store, security_governance=build_default_memory_security_governance_service()
    )
    scope = _entity_scope()
    wrong_identity = RequestIdentity(tenant_id="other-tenant", user_id=_USER)
    entry = UserProfileMemoryEntry(content="cross", kind=MemoryKind.USER_FACT)
    with pytest.raises(MemoryGovernanceDenied):
        indexer.index_memory_entry(wrong_identity, scope, entry)
    assert store.get_entity(scope, entity_memory_entity_id_for_entry(scope, entry.entry_id)) is None


def test_procedural_remember_allow() -> None:
    store = InMemoryProceduralMemoryStore()
    service = ProceduralMemoryService(
        _store=store,
        _strategies=build_default_procedural_memory_strategies(),
        _security_governance=build_default_memory_security_governance_service(),
    )
    scope = EntityMemoryScope(tenant_id=_TENANT, user_id=_USER)
    service.remember_procedure(_identity(), scope, _procedure())
    assert store.get_procedure(scope, "proc-1") is not None


def test_procedural_remember_deny() -> None:
    store = InMemoryProceduralMemoryStore()
    service = ProceduralMemoryService(
        _store=store,
        _strategies=build_default_procedural_memory_strategies(),
        _security_governance=_deny_governance(),
    )
    scope = EntityMemoryScope(tenant_id=_TENANT, user_id=_USER)
    with pytest.raises(MemoryGovernanceDenied):
        service.remember_procedure(_identity(), scope, _procedure())
    assert store.get_procedure(scope, "proc-1") is None


def test_procedural_supersede_deny_keeps_active() -> None:
    store = InMemoryProceduralMemoryStore()
    allow = build_default_memory_security_governance_service()
    service = ProceduralMemoryService(
        _store=store,
        _strategies=build_default_procedural_memory_strategies(),
        _security_governance=allow,
    )
    scope = EntityMemoryScope(tenant_id=_TENANT, user_id=_USER)
    service.remember_procedure(_identity(), scope, _procedure("old"))
    service._security_governance = _deny_governance()
    with pytest.raises(MemoryGovernanceDenied):
        service.supersede_procedure(
            _identity(),
            scope,
            ProcedureSupersessionRequest(
                superseded_procedure_id="old",
                superseding_record=_procedure("new"),
            ),
        )
    old = store.get_procedure(scope, "old")
    assert old is not None
    assert old.status is ProcedureStatus.ACTIVE
    assert store.get_procedure(scope, "new") is None


def test_procedural_trust_escalation_denied() -> None:
    store = InMemoryProceduralMemoryStore()
    service = ProceduralMemoryService(
        _store=store,
        _strategies=build_default_procedural_memory_strategies(),
        _security_governance=build_default_memory_security_governance_service(),
    )
    scope = EntityMemoryScope(tenant_id=_TENANT, user_id=_USER)
    escalated = ProcedureRecord(
        procedure_id="proc-esc",
        procedure_type=ProcedureTypeRef("test"),
        title="escalated",
        source_memory_id="mem-src",
        source_memory_revision=1,
        provenance=MemoryProvenance(source_type=MemoryRecordSourceType.SESSION_EXTRACTION),
        trust=MemoryRecordTrust(trust_class=MemoryTrustClass.USER_EXPLICIT),
    )
    with pytest.raises(MemoryGovernanceDenied):
        service.remember_procedure(_identity(), scope, escalated)


class _ScopedAuthority:
    def __init__(self, scope: LongHorizonMemoryScope) -> None:
        self._scope = scope

    def resolve_canonical_source(
        self,
        scope: LongHorizonMemoryScope,
        memory_id: str,
        revision: int,
    ) -> CanonicalMemorySourceSnapshot:
        if scope != self._scope:
            raise ValueError("scope mismatch")
        return CanonicalMemorySourceSnapshot(
            memory_id=memory_id,
            revision=revision,
            content=f"content-{memory_id}",
            observed_at="2025-01-01T00:00:00+00:00",
        )


def _lh_service(
    store: InMemoryLongHorizonMemoryStore | None = None,
    governance: MemorySecurityGovernanceService | None = None,
) -> LongHorizonMemoryService:
    scope = LongHorizonMemoryScope(tenant_id=_TENANT, user_id=_USER)
    return LongHorizonMemoryService(
        _store=store or InMemoryLongHorizonMemoryStore(),
        _strategies=build_default_long_horizon_strategies(),
        _source_authority=_ScopedAuthority(scope),
        _security_governance=governance or build_default_memory_security_governance_service(),
    )


def test_long_horizon_leaf_compaction_allow() -> None:
    service = _lh_service()
    scope = LongHorizonMemoryScope(tenant_id=_TENANT, user_id=_USER)
    request = LongHorizonCompactionRequest(
        identity=_identity(),
        scope=scope,
        target_level=1,
        sources=(
            LongHorizonCompactionSource(
                memory_id="m1",
                revision=1,
                content="alpha content",
                observed_at="2025-01-01T00:00:00+00:00",
            ),
        ),
    )
    result = service.compact(request)
    assert len(result.created) == 1


def test_long_horizon_compaction_deny_batch_failure() -> None:
    service = _lh_service(governance=_deny_governance())
    scope = LongHorizonMemoryScope(tenant_id=_TENANT, user_id=_USER)
    request = LongHorizonCompactionRequest(
        identity=_identity(),
        scope=scope,
        target_level=1,
        sources=(
            LongHorizonCompactionSource(
                memory_id="m1",
                revision=1,
                content="alpha content",
                observed_at="2025-01-01T00:00:00+00:00",
            ),
        ),
    )
    result = service.compact(request)
    assert not result.created
    assert len(result.failures) == 1


def test_long_horizon_restricted_child_source_denied() -> None:
    store = InMemoryLongHorizonMemoryStore()
    service = _lh_service(store=store)
    scope = LongHorizonMemoryScope(tenant_id=_TENANT, user_id=_USER)
    restricted_child = LongHorizonSummaryRecord(
        summary_id="child-r",
        summary_level=1,
        node_kind=SummaryNodeKind.LEAF,
        revision=1,
        content="restricted leaf",
        source_memory_refs=(MemorySourceRef(memory_id="m1", revision=1),),
        source_count=1,
        governance=MemoryRecordGovernance(data_classification=DataClassification.RESTRICTED),
        created_at="2025-01-01T00:00:00+00:00",
    )
    store.upsert_summary(scope, restricted_child)
    request = LongHorizonCompactionRequest(
        identity=_identity(),
        scope=scope,
        target_level=2,
        child_summaries=(restricted_child,),
    )
    result = service.compact(request)
    assert not result.created
    assert result.failures


def test_policy_exception_fail_closed_entity() -> None:
    class _BoomAdmission:
        policy_id = "boom"
        policy_version = "1"

        def evaluate(self, request: MemoryGovernanceEvaluationRequest) -> MemoryGovernanceDecision:
            raise RuntimeError("policy exploded")

    base = build_default_memory_security_strategy_set()
    strategies = MemorySecurityStrategySet(
        authorization=base.authorization,
        trust=base.trust,
        admission=_BoomAdmission(),
        governance=base.governance,
        retention=base.retention,
    )
    store = InMemoryEntityTemporalMemoryStore()
    indexer = DefaultEntityMemoryIndexer(
        store, security_governance=MemorySecurityGovernanceService(strategies=strategies)
    )
    with pytest.raises(MemoryGovernanceDenied):
        indexer.index_memory_entry(
            _identity(),
            _entity_scope(),
            UserProfileMemoryEntry(content="x", kind=MemoryKind.USER_FACT),
        )


def test_shared_governance_instance_in_wiring() -> None:
    env = ApplicationEnvironmentProfile(
        memory_profile=MemoryProfile(
            enable_procedural_memory=True,
            enable_long_horizon_memory=True,
        )
    )
    governance = resolve_memory_security_governance_service()
    procedural = resolve_procedural_memory_capability(env, security_governance=governance)
    assert procedural is not None

    class _Authority:
        def resolve_canonical_source(self, scope, memory_id: str, revision: int):
            return CanonicalMemorySourceSnapshot(
                memory_id=memory_id,
                revision=revision,
                content="c",
                observed_at="2025-01-01T00:00:00+00:00",
            )

    long_horizon = resolve_long_horizon_memory_capability(
        env,
        source_authority=_Authority(),
        security_governance=governance,
    )
    assert long_horizon is not None
    plane = DefaultMemoryControlPlane(security_governance=governance)
    assert procedural._security_governance is governance
    assert long_horizon._security_governance is governance
    assert plane.security_governance is governance


def test_specialized_services_do_not_import_default_policies() -> None:
    forbidden = (
        "DefaultMemoryGovernancePolicy",
        "DefaultMemoryAuthorizationPolicy",
        "DefaultMemoryAdmissionPolicy",
    )
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    paths = (
        root / "intergrax/memory/entity_memory_indexing.py",
        root / "intergrax/memory/procedural_memory_service.py",
        root / "intergrax/memory/long_horizon_memory_service.py",
    )
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    assert alias.name not in forbidden, rel
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name not in forbidden, rel
