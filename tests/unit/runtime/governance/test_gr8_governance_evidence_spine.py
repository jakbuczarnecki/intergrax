# © Artur Czarnecki. All rights reserved.

"""GR-8 — Governance Evidence Integration qualification."""

from __future__ import annotations

import pytest

from intergrax.contracts.collaborative_work import EffectiveAuthorityDecision
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.evaluated_policy_decision import request_digest_for_payload
from intergrax.contracts.execution_evidence.persistence_port import EvidencePersistencePort
from intergrax.contracts.governed_execution_governance_evidence import (
    GovernedExecutionEvaluationPoint,
    GovernanceDecisionEvidenceFact,
    GovernanceEvidencePersistenceOutcome,
    GovernanceEvidencePersistencePort,
    build_governance_fact_from_policy_decision,
    governance_evidence_id_from_idempotency,
)
from intergrax.contracts.runtime_event_type import RuntimeEventType
from intergrax.runtime.events.evidence_persistence_adapter import as_evidence_persistence_port
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.contracts.root_execution_operation import RootExecutionOperation
from intergrax.contracts.runtime_execution_admission import (
    RootExecutionAuthorityAdmissionDisposition,
    RootExecutionAuthorityAdmissionRequest,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.governance.execution_admission_composition import (
    build_root_execution_authority_admission,
)
from intergrax.runtime.governance.governance_evidence_composition import (
    build_governance_evidence_recorder,
    build_in_memory_governance_evidence_persistence,
    build_runtime_event_governance_evidence_persistence,
)
from intergrax.runtime.governance.governance_evidence_persistence import (
    InMemoryGovernanceEvidencePersistence,
    RuntimeEventGovernanceEvidencePersistence,
)
from intergrax.runtime.governance.runtime_execution_policy_admission import (
    AllowingRuntimeExecutionPolicyAdmission,
    DenyingRuntimeExecutionPolicyAdmission,
    EscalateRuntimeExecutionPolicyAdmission,
    RequireHumanRuntimeExecutionPolicyAdmission,
)

pytestmark = pytest.mark.unit


def _admission_request(**overrides: object) -> RootExecutionAuthorityAdmissionRequest:
    base = {
        "tenant_id": "tenant_a",
        "workspace_id": "workspace_x",
        "principal_id": "principal_1",
        "collaborative_authority_scopes": ("workspace.read", "workspace.write"),
        "effective_authority_decision": EffectiveAuthorityDecision(
            decision=PolicyDecision(action=PolicyAction.ALLOW, reason="collaborative"),
        ),
        "root_execution_operation": RootExecutionOperation.ROOT_AGENT,
        "task_id": mint_task_id(),
        "run_id": mint_run_id(),
        "attempt_id": mint_attempt_id(),
        "execution_id": mint_execution_id(),
    }
    base.update(overrides)
    return RootExecutionAuthorityAdmissionRequest(**base)


def test_root_allow_emits_exactly_one_governance_fact() -> None:
    store = build_in_memory_governance_evidence_persistence()
    recorder = build_governance_evidence_recorder(persistence=store)
    service = build_root_execution_authority_admission(
        runtime_policy_admission=AllowingRuntimeExecutionPolicyAdmission(),
        governance_evidence_recorder=recorder,
    )
    result = service.authorize(_admission_request())
    assert result.disposition is RootExecutionAuthorityAdmissionDisposition.ALLOWED
    assert len(store.facts) == 1
    fact = store.facts[0]
    assert fact.evaluation_point is GovernedExecutionEvaluationPoint.ROOT_EXECUTION_ADMISSION
    assert fact.decision is PolicyAction.ALLOW


def test_root_deny_emits_fact_without_trusted_authority() -> None:
    store = build_in_memory_governance_evidence_persistence()
    recorder = build_governance_evidence_recorder(persistence=store)
    service = build_root_execution_authority_admission(
        runtime_policy_admission=DenyingRuntimeExecutionPolicyAdmission(),
        governance_evidence_recorder=recorder,
    )
    result = service.authorize(_admission_request())
    assert result.disposition is RootExecutionAuthorityAdmissionDisposition.DENIED
    assert result.trusted_parent_execution_authority is None
    assert len(store.facts) == 1
    assert store.facts[0].decision is PolicyAction.DENY


def test_escalate_root_emits_fact_without_trusted_authority() -> None:
    store = build_in_memory_governance_evidence_persistence()
    recorder = build_governance_evidence_recorder(persistence=store)
    service = build_root_execution_authority_admission(
        runtime_policy_admission=EscalateRuntimeExecutionPolicyAdmission(),
        governance_evidence_recorder=recorder,
    )
    result = service.authorize(_admission_request())
    assert result.disposition is RootExecutionAuthorityAdmissionDisposition.ESCALATE
    assert result.trusted_parent_execution_authority is None
    assert len(store.facts) == 1
    assert store.facts[0].decision is PolicyAction.ESCALATE


def test_persistence_failure_does_not_flip_escalate_to_allow() -> None:
    store = build_in_memory_governance_evidence_persistence()
    store.fail_on_persist = True
    recorder = build_governance_evidence_recorder(persistence=store)
    service = build_root_execution_authority_admission(
        runtime_policy_admission=EscalateRuntimeExecutionPolicyAdmission(),
        governance_evidence_recorder=recorder,
    )
    result = service.authorize(_admission_request())
    assert result.disposition is RootExecutionAuthorityAdmissionDisposition.ESCALATE


def test_require_human_root_emits_fact() -> None:
    store = build_in_memory_governance_evidence_persistence()
    recorder = build_governance_evidence_recorder(persistence=store)
    service = build_root_execution_authority_admission(
        runtime_policy_admission=RequireHumanRuntimeExecutionPolicyAdmission(),
        governance_evidence_recorder=recorder,
    )
    result = service.authorize(_admission_request())
    assert result.disposition is RootExecutionAuthorityAdmissionDisposition.REQUIRE_HUMAN
    assert store.facts[0].decision is PolicyAction.REQUIRE_HUMAN


def test_persistence_failure_does_not_flip_allow_to_deny() -> None:
    store = build_in_memory_governance_evidence_persistence()
    store.fail_on_persist = True
    recorder = build_governance_evidence_recorder(persistence=store)
    service = build_root_execution_authority_admission(
        runtime_policy_admission=AllowingRuntimeExecutionPolicyAdmission(),
        governance_evidence_recorder=recorder,
    )
    result = service.authorize(_admission_request())
    assert result.disposition is RootExecutionAuthorityAdmissionDisposition.ALLOWED


def test_persistence_failure_does_not_flip_deny_to_allow() -> None:
    store = build_in_memory_governance_evidence_persistence()
    store.fail_on_persist = True
    recorder = build_governance_evidence_recorder(persistence=store)
    service = build_root_execution_authority_admission(
        runtime_policy_admission=DenyingRuntimeExecutionPolicyAdmission(),
        governance_evidence_recorder=recorder,
    )
    result = service.authorize(_admission_request())
    assert result.disposition is RootExecutionAuthorityAdmissionDisposition.DENIED
    assert result.trusted_parent_execution_authority is None


def test_persistence_failure_does_not_flip_require_human_to_allow() -> None:
    store = build_in_memory_governance_evidence_persistence()
    store.fail_on_persist = True
    recorder = build_governance_evidence_recorder(persistence=store)
    service = build_root_execution_authority_admission(
        runtime_policy_admission=RequireHumanRuntimeExecutionPolicyAdmission(),
        governance_evidence_recorder=recorder,
    )
    result = service.authorize(_admission_request())
    assert result.disposition is RootExecutionAuthorityAdmissionDisposition.REQUIRE_HUMAN


def test_idempotent_replay_does_not_duplicate_facts() -> None:
    store = build_in_memory_governance_evidence_persistence()
    recorder = build_governance_evidence_recorder(persistence=store)
    service = build_root_execution_authority_admission(
        runtime_policy_admission=AllowingRuntimeExecutionPolicyAdmission(),
        governance_evidence_recorder=recorder,
    )
    request = _admission_request()
    service.authorize(request)
    service.authorize(request)
    assert len(store.facts) == 1


def test_cross_tenant_facts_isolated_in_memory_store() -> None:
    store = build_in_memory_governance_evidence_persistence()
    recorder = build_governance_evidence_recorder(persistence=store)
    service_a = build_root_execution_authority_admission(
        runtime_policy_admission=AllowingRuntimeExecutionPolicyAdmission(),
        governance_evidence_recorder=recorder,
    )
    service_a.authorize(_admission_request(tenant_id="tenant_a"))
    service_a.authorize(_admission_request(tenant_id="tenant_b"))
    tenants = {fact.tenant_id for fact in store.facts}
    assert tenants == {"tenant_a", "tenant_b"}


class _AppendCountingEvidencePersistencePort:
    """Test double — canonical ``EvidencePersistencePort`` with append accounting."""

    __slots__ = ("_delegate", "append_calls")

    def __init__(self, delegate: EvidencePersistencePort) -> None:
        self._delegate = delegate
        self.append_calls = 0

    def append(self, event, *, tenant_id: str):
        self.append_calls += 1
        return self._delegate.append(event, tenant_id=tenant_id)

    def list_positioned_for_run(self, run_id: str, *, tenant_id: str, limit: int = 1000, through=None, after=None):
        return self._delegate.list_positioned_for_run(
            run_id,
            tenant_id=tenant_id,
            limit=limit,
            through=through,
            after=after,
        )

    def list_for_task(self, task_id: str, *, tenant_id: str, limit: int = 1000):
        return self._delegate.list_for_task(task_id, tenant_id=tenant_id, limit=limit)

    def list_positioned_for_task_grouped_by_run(self, task_id: str, *, tenant_id: str, limit: int = 1000):
        return self._delegate.list_positioned_for_task_grouped_by_run(
            task_id,
            tenant_id=tenant_id,
            limit=limit,
        )

    def get_by_event_id(self, *, tenant_id: str, event_id):
        return self._delegate.get_by_event_id(tenant_id=tenant_id, event_id=event_id)

    def list_positioned_through(self, boundary, *, tenant_id: str, limit: int = 1000):
        return self._delegate.list_positioned_through(boundary, tenant_id=tenant_id, limit=limit)


def test_root_governance_fact_persists_as_policy_decision_runtime_event() -> None:
    memory_store = InMemoryRuntimeEventStore()
    evidence_port = _AppendCountingEvidencePersistencePort(
        as_evidence_persistence_port(memory_store)
    )
    gov_persistence = build_runtime_event_governance_evidence_persistence(
        evidence_persistence=evidence_port,
    )
    recorder = build_governance_evidence_recorder(persistence=gov_persistence)
    service = build_root_execution_authority_admission(
        runtime_policy_admission=AllowingRuntimeExecutionPolicyAdmission(),
        governance_evidence_recorder=recorder,
    )
    request = _admission_request()
    result = service.authorize(request)
    assert result.disposition is RootExecutionAuthorityAdmissionDisposition.ALLOWED
    assert evidence_port.append_calls == 1
    events = memory_store.list_for_task(str(request.task_id), tenant_id=request.tenant_id)
    assert len(events) == 1
    event = events[0]
    assert event.event_type is RuntimeEventType.POLICY_DECISION
    assert event.tenant_id == request.tenant_id
    assert event.task_id == request.task_id
    assert event.run_id == request.run_id
    assert event.attempt_id == request.attempt_id
    assert event.execution_id == request.execution_id
    payload = event.payload
    assert payload["governance_evidence_schema"] == "governed_execution_governance_decision_fact.v1"
    assert payload["evidence_id"]
    assert payload["decision"] == PolicyAction.ALLOW.value
    assert payload["evaluation_point"] == GovernedExecutionEvaluationPoint.ROOT_EXECUTION_ADMISSION.value
    assert "credentials" not in payload
    assert "raw_prompt" not in payload
    assert "auth_token" not in payload
    first_event_id = event.event_id
    replay = service.authorize(request)
    assert replay.disposition is RootExecutionAuthorityAdmissionDisposition.ALLOWED
    assert evidence_port.append_calls == 2
    events_after_replay = memory_store.list_for_task(
        str(request.task_id),
        tenant_id=request.tenant_id,
    )
    assert len(events_after_replay) == 1
    assert events_after_replay[0].event_id == first_event_id


def test_runtime_event_governance_evidence_rejects_incomplete_execution_correlation() -> None:
    memory_store = InMemoryRuntimeEventStore()
    evidence_port = _AppendCountingEvidencePersistencePort(
        as_evidence_persistence_port(memory_store)
    )
    adapter = RuntimeEventGovernanceEvidencePersistence(evidence_persistence=evidence_port)
    digest = request_digest_for_payload({"tenant_id": "tenant_a", "op": "root"})
    idempotency_key = f"root_admission:{digest}:allow:rule"
    fact = build_governance_fact_from_policy_decision(
        evaluation_point=GovernedExecutionEvaluationPoint.ROOT_EXECUTION_ADMISSION,
        tenant_id="tenant_a",
        workspace_id="ws",
        principal_id="principal",
        decision=PolicyDecision(action=PolicyAction.ALLOW, reason="ok"),
        request_digest=digest,
        idempotency_key=idempotency_key,
        action="root_agent",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=None,
    )
    outcome = adapter.persist(fact)
    assert outcome.persisted is False
    assert outcome.error_code == "GovernanceEvidenceIncompleteCorrelation"
    assert evidence_port.append_calls == 0
    assert memory_store.list_for_task(str(fact.task_id), tenant_id=fact.tenant_id) == []


def test_reference_store_idempotency_conflict_on_same_evidence_id() -> None:
    store = InMemoryGovernanceEvidencePersistence()
    digest = request_digest_for_payload({"k": "v"})
    key = "mse:conflict:deny"
    evidence_id = governance_evidence_id_from_idempotency(key)
    first = build_governance_fact_from_policy_decision(
        evaluation_point=GovernedExecutionEvaluationPoint.MEANINGFUL_SIDE_EFFECT,
        tenant_id="tenant_a",
        workspace_id="ws",
        principal_id="p",
        decision=PolicyDecision(action=PolicyAction.DENY, reason="one"),
        request_digest=digest,
        idempotency_key=key,
        action="effect",
    )
    second = build_governance_fact_from_policy_decision(
        evaluation_point=GovernedExecutionEvaluationPoint.MEANINGFUL_SIDE_EFFECT,
        tenant_id="tenant_a",
        workspace_id="ws",
        principal_id="p",
        decision=PolicyDecision(action=PolicyAction.DENY, reason="two"),
        request_digest=digest,
        idempotency_key=key,
        action="effect",
    )
    assert first.evidence_id == evidence_id == second.evidence_id
    assert first != second
    ok = store.persist(first)
    assert ok.persisted is True
    conflict = store.persist(second)
    assert conflict.persisted is False
    assert conflict.error_code == "GovernanceEvidenceIdempotencyConflict"
    assert len(store.facts) == 1


def test_custom_persistence_port_via_composition() -> None:
    class _Capturing(GovernanceEvidencePersistencePort):
        def __init__(self) -> None:
            self.captured: list[GovernanceDecisionEvidenceFact] = []

        def persist(
            self,
            fact: GovernanceDecisionEvidenceFact,
        ) -> GovernanceEvidencePersistenceOutcome:
            self.captured.append(fact)
            return GovernanceEvidencePersistenceOutcome(
                persisted=True,
                evidence_id=fact.evidence_id,
            )

    capture = _Capturing()
    recorder = build_governance_evidence_recorder(persistence=capture)
    service = build_root_execution_authority_admission(
        runtime_policy_admission=DenyingRuntimeExecutionPolicyAdmission(),
        governance_evidence_recorder=recorder,
    )
    service.authorize(_admission_request())
    assert len(capture.captured) == 1


def test_governance_persistence_port_has_no_policy_decision_return() -> None:
    import inspect

    from intergrax.contracts.governed_execution_governance_evidence import (
        GovernanceEvidencePersistencePort,
    )

    hints = inspect.signature(GovernanceEvidencePersistencePort.persist).return_annotation
    assert "PolicyDecision" not in str(hints)


def test_build_fact_rejects_modify_action() -> None:
    with pytest.raises(ValueError, match="governance_evidence_requires"):
        build_governance_fact_from_policy_decision(
            evaluation_point=GovernedExecutionEvaluationPoint.MEANINGFUL_SIDE_EFFECT,
            tenant_id="tenant_a",
            workspace_id="ws",
            principal_id="p",
            decision=PolicyDecision(action=PolicyAction.MODIFY, reason="x"),
            request_digest="sha256:" + "aa" * 32,
            idempotency_key="k1",
            action="act",
        )


def test_build_fact_accepts_escalate_action() -> None:
    fact = build_governance_fact_from_policy_decision(
        evaluation_point=GovernedExecutionEvaluationPoint.MEANINGFUL_SIDE_EFFECT,
        tenant_id="tenant_a",
        workspace_id="ws",
        principal_id="p",
        decision=PolicyDecision(action=PolicyAction.ESCALATE, reason="escalate"),
        request_digest="sha256:" + "bb" * 32,
        idempotency_key="k-escalate",
        action="act",
    )
    assert fact.decision is PolicyAction.ESCALATE
