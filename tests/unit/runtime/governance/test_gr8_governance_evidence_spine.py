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
from intergrax.contracts.governed_execution_governance_evidence import (
    GovernedExecutionEvaluationPoint,
    GovernanceEvidencePersistencePort,
    build_governance_fact_from_policy_decision,
)
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
)
from intergrax.runtime.governance.runtime_execution_policy_admission import (
    AllowingRuntimeExecutionPolicyAdmission,
    DenyingRuntimeExecutionPolicyAdmission,
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


def test_custom_persistence_port_via_composition() -> None:
    class _Capturing(GovernanceEvidencePersistencePort):
        def __init__(self) -> None:
            self.captured: list[object] = []

        def persist(self, fact):  # type: ignore[no-untyped-def]
            self.captured.append(fact)
            from intergrax.contracts.governed_execution_governance_evidence import (
                GovernanceEvidencePersistenceOutcome,
            )

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


def test_build_fact_rejects_non_terminal_actions() -> None:
    with pytest.raises(ValueError, match="governance_evidence_requires"):
        build_governance_fact_from_policy_decision(
            evaluation_point=GovernedExecutionEvaluationPoint.MEANINGFUL_SIDE_EFFECT,
            tenant_id="tenant_a",
            workspace_id="ws",
            principal_id="p",
            decision=PolicyDecision(action=PolicyAction.ESCALATE, reason="x"),
            request_digest="sha256:aa",
            idempotency_key="k1",
            action="act",
        )
