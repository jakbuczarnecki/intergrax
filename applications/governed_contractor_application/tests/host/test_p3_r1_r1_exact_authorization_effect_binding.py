# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P3-R1-R1-Q2 — exact authorization → execution → provider → receipt binding."""

from __future__ import annotations

import pytest

from applications.governed_contractor_application.tests.host.test_gr7_a2_external_work_erl_bridge import (
    _PRINCIPAL,
    _TENANT,
    _WORKSPACE,
    _CREATE_IDEMP,
    _build_runtime,
    _create_meta,
    _create_step,
)
from external_contractor_adapter.tests.fakes.deterministic_external_work import (
    DeterministicExternalWorkFake,
)
from intergrax.contracts.execution_identity import mint_attempt_id, mint_execution_id
from intergrax.contracts.governed_execution_governance_evidence import (
    GovernedExecutionEvaluationPoint,
    GovernanceDecisionEvidenceFact,
)
from intergrax.contracts.provider_invocation import ProviderInvocationStatus
from intergrax.runtime.execution_evidence.attestor import build_deterministic_test_attestor
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.runtime.governance.governance_evidence_persistence import (
    InMemoryGovernanceEvidencePersistence,
)
from tests.unit.runtime.governance.gr3_test_support import default_gr3_identity_bundle

pytestmark = [pytest.mark.unit, pytest.mark.gate]

def _build_runtime_with_governance_evidence(
    fake: DeterministicExternalWorkFake,
    task_id: object,
    *,
    deny_create: bool = False,
) -> tuple[object, InMemoryGovernanceEvidencePersistence]:
    persistence = InMemoryGovernanceEvidencePersistence()
    runtime, _ = _build_runtime(
        fake,
        task_id,
        deny_create=deny_create,
        governance_evidence_persistence=persistence,
        attestor=build_deterministic_test_attestor(),
    )
    return runtime, persistence


def _allow_fact_for_execution(
    facts: list[GovernanceDecisionEvidenceFact],
    *,
    execution_id: object,
) -> GovernanceDecisionEvidenceFact:
    matches = [
        fact
        for fact in facts
        if fact.evaluation_point is GovernedExecutionEvaluationPoint.MEANINGFUL_SIDE_EFFECT
        and fact.decision is PolicyAction.ALLOW
        and fact.execution_id is not None
        and str(fact.execution_id) == str(execution_id)
    ]
    assert len(matches) == 1, f"expected one ALLOW MSE fact, got {len(matches)}"
    assert matches[0].has_full_execution_correlation
    return matches[0]


def test_exact_authorization_to_governed_effect_chain() -> None:
    fake = DeterministicExternalWorkFake()
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    runtime, persistence = _build_runtime_with_governance_evidence(fake, task_id)
    step, _ = _create_step(
        runtime, fake, task_id, run_id, attempt_id, execution_id,
    )
    assert step.governed_result is not None
    assert step.receipt is not None
    assert step.attestation is not None and step.attestation.attestation_succeeded

    ger = step.governed_result
    receipt = step.receipt
    event = receipt.execution_boundary_event

    fact = _allow_fact_for_execution(persistence.facts, execution_id=execution_id)

    assert str(fact.task_id) == str(event.execution.task_id)
    assert str(fact.run_id) == str(event.execution.run_id)
    assert str(fact.attempt_id) == str(event.execution.attempt_id)
    assert str(fact.execution_id) == str(event.execution.execution_id)
    assert fact.tenant_id == event.tenant_id
    assert fact.decision is PolicyAction.ALLOW

    assert event.provider_invocation.invocation_id == ger.provider_invocation.invocation_id
    assert ger.provider_outcome.invocation_id == ger.provider_invocation.invocation_id
    assert ger.provider_outcome.status is ProviderInvocationStatus.SUCCEEDED

    policy_decision = ger.evaluated_policy_decision.decision
    assert event.policy.decision_id == policy_decision.decision_id
    assert event.policy.action is PolicyAction.ALLOW

    assert ger.proof is not None
    assert ger.proof.governance_evidence is not None
    assert event.governance_evidence is not None
    assert event.governance_evidence.evidence_id == fact.evidence_id
    assert ger.proof.governance_evidence.evidence_id == fact.evidence_id


def test_cross_execution_authorization_does_not_bind_other_effect() -> None:
    fake = DeterministicExternalWorkFake()
    task_id, run_id, attempt_id, execution_id_e1 = default_gr3_identity_bundle()
    execution_id_e2 = mint_execution_id()
    runtime, persistence = _build_runtime_with_governance_evidence(fake, task_id)

    _create_step(runtime, fake, task_id, run_id, attempt_id, execution_id_e1)
    fact_e1 = _allow_fact_for_execution(persistence.facts, execution_id=execution_id_e1)

    step2, _ = _create_step(
        runtime,
        fake,
        task_id,
        run_id,
        attempt_id,
        execution_id_e2,
        idempotency_key=f"{_CREATE_IDEMP}-e2",
    )
    assert step2.receipt is not None
    event2 = step2.receipt.execution_boundary_event
    assert str(event2.execution.execution_id) == str(execution_id_e2)
    assert str(fact_e1.execution_id) != str(event2.execution.execution_id)
    assert event2.governance_evidence is not None
    assert event2.governance_evidence.evidence_id != fact_e1.evidence_id


def test_cross_attempt_authorization_does_not_bind_other_attempt() -> None:
    fake = DeterministicExternalWorkFake()
    task_id, run_id, attempt_id_a1, execution_id_e1 = default_gr3_identity_bundle()
    attempt_id_a2 = mint_attempt_id()
    execution_id_e2 = mint_execution_id()
    runtime, persistence = _build_runtime_with_governance_evidence(fake, task_id)

    _create_step(runtime, fake, task_id, run_id, attempt_id_a1, execution_id_e1)
    fact_a1 = _allow_fact_for_execution(persistence.facts, execution_id=execution_id_e1)

    step2, _ = _create_step(
        runtime,
        fake,
        task_id,
        run_id,
        attempt_id_a2,
        execution_id_e2,
        idempotency_key=f"{_CREATE_IDEMP}-a2",
    )
    assert step2.receipt is not None
    event2 = step2.receipt.execution_boundary_event
    assert str(event2.execution.attempt_id) == str(attempt_id_a2)
    assert str(fact_a1.attempt_id) != str(event2.execution.attempt_id)
    assert event2.governance_evidence is not None
    assert event2.governance_evidence.evidence_id != fact_a1.evidence_id


def test_deny_authorization_cannot_produce_successful_receipt() -> None:
    fake = DeterministicExternalWorkFake()
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    runtime, persistence = _build_runtime_with_governance_evidence(
        fake, task_id, deny_create=True,
    )
    step, _ = _create_step(
        runtime, fake, task_id, run_id, attempt_id, execution_id,
    )
    assert step.receipt is None
    assert step.governed_result is None
    deny_facts = [
        f
        for f in persistence.facts
        if f.decision is PolicyAction.DENY
        and f.execution_id is not None
        and str(f.execution_id) == str(execution_id)
    ]
    assert deny_facts


def test_tenant_mismatch_blocks_attributable_receipt() -> None:
    from applications.governed_contractor_application.tests.host.test_p3_r1_r1_governed_identity_tenant_gate import (
        test_cross_tenant_negative_no_provider_and_no_ger,
    )

    test_cross_tenant_negative_no_provider_and_no_ger()
