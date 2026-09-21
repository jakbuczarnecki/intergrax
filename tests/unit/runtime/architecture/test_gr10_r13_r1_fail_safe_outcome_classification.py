# © Artur Czarnecki. All rights reserved.

"""GR-10-R13-R1 — fail-safe orchestration reliability outcome classification."""

from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path

import pytest

from intergrax.contracts.enterprise_reliability.repeat_eligibility import (
    evaluate_external_effect_repeat_eligibility,
)
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    reset_active_execution_identity,
)
from intergrax.contracts.provider_invocation import ProviderInvocationStatus
from intergrax.runtime.governance.orchestration_consequential_effect_reliability_boundary import (
    OrchestrationConsequentialEffectDefinitiveFailureError,
    OrchestrationConsequentialEffectUncertaintyError,
    ProviderInvocationOrchestrationConsequentialEffectReliabilityBoundary,
    orchestration_slot_invocation_id,
)
from intergrax.runtime.governance.orchestration_consequential_effect_reliability_composition import (
    build_production_orchestration_consequential_effect_reliability_boundary,
)
from testing_support.orchestration.orchestration_consequential_effect_reliability_doubles import (
    DurableTestProviderInvocationStore,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
_REPO_ROOT = Path(__file__).resolve().parents[4]
_BOUNDARY_SOURCE = (
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "governance"
    / "orchestration_consequential_effect_reliability_boundary.py"
)


def _boundary(
    store: DurableTestProviderInvocationStore,
    *,
    tenant_id: str = "platform",
) -> ProviderInvocationOrchestrationConsequentialEffectReliabilityBoundary:
    return ProviderInvocationOrchestrationConsequentialEffectReliabilityBoundary(
        store=store,
        clock=lambda: _T0,
        tenant_id=tenant_id,
    )


def _identity_token():
    return bind_active_execution_identity(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )


@pytest.mark.asyncio
async def test_scenario_a_success_persists_succeeded() -> None:
    store = DurableTestProviderInvocationStore()
    boundary = _boundary(store)
    token = _identity_token()
    calls = 0

    async def _effect() -> str:
        nonlocal calls
        calls += 1
        return "ok"

    try:
        result = await boundary.execute_admitted_effect(
            slot_id="slot-a",
            operation_id="op-success",
            idempotency_key="op-success",
            execute=_effect,
        )
    finally:
        reset_active_execution_identity(token)

    assert result == "ok"
    assert calls == 1
    invocation_id = orchestration_slot_invocation_id(
        tenant_id="platform",
        provider_id="platform.orchestration.topology_slot",
        slot_id="slot-a",
        idempotency_key="op-success",
    )
    outcome = store.get_outcome(invocation_id)
    assert outcome is not None
    assert outcome.status is ProviderInvocationStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_scenario_b_definitive_failure_persists_failed() -> None:
    store = DurableTestProviderInvocationStore()
    boundary = _boundary(store)
    token = _identity_token()
    calls = 0

    async def _effect() -> None:
        nonlocal calls
        calls += 1
        raise OrchestrationConsequentialEffectDefinitiveFailureError("provider rejected")

    try:
        with pytest.raises(OrchestrationConsequentialEffectDefinitiveFailureError):
            await boundary.execute_admitted_effect(
                slot_id="slot-a",
                operation_id="op-fail",
                idempotency_key="op-fail",
                execute=_effect,
            )
    finally:
        reset_active_execution_identity(token)

    assert calls == 1
    invocation_id = orchestration_slot_invocation_id(
        tenant_id="platform",
        provider_id="platform.orchestration.topology_slot",
        slot_id="slot-a",
        idempotency_key="op-fail",
    )
    outcome = store.get_outcome(invocation_id)
    assert outcome is not None
    assert outcome.status is ProviderInvocationStatus.FAILED


@pytest.mark.asyncio
async def test_scenario_d_connection_error_unknown_not_failed() -> None:
    store = DurableTestProviderInvocationStore()
    boundary = _boundary(store)
    token = _identity_token()
    calls = 0

    async def _effect() -> None:
        nonlocal calls
        calls += 1
        raise ConnectionError("transport reset")

    try:
        with pytest.raises(OrchestrationConsequentialEffectUncertaintyError):
            await boundary.execute_admitted_effect(
                slot_id="slot-a",
                operation_id="op-conn",
                idempotency_key="op-conn",
                execute=_effect,
            )
    finally:
        reset_active_execution_identity(token)

    assert calls == 1
    invocation_id = orchestration_slot_invocation_id(
        tenant_id="platform",
        provider_id="platform.orchestration.topology_slot",
        slot_id="slot-a",
        idempotency_key="op-conn",
    )
    outcome = store.get_outcome(invocation_id)
    assert outcome is not None
    assert outcome.status is ProviderInvocationStatus.UNKNOWN


@pytest.mark.asyncio
async def test_scenario_unknown_no_blind_retry_on_repeat() -> None:
    store = DurableTestProviderInvocationStore()
    boundary = _boundary(store)
    token = _identity_token()
    calls = 0

    async def _effect() -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("sdk failure after send")

    try:
        with pytest.raises(OrchestrationConsequentialEffectUncertaintyError):
            await boundary.execute_admitted_effect(
                slot_id="slot-a",
                operation_id="op-unknown",
                idempotency_key="op-unknown",
                execute=_effect,
            )
        with pytest.raises(OrchestrationConsequentialEffectUncertaintyError):
            await boundary.execute_admitted_effect(
                slot_id="slot-a",
                operation_id="op-unknown",
                idempotency_key="op-unknown",
                execute=_effect,
            )
    finally:
        reset_active_execution_identity(token)

    assert calls == 1


@pytest.mark.asyncio
async def test_scenario_h_crash_intent_without_outcome_no_replay() -> None:
    store = DurableTestProviderInvocationStore()
    boundary = _boundary(store)
    token = _identity_token()
    calls = 0

    async def _effect() -> None:
        nonlocal calls
        calls += 1
        return "ok"

    invocation_id = orchestration_slot_invocation_id(
        tenant_id="platform",
        provider_id="platform.orchestration.topology_slot",
        slot_id="slot-a",
        idempotency_key="op-crash",
    )
    seed = boundary._mint_invocation(
        slot_id="slot-a",
        operation_id="op-crash",
        idempotency_key="op-crash",
    )
    store.put_invocation(seed)

    try:
        with pytest.raises(OrchestrationConsequentialEffectUncertaintyError):
            await boundary.execute_admitted_effect(
                slot_id="slot-a",
                operation_id="op-crash",
                idempotency_key="op-crash",
                execute=_effect,
            )
    finally:
        reset_active_execution_identity(token)

    assert calls == 0
    assert store.get_outcome(invocation_id) is None


def test_scenario_j_cross_tenant_invocation_identity() -> None:
    key = "shared-key"
    tenant_a = orchestration_slot_invocation_id(
        tenant_id="tenant-a",
        provider_id="platform.orchestration.topology_slot",
        slot_id="slot-a",
        idempotency_key=key,
    )
    tenant_b = orchestration_slot_invocation_id(
        tenant_id="tenant-b",
        provider_id="platform.orchestration.topology_slot",
        slot_id="slot-a",
        idempotency_key=key,
    )
    assert tenant_a != tenant_b


def test_scenario_k_cross_slot_invocation_identity() -> None:
    key = "shared-key"
    slot_a = orchestration_slot_invocation_id(
        tenant_id="platform",
        provider_id="platform.orchestration.topology_slot",
        slot_id="slot-a",
        idempotency_key=key,
    )
    slot_b = orchestration_slot_invocation_id(
        tenant_id="platform",
        provider_id="platform.orchestration.topology_slot",
        slot_id="slot-b",
        idempotency_key=key,
    )
    assert slot_a != slot_b


def test_custom_durable_store_production_composition() -> None:
    store = DurableTestProviderInvocationStore()
    boundary = build_production_orchestration_consequential_effect_reliability_boundary(
        provider_invocation_store=store,
        clock=lambda: _T0,
        tenant_id="t1",
    )
    assert isinstance(
        boundary,
        ProviderInvocationOrchestrationConsequentialEffectReliabilityBoundary,
    )


def test_static_gate_no_post_dispatch_exception_to_failed() -> None:
    source = _BOUNDARY_SOURCE.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(_BOUNDARY_SOURCE))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if node.type is None:
            continue
        if not isinstance(node.type, ast.Name) or node.type.id != "Exception":
            continue
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Attribute)
                and isinstance(child.value, ast.Name)
                and child.value.id == "ProviderInvocationStatus"
                and child.attr == "FAILED"
            ):
                pytest.fail("post-dispatch except Exception must not assign FAILED")


def test_static_gate_definitive_failure_has_typed_handler() -> None:
    source = _BOUNDARY_SOURCE.read_text(encoding="utf-8-sig")
    assert "OrchestrationConsequentialEffectDefinitiveFailureError" in source
    assert "ProviderInvocationStatus.UNKNOWN" in source
    assert "ProviderInvocationStatus.FAILED" in source


def test_unknown_repeat_eligibility_requires_reconciliation() -> None:
    from intergrax.contracts.enterprise_reliability.effect_contract import (
        ExternalEffectCategory,
        ExternalEffectCapabilitySupport,
        ExternalEffectContract,
        ExternalEffectSafetyCapabilities,
        UnknownUncertaintyPosture,
        evaluate_unknown_uncertainty_posture,
    )
    from intergrax.contracts.enterprise_reliability.repeat_eligibility import (
        ExternalEffectRepeatEligibilityReason,
        ExternalEffectRepeatEligibilityRequest,
        ExternalEffectRepeatEligibilityVerdict,
    )
    from intergrax.contracts.provider_invocation import (
        ProviderInvocation,
        ProviderInvocationOutcome,
    )

    contract = ExternalEffectContract(
        contract_id="c-orch",
        operation_key="topology_slot:slot-a",
        category=ExternalEffectCategory.INFRASTRUCTURE,
        safety=ExternalEffectSafetyCapabilities(
            idempotency=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
            reconciliation=ExternalEffectCapabilitySupport.SUPPORTED,
            compensation=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
        ),
        reconciliation_probe_refs=("probe",),
    )
    assert evaluate_unknown_uncertainty_posture(contract) is (
        UnknownUncertaintyPosture.RECONCILE_ONLY
    )
    invocation = ProviderInvocation(
        invocation_id="orch-slot:test",
        provider_id="platform.orchestration.topology_slot",
        operation="topology_slot:slot-a",
        task_id="op",
        run_id="run",
        idempotency_key="key",
        request_digest="digest",
        started_at=_T0,
    )
    outcome = ProviderInvocationOutcome(
        invocation_id=invocation.invocation_id,
        status=ProviderInvocationStatus.UNKNOWN,
        completed_at=_T0,
        response_digest="out",
    )
    decision = evaluate_external_effect_repeat_eligibility(
        ExternalEffectRepeatEligibilityRequest(
            invocation=invocation,
            outcome=outcome,
            effect_contract=contract,
        ),
    )
    assert decision.verdict is ExternalEffectRepeatEligibilityVerdict.NOT_ALLOWED
    assert decision.reason in {
        ExternalEffectRepeatEligibilityReason.DENIED_RECONCILIATION_REQUIRED,
        ExternalEffectRepeatEligibilityReason.DENIED_IDEMPOTENCY_NOT_SUPPORTED,
    }
