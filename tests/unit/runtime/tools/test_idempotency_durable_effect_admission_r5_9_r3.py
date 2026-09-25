# © Artur Czarnecki. All rights reserved.

"""Durable PRE_EFFECT vs MAY_HAVE_STARTED ledger semantics (R5.9-R3 W2/W3)."""

from __future__ import annotations

import pytest

from intergrax.contracts.idempotency_store import (
    ActiveInvocationClaimError,
    InvocationUncertaintyError,
    PreEffectSuspendedWorkRecoveryAuthority,
)
from intergrax.runtime.tools.idempotency_pre_effect_coordinator import (
    IdempotencyPreEffectCoordinator,
)
from intergrax.runtime.tools.in_memory_idempotency_store import InMemoryIdempotencyStore
from intergrax.runtime.tools.operation_identity import (
    compute_invocation_operation_identity,
)
from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.execution_models import ToolExecutionRequest
from pydantic import BaseModel
from tests.unit.runtime.tools.test_idempotent_invoker import (
    _RUN_ID,
    _state_with_enforce_allow,
)


class _In(BaseModel):
    value: int = 1


class _Out(BaseModel):
    result: int = 1


pytestmark = pytest.mark.unit


def _side_effect_contract() -> ToolContract:
    return ToolContract(
        tool_id="double",
        name="double",
        description="double",
        input_schema=_In,
        output_schema=_Out,
        error_mapping={},
        side_effects=True,
    )


def test_pre_effect_reconcile_clears_only_before_durable_admission() -> None:
    store = InMemoryIdempotencyStore()
    coordinator = IdempotencyPreEffectCoordinator(idempotency_store=store)
    contract = _side_effect_contract()
    request = ToolExecutionRequest(
        run_id=_RUN_ID,
        step_id="step1",
        tool_id=contract.tool_id,
        input=_In(),
        idempotency_key="idem-w2-proof",
    )
    state = _state_with_enforce_allow()
    operation_identity = compute_invocation_operation_identity(
        request.tool_id,
        request.input,
    )
    first = coordinator.before_external_effect(
        state=state,
        contract=contract,
        request=request,
    )
    assert first.claim_context is not None
    authority = PreEffectSuspendedWorkRecoveryAuthority(owner_id="host-b", fence=2)
    assert coordinator._store.reconcile_abandoned_pre_effect_not_started(
        state.tenant_id,
        request.idempotency_key,
        operation_identity,
        recovery_authority=authority,
    )
    second = coordinator.before_external_effect(
        state=state,
        contract=contract,
        request=request,
    )
    assert second.claim_context is not None
    coordinator.admit_external_effect_may_have_started(
        claim_context=second.claim_context,
    )
    assert store.external_effect_may_have_started(
        state.tenant_id,
        request.idempotency_key,
    )
    assert not coordinator._store.reconcile_abandoned_pre_effect_not_started(
        state.tenant_id,
        request.idempotency_key,
        operation_identity,
        recovery_authority=authority,
    )


def test_blocked_claim_with_durable_admission_raises_uncertainty() -> None:
    store = InMemoryIdempotencyStore()
    coordinator = IdempotencyPreEffectCoordinator(idempotency_store=store)
    contract = _side_effect_contract()
    request = ToolExecutionRequest(
        run_id=_RUN_ID,
        step_id="step1",
        tool_id=contract.tool_id,
        input=_In(),
        idempotency_key="idem-w3-proof",
    )
    state = _state_with_enforce_allow()
    first = coordinator.before_external_effect(
        state=state,
        contract=contract,
        request=request,
    )
    assert first.claim_context is not None
    coordinator.admit_external_effect_may_have_started(
        claim_context=first.claim_context,
    )
    with pytest.raises(InvocationUncertaintyError):
        coordinator.before_external_effect(
            state=state,
            contract=contract,
            request=request,
        )


def test_pre_effect_blocked_without_admission_raises_active_claim() -> None:
    store = InMemoryIdempotencyStore()
    coordinator = IdempotencyPreEffectCoordinator(idempotency_store=store)
    contract = _side_effect_contract()
    request = ToolExecutionRequest(
        run_id=_RUN_ID,
        step_id="step1",
        tool_id=contract.tool_id,
        input=_In(),
        idempotency_key="idem-active-claim",
    )
    state = _state_with_enforce_allow()
    first = coordinator.before_external_effect(
        state=state,
        contract=contract,
        request=request,
    )
    assert first.claim_context is not None
    with pytest.raises(ActiveInvocationClaimError):
        coordinator.before_external_effect(
            state=state,
            contract=contract,
            request=request,
        )
