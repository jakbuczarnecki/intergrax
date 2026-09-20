# © Artur Czarnecki. All rights reserved.

"""GR-10-R13 — orchestration consequential reliability e2e scenarios."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    reset_active_execution_identity,
)
from intergrax.contracts.provider_invocation import ProviderInvocationStatus
from intergrax.runtime.execution.orchestration_topology_slot_mse_enforcement import (
    OrchestrationTopologyReliabilityCompositionError,
    build_orchestration_topology_slot_mse_policy,
)
from intergrax.runtime.governance.orchestration_consequential_effect_reliability_boundary import (
    OrchestrationConsequentialEffectUncertaintyError,
    ProviderInvocationOrchestrationConsequentialEffectReliabilityBoundary,
)
from intergrax.runtime.governance.orchestration_consequential_effect_reliability_composition import (
    OrchestrationConsequentialEffectReliabilityCompositionError,
    build_production_orchestration_consequential_effect_reliability_boundary,
)
from testing_support.orchestration.orchestration_consequential_effect_reliability_doubles import (
    DurableTestProviderInvocationStore,
    PassthroughOrchestrationConsequentialEffectReliability,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_production_topology_policy_requires_reliability_port() -> None:
    with pytest.raises(OrchestrationTopologyReliabilityCompositionError):
        build_orchestration_topology_slot_mse_policy(
            meaningful_side_effect_authorization=None,
            production_mode=True,
        )


def test_production_composition_requires_durable_store() -> None:
    from applications.governed_contractor_application.host.stores import (
        InMemoryProviderInvocationStore,
    )

    with pytest.raises(OrchestrationConsequentialEffectReliabilityCompositionError):
        build_production_orchestration_consequential_effect_reliability_boundary(
            provider_invocation_store=InMemoryProviderInvocationStore(),
            clock=lambda: _T0,
            tenant_id="t1",
        )


@pytest.mark.asyncio
async def test_scenario_c_timeout_unknown_persisted() -> None:
    store = DurableTestProviderInvocationStore()
    boundary = ProviderInvocationOrchestrationConsequentialEffectReliabilityBoundary(
        store=store,
        clock=lambda: _T0,
    )
    token = bind_active_execution_identity(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )
    calls = 0

    async def _effect() -> None:
        nonlocal calls
        calls += 1
        raise TimeoutError("provider timeout")

    try:
        with pytest.raises(OrchestrationConsequentialEffectUncertaintyError):
            await boundary.execute_admitted_effect(
                slot_id="slot-a",
                operation_id="op-1",
                idempotency_key="op-1",
                execute=_effect,
            )
    finally:
        reset_active_execution_identity(token)

    assert calls == 1
    outcome = store.get_outcome("orch-slot:op-1")
    assert outcome is not None
    assert outcome.status is ProviderInvocationStatus.UNKNOWN


@pytest.mark.asyncio
async def test_scenario_g_stable_idempotency_key() -> None:
    passthrough = PassthroughOrchestrationConsequentialEffectReliability()

    async def _effect() -> str:
        return "ok"

    await passthrough.execute_admitted_effect(
        slot_id="slot-a",
        operation_id="op-stable",
        idempotency_key="op-stable",
        execute=_effect,
    )
    await passthrough.execute_admitted_effect(
        slot_id="slot-a",
        operation_id="op-stable",
        idempotency_key="op-stable",
        execute=_effect,
    )
    assert passthrough.last_idempotency_key == "op-stable"
    assert passthrough.calls == 2
