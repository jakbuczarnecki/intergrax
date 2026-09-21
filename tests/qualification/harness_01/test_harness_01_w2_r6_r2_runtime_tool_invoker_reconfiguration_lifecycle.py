# © Artur Czarnecki. All rights reserved.

"""HARNESS-01-R5-W2-R6-R2 — RuntimeToolInvoker reconfiguration lifecycle ownership."""

from __future__ import annotations

import pytest

from intergrax.contracts.dependency_concurrency_admission import (
    DependencyConcurrencyAdmissionRequest,
    DependencyConcurrencyIdentity,
    DependencyConcurrencyKind,
    DependencyConcurrencyOverloadMode,
    DependencyConcurrencyPolicy,
)
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.runtime_tool_invoker_composition import (
    build_production_runtime_tool_invoker,
)
from intergrax.runtime.resilience.dependency_attempt_execution_boundary import (
    DependencyAttemptExecutionBoundary,
    DependencyAttemptExecutionBoundaryClosedError,
)
from intergrax.runtime.resilience.local_dependency_concurrency_admission import (
    LocalDependencyConcurrencyAdmission,
)
from intergrax.runtime.tools.in_memory_idempotency_store import InMemoryIdempotencyStore
from tests.unit.runtime.nexus.tools.conftest import FakeRegistry
from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from pydantic import BaseModel

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _In(BaseModel):
    value: int


class _Out(BaseModel):
    value: int


def _minimal_contract() -> ToolContract:
    return ToolContract(
        tool_id="probe.tool",
        name="probe",
        description="probe",
        input_schema=_In,
        output_schema=_Out,
        side_effects=False,
        error_mapping={},
        risk_level=ToolRiskLevel.LOW,
    )


def _admission_policy() -> dict[DependencyConcurrencyIdentity, DependencyConcurrencyPolicy]:
    identity = DependencyConcurrencyIdentity(
        kind=DependencyConcurrencyKind.TOOL,
        value="probe.tool",
    )
    return {
        identity: DependencyConcurrencyPolicy(
            max_concurrent_calls=4,
            overload_mode=DependencyConcurrencyOverloadMode.REJECT,
        ),
    }


def _admission_request() -> DependencyConcurrencyAdmissionRequest:
    return DependencyConcurrencyAdmissionRequest(
        dependency=DependencyConcurrencyIdentity(
            kind=DependencyConcurrencyKind.TOOL,
            value="probe.tool",
        ),
        tenant_id="tenant-r6r2",
    )


def _boundary_acquire_and_release(boundary: DependencyAttemptExecutionBoundary) -> None:
    handle = boundary.acquire(_admission_request())
    boundary.complete_direct(handle)


def _invoker_with_boundary() -> tuple[RuntimeToolInvoker, DependencyAttemptExecutionBoundary]:
    boundary = DependencyAttemptExecutionBoundary(
        LocalDependencyConcurrencyAdmission(_admission_policy()),
    )
    registry = FakeRegistry(_minimal_contract())
    invoker = build_production_runtime_tool_invoker(
        registry=registry,
        dependency_attempt_boundary=boundary,
    )
    return invoker, boundary


def test_harness_01_w2_r6_r2_boundary_operational_after_reconfiguration() -> None:
    invoker, boundary = _invoker_with_boundary()
    replacement = invoker.with_idempotency_store(InMemoryIdempotencyStore())
    assert replacement._dependency_attempt_boundary is boundary
    _boundary_acquire_and_release(boundary)
    assert invoker._dependency_attempt_boundary is None


def test_harness_01_w2_r6_r2_source_pool_closed_and_boundary_transferred() -> None:
    invoker, boundary = _invoker_with_boundary()
    replacement = invoker.with_idempotency_store(InMemoryIdempotencyStore())
    assert invoker._execution_pool_closed is True
    assert invoker._dependency_attempt_boundary is None
    assert replacement._dependency_attempt_boundary is boundary
    assert replacement._execution_pool_closed is False


def test_harness_01_w2_r6_r2_replacement_close_shuts_down_boundary_once() -> None:
    invoker, boundary = _invoker_with_boundary()
    replacement = invoker.with_idempotency_store(InMemoryIdempotencyStore())
    replacement.close()
    with pytest.raises(DependencyAttemptExecutionBoundaryClosedError):
        boundary.acquire(_admission_request())


def test_harness_01_w2_r6_r2_replacement_double_close_is_idempotent() -> None:
    invoker, _boundary = _invoker_with_boundary()
    replacement = invoker.with_idempotency_store(InMemoryIdempotencyStore())
    replacement.close()
    replacement.close()


def test_harness_01_w2_r6_r2_already_configured_is_noop_for_lifecycle() -> None:
    invoker, boundary = _invoker_with_boundary()
    configured = invoker.with_idempotency_store(InMemoryIdempotencyStore())
    again = configured.with_idempotency_store(InMemoryIdempotencyStore())
    assert again is configured
    assert configured._dependency_attempt_boundary is boundary
    assert configured._execution_pool_closed is False
    configured.close()


def test_harness_01_w2_r6_r2_reconfiguration_preserves_shared_dependencies() -> None:
    invoker, boundary = _invoker_with_boundary()
    registry = invoker.registry
    executor = invoker._executor
    scope_policy = invoker._scope_policy
    sandbox = invoker._sandbox_availability
    governance = invoker._agent_runtime_governance
    guard = invoker._inner_execution_guard
    mse = invoker._meaningful_side_effect_authorization
    store = invoker._external_operation_store
    owner = invoker._external_operation_owner
    cancellation = invoker._external_operation_cancellation_port
    wiring = invoker._invocation_wiring_resolver

    replacement = invoker.with_idempotency_store(InMemoryIdempotencyStore())

    assert replacement.registry is registry
    assert replacement._executor is executor
    assert replacement._scope_policy is scope_policy
    assert replacement._sandbox_availability is sandbox
    assert replacement._agent_runtime_governance is governance
    assert replacement._inner_execution_guard is guard
    assert replacement._meaningful_side_effect_authorization is mse
    assert replacement._external_operation_store is store
    assert replacement._external_operation_owner is owner
    assert replacement._external_operation_cancellation_port is cancellation
    assert replacement._invocation_wiring_resolver is wiring
    assert replacement._dependency_attempt_boundary is boundary
    replacement.close()
