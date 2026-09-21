# © Artur Czarnecki. All rights reserved.

"""HARNESS-01-R5-W2-R6-R3 — transactional RuntimeToolInvoker ownership transfer."""

from __future__ import annotations

import pytest

from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.runtime_tool_invoker_composition import (
    ProductionRuntimeToolInvokerCompositionError,
)
from intergrax.runtime.resilience.dependency_attempt_execution_boundary import (
    DependencyAttemptExecutionBoundaryClosedError,
)
from intergrax.runtime.tools.in_memory_idempotency_store import InMemoryIdempotencyStore
from tests.qualification.harness_01.test_harness_01_w2_r6_r2_runtime_tool_invoker_reconfiguration_lifecycle import (
    _admission_request,
    _boundary_acquire_and_release,
    _invoker_with_boundary,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _forced_reconfiguration_failure(
    invoker: RuntimeToolInvoker,
    *,
    idempotency_store: InMemoryIdempotencyStore | None = None,
) -> None:
    store = idempotency_store or InMemoryIdempotencyStore()
    with pytest.raises(ProductionRuntimeToolInvokerCompositionError):
        invoker.with_idempotency_store(store, production_mode=True)


def test_harness_01_w2_r6_r3_failure_restores_boundary_ownership() -> None:
    invoker, boundary = _invoker_with_boundary()
    original = invoker._dependency_attempt_boundary
    _forced_reconfiguration_failure(invoker)
    assert invoker._dependency_attempt_boundary is original is boundary


def test_harness_01_w2_r6_r3_failure_keeps_pool_open() -> None:
    invoker, _boundary = _invoker_with_boundary()
    _forced_reconfiguration_failure(invoker)
    assert invoker._execution_pool_closed is False


def test_harness_01_w2_r6_r3_boundary_operational_after_failure() -> None:
    invoker, boundary = _invoker_with_boundary()
    _forced_reconfiguration_failure(invoker)
    _boundary_acquire_and_release(boundary)


def test_harness_01_w2_r6_r3_source_close_after_failure() -> None:
    invoker, boundary = _invoker_with_boundary()
    _forced_reconfiguration_failure(invoker)
    invoker.close()
    with pytest.raises(DependencyAttemptExecutionBoundaryClosedError):
        boundary.acquire(_admission_request())


def test_harness_01_w2_r6_r3_repeat_failure_then_close() -> None:
    invoker, boundary = _invoker_with_boundary()
    _forced_reconfiguration_failure(invoker)
    _forced_reconfiguration_failure(invoker)
    assert invoker._dependency_attempt_boundary is boundary
    assert invoker._execution_pool_closed is False
    invoker.close()
    with pytest.raises(DependencyAttemptExecutionBoundaryClosedError):
        boundary.acquire(_admission_request())


def test_harness_01_w2_r6_r3_success_after_failure() -> None:
    invoker, boundary = _invoker_with_boundary()
    _forced_reconfiguration_failure(invoker)
    assert invoker._dependency_attempt_boundary is boundary
    replacement = invoker.with_idempotency_store(InMemoryIdempotencyStore())
    assert replacement._dependency_attempt_boundary is boundary
    assert invoker._execution_pool_closed is True
    assert invoker._dependency_attempt_boundary is None
    _boundary_acquire_and_release(boundary)
    replacement.close()
    with pytest.raises(DependencyAttemptExecutionBoundaryClosedError):
        boundary.acquire(_admission_request())
