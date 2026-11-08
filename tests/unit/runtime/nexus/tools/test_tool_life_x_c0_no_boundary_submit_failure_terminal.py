# © Artur Czarnecki. All rights reserved.

"""T-IGX-024-A (C0): a failed ``pool.submit`` in the no-dependency-boundary branch of
``RuntimeToolInvoker._execute_once`` must not leave the durable external-operation record RUNNING.

Roadmap section 3.0.5: "No Tool invocation may remain RUNNING without end". The boundary branch
already terminalises (FAILED) on a submit failure; this module pins the same behaviour for the
no-boundary branch and pins what must NOT change (OD-TLX-C Option A result/ledger semantics,
``bind_future`` residual R-1 not marked FAILED, success path).
"""

from __future__ import annotations

from typing import Any

import pytest
from intergrax.contracts.external_operation_cancellation import ExternalOperationPhysicalState
from intergrax.contracts.idempotency_store import InvocationStatus, InvocationUncertaintyError
from intergrax.runtime.external_operations.external_operation_ownership import (
    ProcessLocalExternalOperationOwner,
)
from intergrax.runtime.external_operations.external_operation_state_store import (
    InMemoryExternalOperationStateStore,
)
from intergrax.runtime.nexus.errors.error_codes import RuntimeErrorCode
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor
from intergrax.runtime.nexus.tools.tool_operation_termination import ToolExecutorTerminationPort
from intergrax.runtime.tools.idempotency_pre_effect_coordinator import (
    IdempotencyPreEffectCoordinator,
)
from intergrax.runtime.tools.in_memory_idempotency_store import InMemoryIdempotencyStore
from intergrax.tools.core.contracts import ToolContract, ToolRetryPolicy
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.registry import ToolRegistry
from pydantic import BaseModel
from testing_support.builder import canonical_governed_execution_scope, canonical_run_id_for_tests

# Import order matters: loading the side-effect-safety helper module first avoids a circular import
# (see C0-TLX reproduction). Only helpers are imported from it; no test function is imported.
from tests.unit.runtime.tools import test_tools_side_effect_safety as _helpers  # noqa: F401
from tests.unit.runtime.tools.test_tools_side_effect_safety import (
    CountingExecutor,
    _register_tool,
    _request,
    _side_effect_safety_governed_scope,
    _state_with_enforce_allow,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_RUN_ID = canonical_run_id_for_tests("tool-life-c0-submit-failure")
_PHYS = ExternalOperationPhysicalState


class _In(BaseModel):
    value: int


class _Out(BaseModel):
    value: int


class _CountingHandler:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, request: ToolExecutionRequest[_In]) -> _Out:
        self.calls += 1
        return _Out(value=request.input.value)


class _InvokeState:
    tenant_id = "tenant-c0-submit-failure"
    run_id = _RUN_ID
    task_id = None
    verified_agent_governance_human_approval = None
    agent_governance_approval_consumption = None
    declarative_hitl_grant = None

    def __init__(self) -> None:
        self.request = type("Req", (), {"metadata": {}})()
        self._context = type(
            "Ctx",
            (),
            {"config": type("Cfg", (), {"policy_bundle": None, "production_mode": False})()},
        )()

    @property
    def context(self):
        return self._context

    def trace_event(self, *args: object, **kwargs: object) -> None:
        return


def _r4_harness() -> tuple[RuntimeToolInvoker, ToolContract, _CountingHandler, InMemoryExternalOperationStateStore]:
    """Direct ``_execute_once`` harness (same shape as the R4 terminal-lifecycle module)."""
    store = InMemoryExternalOperationStateStore()
    registry = ToolRegistry()
    contract = ToolContract(
        tool_id="c0.submit.failure",
        name="c0",
        description="",
        input_schema=_In,
        output_schema=_Out,
        error_mapping={},
        side_effects=True,
    )
    handler = _CountingHandler()
    registry.register(contract, handler)
    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=RegistryToolExecutor(registry),
        external_operation_store=store,
        external_operation_owner=ProcessLocalExternalOperationOwner.mint(),
        dependency_attempt_boundary=None,
    )
    return invoker, contract, handler, store


def _r4_request(step_id: str) -> ToolExecutionRequest[_In]:
    return ToolExecutionRequest(
        run_id=_RUN_ID,
        step_id=step_id,
        tool_id="c0.submit.failure",
        input=_In(value=7),
    )


def _physical_states(store: InMemoryExternalOperationStateStore) -> dict[Any, ExternalOperationPhysicalState]:
    return {key: rec.physical_state for key, rec in store._records.items()}  # noqa: SLF001


def _new_states(
    store: InMemoryExternalOperationStateStore,
    before: dict[Any, ExternalOperationPhysicalState],
) -> dict[Any, ExternalOperationPhysicalState]:
    return {k: s for k, s in _physical_states(store).items() if k not in before}


def _public_path_harness(
    *, side_effects: bool, with_coordinator: bool, max_attempts: int = 1
) -> tuple[RuntimeToolInvoker, CountingExecutor, InMemoryExternalOperationStateStore, InMemoryIdempotencyStore]:
    """Public ``invoke()`` harness (same wiring as the C0-TLX reproduction)."""
    ext_store = InMemoryExternalOperationStateStore()
    ledger = InMemoryIdempotencyStore()
    registry = ToolRegistry()
    _register_tool(
        registry,
        tool_id="tool_a",
        side_effects=side_effects,
        retry_policy=ToolRetryPolicy(max_attempts=max_attempts),
    )
    executor = CountingExecutor()
    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=executor,
        pre_effect_coordinator=(
            IdempotencyPreEffectCoordinator(idempotency_store=ledger) if with_coordinator else None
        ),
        external_operation_store=ext_store,
        external_operation_owner=ProcessLocalExternalOperationOwner.mint(),
        dependency_attempt_boundary=None,
    )
    return invoker, executor, ext_store, ledger


def _second_step_request(key: str):
    import dataclasses

    # Operation id is deterministic per tenant/tool/step; a NEW step id is required so that call 2
    # is not suppressed before submit (a reused step would prove nothing).
    return dataclasses.replace(_request(key=key, value=9), step_id="step2")


# --------------------------------------------------------------------------------------------
# Positive: defect closed
# --------------------------------------------------------------------------------------------


def test_t1_post_close_submit_failure_terminalises_record_failed_store_level() -> None:
    invoker, contract, handler, store = _r4_harness()
    state = _InvokeState()
    with canonical_governed_execution_scope(_RUN_ID):
        out1 = invoker._execute_once(state, "agent", contract, _r4_request("1"))  # noqa: SLF001
        assert out1.value == 7
        call1_states = _physical_states(store)
        assert list(call1_states.values()) == [_PHYS.SUCCEEDED]

        invoker.close()
        with pytest.raises(RuntimeError):
            invoker._execute_once(state, "agent", contract, _r4_request("2"))  # noqa: SLF001

    new = _new_states(store, call1_states)
    assert len(new) == 1
    assert list(new.values()) == [_PHYS.FAILED]
    assert _PHYS.RUNNING not in _physical_states(store).values()
    for key, phys in call1_states.items():
        assert _physical_states(store)[key] is _PHYS.SUCCEEDED
    assert handler.calls == 1


@pytest.mark.parametrize(
    ("side_effects", "with_coordinator"),
    [(True, True), (False, False)],
    ids=["side_effects+idempotency", "no_side_effects"],
)
def test_t2_post_close_public_path_record_failed_and_result_semantics_unchanged(
    side_effects: bool, with_coordinator: bool
) -> None:
    invoker, executor, ext_store, ledger = _public_path_harness(
        side_effects=side_effects, with_coordinator=with_coordinator
    )
    state = _state_with_enforce_allow()
    with _side_effect_safety_governed_scope():
        first = invoker.invoke(state=state, agent_id="agent", request=_request(key="c0-first"))
        assert first.success
        before = _physical_states(ext_store)
        invoker.close()
        req2 = _second_step_request("c0-second")
        second = invoker.invoke(state=state, agent_id="agent", request=req2)

        assert not second.success
        assert executor.calls == 1
        assert second.error is not None
        if side_effects:
            # OD-TLX-C Option A unchanged by the correction.
            assert second.error.error_code == RuntimeErrorCode.ABANDONED
            assert second.error.error_message == "abandoned"
            assert ledger.get_status("tenant_test", "c0-second") == InvocationStatus.UNCERTAIN
            with pytest.raises(InvocationUncertaintyError):
                invoker.invoke(state=state, agent_id="agent", request=req2)
            assert executor.calls == 1
            assert ledger.get_status("tenant_test", "c0-second") == InvocationStatus.UNCERTAIN
        else:
            # Value observed on the UNFIXED tree (red-first step A); not a guess. Message is an
            # implementation detail of ThreadPoolExecutor and is intentionally not asserted.
            assert second.error.error_code == RuntimeErrorCode.TOOL_ERROR

        # Defect closed: the new record is terminal FAILED, never RUNNING.
        new = _new_states(ext_store, before)
        assert len(new) == 1
        assert list(new.values()) == [_PHYS.FAILED]
        assert _PHYS.RUNNING not in _physical_states(ext_store).values()


def test_t3_generic_submit_failure_on_open_invoker_terminalises_record_and_propagates_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invoker, contract, handler, store = _r4_harness()
    boom = RuntimeError("c0-boom")

    def _failing_submit(*args: object, **kwargs: object) -> object:
        raise boom

    monkeypatch.setattr(invoker._execution_pool, "submit", _failing_submit)  # noqa: SLF001
    try:
        with canonical_governed_execution_scope(_RUN_ID):
            with pytest.raises(RuntimeError) as excinfo:
                invoker._execute_once(_InvokeState(), "agent", contract, _r4_request("1"))  # noqa: SLF001
    finally:
        monkeypatch.undo()
        invoker.close()
    assert excinfo.value is boom
    assert list(_physical_states(store).values()) == [_PHYS.FAILED]
    assert handler.calls == 0


def test_t4_retry_attempts_gt_one_submit_failing_post_close_admits_retry_after_failed() -> None:
    # Signed condition (2) support: effective max attempts > 1 (read-only tool, max_attempts=3) with a
    # failing pool.submit. FAILED (not RUNNING) lets attempt > 1 be admitted. Result mapping /
    # ledger semantics are asserted via the same fields as T2; the auditor re-checks at the post-fix SHA.
    invoker, executor, ext_store, _ledger = _public_path_harness(
        side_effects=False, with_coordinator=False, max_attempts=3
    )
    submit_attempts: list[int] = []
    real_submit = invoker._execution_pool.submit  # noqa: SLF001

    def _counting_submit(*args: object, **kwargs: object) -> object:
        submit_attempts.append(1)
        return real_submit(*args, **kwargs)

    state = _state_with_enforce_allow()
    with _side_effect_safety_governed_scope():
        first = invoker.invoke(state=state, agent_id="agent", request=_request(key="c0-first"))
        assert first.success
        before = _physical_states(ext_store)
        invoker.close()
        invoker._execution_pool.submit = _counting_submit  # type: ignore[method-assign]  # noqa: SLF001
        second = invoker.invoke(state=state, agent_id="agent", request=_second_step_request("c0-second"))

    assert not second.success
    assert executor.calls == 1
    assert second.error is not None
    # Final result mapping: same code as the single-attempt non-side-effecting case (T2), observed
    # identical on the unfixed tree. Message text is not pinned.
    assert second.error.error_code == RuntimeErrorCode.TOOL_ERROR
    new = _new_states(ext_store, before)
    assert list(new.values()) == [_PHYS.FAILED]
    assert _PHYS.RUNNING not in _physical_states(ext_store).values()
    # With FAILED (not RUNNING) every attempt up to the effective maximum is admitted again.
    assert len(submit_attempts) == 3


# --------------------------------------------------------------------------------------------
# Negative: what must NOT change
# --------------------------------------------------------------------------------------------


class _BindFailingPort(ToolExecutorTerminationPort):
    def bind_future(self, operation_id: str, future: object) -> None:  # type: ignore[override]
        raise ValueError("c0-bind-boom")


def test_n1_bind_future_failure_after_real_submit_is_not_marked_failed() -> None:
    invoker, contract, handler, store = _r4_harness()
    invoker._tool_termination_port = _BindFailingPort()  # noqa: SLF001
    try:
        with canonical_governed_execution_scope(_RUN_ID):
            with pytest.raises(ValueError, match="c0-bind-boom"):
                invoker._execute_once(_InvokeState(), "agent", contract, _r4_request("1"))  # noqa: SLF001
    finally:
        # Join the (real) worker started by the successful submit.
        invoker.close()
    states = list(_physical_states(store).values())
    assert len(states) == 1
    # Asserts only "not FAILED": the bind_future residual R-1 (record left non-terminal) is
    # intentionally NOT pinned here.
    assert states[0] is not _PHYS.FAILED
