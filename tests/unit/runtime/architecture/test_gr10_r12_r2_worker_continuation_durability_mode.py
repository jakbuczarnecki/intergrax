# © Artur Czarnecki. All rights reserved.

"""GR-10-R12-R2 — worker production_mode propagation and continuation durability."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from intergrax.contracts.execution_continuation import (
    ExecutionContinuationError,
    ExecutionContinuationErrorCode,
)
from intergrax.runtime.execution.continuation.persistence import (
    ExecutionContinuationDurableBacking,
    InMemoryExecutionContinuationStateStore,
    execution_continuation_state_store_from_durable_export,
    export_durable_continuation_state,
)
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.nexus_worker_execution import NexusWorkerRuntime
from intergrax.runtime.task.worker_bootstrap import (
    build_nexus_task_execution_registry,
    create_nexus_celery_worker_app,
)
from testing_support.admitted_root_governance_identity import (
    lab_admitted_root_governance_identity_for_task,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[4]
_WORKER_EXECUTION = _REPO / "intergrax/runtime/task/nexus_worker_execution.py"
_WORKER_BOOTSTRAP = _REPO / "intergrax/runtime/task/worker_bootstrap.py"
_QUEUE_WIRING = _REPO / "intergrax/applications/_shared/queue_worker_wiring.py"


def _durable_store():
    return execution_continuation_state_store_from_durable_export(
        export_durable_continuation_state(ExecutionContinuationDurableBacking()),
    )


def _empty_registry() -> AgentRegistry:
    return AgentRegistry()


def test_gr10_r12_r2_from_registry_requires_explicit_production_mode() -> None:
    signature = inspect.signature(NexusWorkerRuntime.from_registry)
    assert signature.parameters["production_mode"].default is inspect.Parameter.empty


def test_gr10_r12_r2_worker_bootstrap_forwards_production_mode_to_nexus_loop() -> None:
    source = _WORKER_EXECUTION.read_text(encoding="utf-8")
    assert "production_mode=production_mode" in source
    tree = ast.parse(source, filename=str(_WORKER_EXECUTION))
    nexus_loop_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "NexusLoop"
    ]
    assert nexus_loop_calls
    for call in nexus_loop_calls:
        kw_names = {kw.arg for kw in call.keywords if kw.arg}
        assert "production_mode" in kw_names


def test_gr10_r12_r2_worker_bootstrap_no_implicit_continuation_wire() -> None:
    bootstrap = _WORKER_BOOTSTRAP.read_text(encoding="utf-8")
    assert "wire_execution_continuation_state_store" not in bootstrap


def test_gr10_r12_r2_production_worker_no_store_blocks() -> None:
    with pytest.raises(ExecutionContinuationError) as exc:
        NexusWorkerRuntime.from_registry(
            _empty_registry(),
            production_mode=True,
            admit_root_governance_identity=lab_admitted_root_governance_identity_for_task,
            execution_continuation_state_store=None,
        )
    assert exc.value.code is ExecutionContinuationErrorCode.NON_DURABLE_CONTINUATION_STORE


def test_gr10_r12_r2_production_worker_in_memory_blocks() -> None:
    store = InMemoryExecutionContinuationStateStore()
    with pytest.raises(ExecutionContinuationError) as exc:
        NexusWorkerRuntime.from_registry(
            _empty_registry(),
            production_mode=True,
            admit_root_governance_identity=lab_admitted_root_governance_identity_for_task,
            execution_continuation_state_store=store,
        )
    assert exc.value.code is ExecutionContinuationErrorCode.NON_DURABLE_CONTINUATION_STORE


class _CustomNonDurableStore(InMemoryExecutionContinuationStateStore):
    """Plugin-shaped store with is_durable=False."""


def test_gr10_r12_r2_production_worker_custom_non_durable_blocks() -> None:
    with pytest.raises(ExecutionContinuationError) as exc:
        NexusWorkerRuntime.from_registry(
            _empty_registry(),
            production_mode=True,
            admit_root_governance_identity=lab_admitted_root_governance_identity_for_task,
            execution_continuation_state_store=_CustomNonDurableStore(),
        )
    assert exc.value.code is ExecutionContinuationErrorCode.NON_DURABLE_CONTINUATION_STORE


def test_gr10_r12_r2_production_worker_durable_passes() -> None:
    store = _durable_store()
    runtime = NexusWorkerRuntime.from_registry(
        _empty_registry(),
        production_mode=True,
        admit_root_governance_identity=lab_admitted_root_governance_identity_for_task,
        execution_continuation_state_store=store,
    )
    assert runtime.host_execution is not None


def test_gr10_r12_r2_lab_worker_implicit_in_memory_passes() -> None:
    runtime = NexusWorkerRuntime.from_registry(
        _empty_registry(),
        production_mode=False,
        admit_root_governance_identity=lab_admitted_root_governance_identity_for_task,
        execution_continuation_state_store=None,
    )
    assert runtime.host_execution is not None


def test_gr10_r12_r2_build_registry_production_without_store_blocks() -> None:
    with pytest.raises(ExecutionContinuationError) as exc:
        build_nexus_task_execution_registry(
            _empty_registry(),
            production_mode=True,
            admit_root_governance_identity=lab_admitted_root_governance_identity_for_task,
        )
    assert exc.value.code is ExecutionContinuationErrorCode.NON_DURABLE_CONTINUATION_STORE


def test_gr10_r12_r2_queue_wiring_passes_production_mode_kwarg() -> None:
    source = _QUEUE_WIRING.read_text(encoding="utf-8")
    assert "production_mode: bool" in source
    assert "production_mode=production_mode" in source


def test_gr10_r12_r2_celery_root_requires_production_mode() -> None:
    signature = inspect.signature(create_nexus_celery_worker_app)
    assert signature.parameters["production_mode"].default is inspect.Parameter.empty
