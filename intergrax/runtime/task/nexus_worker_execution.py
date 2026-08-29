# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Nexus Task v2 worker execution core (§41, J.3)."""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from intergrax.contracts.execution_identity import AttemptId
from intergrax.runtime.background_execution.bootstrap import BackgroundExecutionIdentity
from intergrax.runtime.background_execution.identity_admission import (
    assert_handler_run_id_matches_identity,
    assert_payload_run_id_consistent,
)
from intergrax.runtime.long_running.persistence_contract import TaskCheckpointPersistence
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.execution.budget.ledger import ExecutionBudgetLedgerFactory
from intergrax.runtime.execution.budget.persistence import (
    RunBudgetPersistence,
    create_durable_run_budget_ledger_factory,
)
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task_run_bridge import (
    task_from_execution_request,
    task_result_to_payload,
)
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner
from intergrax.runtime.task.worker_payload import decode_execution_request
from intergrax.tools.execution_models import ToolExecutionError, ToolExecutionResult


def _run_coro_sync(coro):
    """Run async Nexus work from Celery's synchronous worker handler."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coro).result()


class NexusTaskWorkerOutput(BaseModel):
    """Worker handler output returned through Tier-0 queue plane."""

    result_payload: Dict[str, Any] = Field(default_factory=dict)
    schema_version: str = "nexus_task_worker.v1"


@runtime_checkable
class WorkerRunLifecycle(Protocol):
    """Optional run lifecycle hooks invoked inside the worker process."""

    def mark_running(self, run_id: str) -> None: ...

    def mark_completed(self, run_id: str, result_payload: Optional[dict] = None) -> None: ...

    def mark_failed(self, run_id: str, error_type: str, error_message: str) -> None: ...


class NexusWorkerRuntime:
    """Composition root for worker-side Nexus execution."""

    def __init__(
        self,
        task_runner: UnifiedTaskRunner,
        *,
        lifecycle: Optional[WorkerRunLifecycle] = None,
    ) -> None:
        self._task_runner = task_runner
        self._lifecycle = lifecycle

    @classmethod
    def from_registry(
        cls,
        registry: AgentRegistry,
        *,
        checkpoint_store: Optional[TaskCheckpointPersistence] = None,
        lifecycle: Optional[WorkerRunLifecycle] = None,
        run_budget: RunBudget | None = None,
        run_budget_persistence: RunBudgetPersistence | None = None,
        execution_budget_ledger_factory: ExecutionBudgetLedgerFactory | None = None,
    ) -> NexusWorkerRuntime:
        resolved_factory = execution_budget_ledger_factory
        if resolved_factory is None and run_budget_persistence is not None:
            resolved_factory = create_durable_run_budget_ledger_factory(
                run_budget_persistence,
                run_budget,
            )
        loop = NexusLoop(
            registry,
            checkpoint_store=checkpoint_store,
            run_budget=run_budget,
            execution_budget_ledger_factory=resolved_factory,
        )
        return cls(UnifiedTaskRunner(loop), lifecycle=lifecycle)

    @property
    def task_runner(self) -> UnifiedTaskRunner:
        return self._task_runner

    def execute_payload(
        self,
        payload: bytes,
        *,
        tenant_id: str,
        run_id: str,
        execution_identity: BackgroundExecutionIdentity,
    ) -> Dict[str, Any]:
        request = decode_execution_request(payload)
        if request.tenant_id != execution_identity.tenant_id:
            raise ValueError(
                "tenant mismatch between payload and background execution identity: "
                f"payload={request.tenant_id!r} identity={execution_identity.tenant_id!r}"
            )
        if tenant_id != execution_identity.tenant_id:
            raise ValueError(
                "tenant mismatch between worker scope and background execution identity: "
                f"worker={tenant_id!r} identity={execution_identity.tenant_id!r}"
            )
        assert_handler_run_id_matches_identity(
            handler_run_id=run_id,
            execution_identity=execution_identity,
        )
        assert_payload_run_id_consistent(
            payload_run_id=request.run_id,
            execution_identity=execution_identity,
        )
        resolved_run_id = execution_identity.run_id
        resolved_attempt_id: AttemptId = execution_identity.attempt_id

        if self._lifecycle is not None:
            self._lifecycle.mark_running(str(resolved_run_id))

        try:
            task = task_from_execution_request(
                request,
                execution_identity=execution_identity,
            )
            result = _run_coro_sync(
                self._task_runner.run_task(
                    task,
                    run_id=resolved_run_id,
                    attempt_id=resolved_attempt_id,
                )
            )
            result_payload = task_result_to_payload(result)
            if self._lifecycle is not None:
                self._lifecycle.mark_completed(str(resolved_run_id), result_payload=result_payload)
            return result_payload
        except Exception as exc:
            if self._lifecycle is not None:
                self._lifecycle.mark_failed(
                    str(resolved_run_id),
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            raise


def make_nexus_task_worker_handler(
    runtime: NexusWorkerRuntime,
):
    """Build a TaskExecutionRegistry handler for ``nexus.task.v2``."""

    def handler(
        *,
        tenant_id: str,
        run_id: str,
        payload: bytes,
        idempotency_key: Optional[str] = None,
        execution_identity: BackgroundExecutionIdentity,
    ) -> ToolExecutionResult[NexusTaskWorkerOutput]:
        _ = idempotency_key
        try:
            result_payload = runtime.execute_payload(
                payload,
                tenant_id=tenant_id,
                run_id=run_id,
                execution_identity=execution_identity,
            )
            return ToolExecutionResult.ok(
                NexusTaskWorkerOutput(result_payload=result_payload)
            )
        except Exception as exc:
            return ToolExecutionResult(
                success=False,
                output=None,
                error=ToolExecutionError(
                    error_code=type(exc).__name__,
                    error_message=str(exc),
                ),
            )

    return handler


def register_nexus_task_worker(
    registry,
    runtime: NexusWorkerRuntime,
    *,
    logical_task_name: str = "nexus.task.v2",
) -> None:
    from intergrax.runtime.task.worker_payload import NEXUS_TASK_V2_LOGICAL_NAME

    name = logical_task_name or NEXUS_TASK_V2_LOGICAL_NAME
    registry.register(name, make_nexus_task_worker_handler(runtime))
