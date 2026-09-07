# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from intergrax.runtime.interactions.intake_service import InteractionIntakeService
from intergrax.runtime.task.task import Task, TaskContext, TaskResult, TaskState


class _RecordingExecutor:
    def __init__(self) -> None:
        self.prepare_calls = 0
        self.execute_calls = 0
        self.last_task: Task | None = None

    def prepare(self, task: Task) -> Task:
        self.prepare_calls += 1
        prepared = task.model_copy(
            update={"context": task.context.model_copy(update={"capability": "prepared.cap"})}
        )
        self.last_task = prepared
        return prepared

    async def execute(self, task: Task) -> TaskResult:
        self.execute_calls += 1
        prepared = self.prepare(task)
        self.last_task = prepared
        return TaskResult(task_id=prepared.task_id, state=TaskState.COMPLETED, answer="ok")


class _RejectingVerifier:
    def __init__(self) -> None:
        self.verify_calls = 0

    def verify(self, *, headers, body) -> None:
        self.verify_calls += 1
        raise ValueError("verification failed")


class _CountingExecutor:
    def __init__(self) -> None:
        self.execute_calls = 0

    async def execute(self, task: Task) -> TaskResult:
        self.execute_calls += 1
        return TaskResult(task_id=task.task_id, state=TaskState.COMPLETED, answer="ok")


def test_interaction_intake_service_signature_has_no_nexus_loop() -> None:
    signature = inspect.signature(InteractionIntakeService.__init__)
    assert "nexus_loop" not in signature.parameters


def test_interaction_intake_service_does_not_store_nexus_loop() -> None:
    service = InteractionIntakeService()
    assert not hasattr(service, "_nexus_loop")


def test_interaction_intake_service_source_has_no_nexus_tokens() -> None:
    source_path = Path(inspect.getfile(InteractionIntakeService))
    source = source_path.read_text(encoding="utf-8")
    assert "NexusLoop" not in source
    assert "NexusLoopTaskExecutor" not in source
    assert "hasattr(" not in source
    assert "# type: ignore[attr-defined]" not in source


def test_interaction_intake_service_source_does_not_mint_execution_identity() -> None:
    source = Path(inspect.getfile(InteractionIntakeService)).read_text(encoding="utf-8")
    assert "mint_run_id" not in source
    assert "mint_attempt_id" not in source
    assert "mint_execution_id" not in source


def test_interaction_intake_service_source_has_no_strategy_selection() -> None:
    source = Path(inspect.getfile(InteractionIntakeService)).read_text(encoding="utf-8")
    assert "StrategyExecutionRouter" not in source
    assert "INFERENCE" not in source
    assert "AGENTIC" not in source
    assert "ORCHESTRATION" not in source


@pytest.mark.asyncio
async def test_interaction_intake_uses_task_executor_when_execute_true() -> None:
    executor = _RecordingExecutor()
    service = InteractionIntakeService(task_executor=executor)
    intake = await service.intake_payload(
        {"message": "hello", "capability": "echo.basic", "user_id": "u1"},
        tenant_id="t1",
        execute=True,
    )
    assert intake.executed is True
    assert intake.result is not None
    assert intake.result.answer == "ok"
    assert executor.prepare_calls == 1
    assert executor.execute_calls == 1


@pytest.mark.asyncio
async def test_interaction_intake_does_not_execute_when_execute_false() -> None:
    executor = _RecordingExecutor()
    service = InteractionIntakeService(task_executor=executor)
    intake = await service.intake_payload(
        {"message": "hello", "capability": "echo.basic", "user_id": "u1"},
        tenant_id="t1",
        execute=False,
    )
    assert intake.executed is False
    assert intake.result is None
    assert executor.execute_calls == 0
    assert executor.prepare_calls == 1


@pytest.mark.asyncio
async def test_interaction_intake_execute_false_enriches_once_without_executor() -> None:
    enricher_calls = 0

    def enricher(task: Task) -> Task:
        nonlocal enricher_calls
        enricher_calls += 1
        return task.model_copy(
            update={"context": task.context.model_copy(update={"capability": "enriched.cap"})}
        )

    service = InteractionIntakeService(task_enricher=enricher)
    intake = await service.intake_payload(
        {"message": "hello", "capability": "echo.basic", "user_id": "u1"},
        tenant_id="t1",
        execute=False,
    )
    assert intake.executed is False
    assert enricher_calls == 1
    assert intake.task.context.capability == "enriched.cap"


@pytest.mark.asyncio
async def test_interaction_intake_requires_executor_for_execute_true() -> None:
    service = InteractionIntakeService()
    with pytest.raises(ValueError, match="Task executor is not configured"):
        await service.intake_payload(
            {"message": "hello", "user_id": "u1"},
            tenant_id="t1",
            execute=True,
        )


@pytest.mark.asyncio
async def test_interaction_intake_verifier_runs_before_executor() -> None:
    verifier = _RejectingVerifier()
    executor = _CountingExecutor()
    service = InteractionIntakeService(task_executor=executor, verifier=verifier)
    with pytest.raises(ValueError, match="verification failed"):
        await service.intake_http(
            headers={},
            body=b'{"message":"hello","user_id":"u1"}',
            content_type="application/json",
            tenant_id="t1",
            execute=True,
        )
    assert verifier.verify_calls == 1
    assert executor.execute_calls == 0
