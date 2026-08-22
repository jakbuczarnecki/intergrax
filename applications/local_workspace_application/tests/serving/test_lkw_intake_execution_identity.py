# © Artur Czarnecki. All rights reserved.

"""Regression tests for canonical TaskId minting at LKW HTTP intake boundaries."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.contracts.execution_identity import (
    mint_run_id,
    mint_task_id,
    validate_run_id,
    validate_task_id,
)
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.task.task import Task, TaskResult, TaskState
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.host.task_executor import LocalWorkspaceTaskExecutor
from local_workspace_application.serving.fastapi_router import LocalWorkspaceRunService
from local_workspace_application.serving.schemas import LocalWorkspaceRunRequestV1
from local_workspace_application.serving.workspace_routes import mount_managed_workspace_routes
from local_workspace_application.workspaces.models import WorkspaceDocumentReference
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository
from local_workspace_application.workspaces.sync_runtime import build_managed_workspace_sync_runtime
from local_workspace_application.workspaces.sync_service import ManagedWorkspaceSyncService

_PREFIX = "/v1/local_workspace"
_TENANT = "tenant-intake"


def _search_task_result(
    *,
    workspace_id: str,
    source_id: str,
    document: WorkspaceDocumentReference,
) -> TaskResult:
    run_id = mint_run_id()
    task_id = mint_task_id()
    return TaskResult(
        task_id=task_id,
        run_id=run_id,
        state=TaskState.COMPLETED,
        answer="ok",
        agent_id="local_search",
        execution_result=AgentExecutionResult(
            agent_id="local_search",
            run_id=run_id,
            status=AgentExecutionStatus.COMPLETED,
            summary="ok",
            structured_data={
                "search_summary": {
                    "query": "payment",
                    "workspace_id": workspace_id,
                    "result_count": 1,
                    "evidence": [
                        {
                            "document_id": document.document_id,
                            "source_id": source_id,
                            "workspace_id": workspace_id,
                            "source_path": document.source_path,
                            "file_name": document.file_name,
                            "score": 0.91,
                            "snippet": "platform snippet",
                            "metadata": {},
                        }
                    ],
                }
            },
        ),
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_task_intake_mints_canonical_task_id_and_reaches_executor() -> None:
    result_run_id = mint_run_id()
    result_task_id = mint_task_id()
    task_result = TaskResult(
        task_id=result_task_id,
        run_id=result_run_id,
        state=TaskState.COMPLETED,
        answer="answer",
        agent_id="local_search",
        metadata={},
    )
    executor = AsyncMock(spec=LocalWorkspaceTaskExecutor)
    executor.execute = AsyncMock(return_value=task_result)
    executor.nexus_loop = None
    service = LocalWorkspaceRunService(task_executor=executor, default_agent_id="local_search")

    response = await service.run_task(
        LocalWorkspaceRunRequestV1(
            message="find docs",
            capability="local.workspace.search",
        )
    )

    executor.execute.assert_awaited_once()
    captured_task: Task = executor.execute.await_args.args[0]
    assert str(captured_task.task_id).startswith("task_")
    assert not str(captured_task.task_id).startswith("run_")
    validate_task_id(captured_task.task_id)
    validate_run_id(response.run_id)
    assert response.run_id == result_run_id


@pytest.mark.unit
def test_search_workspace_intake_mints_canonical_task_id_and_reaches_executor(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    data_home = tmp_path / "data"
    data_home.mkdir()
    monkeypatch.setenv("DATA_HOME", str(data_home))
    monkeypatch.setenv("INTERGRAX_ALLOWED_READ_ROOTS", str(tmp_path / "docs"))
    (tmp_path / "docs").mkdir()
    settings = LocalWorkspaceBackendSettings.from_env()
    settings = replace(settings, data_home=str(data_home))

    executor = AsyncMock(spec=LocalWorkspaceTaskExecutor)
    sync = ManagedWorkspaceSyncService(repo, executor)  # type: ignore[arg-type]
    runtime = build_managed_workspace_sync_runtime(
        document_store=store,
        sync_service=sync,
        repository=repo,
    )
    app = FastAPI()
    service = mount_managed_workspace_routes(
        app,
        task_executor=executor,  # type: ignore[arg-type]
        settings=settings,
        repository=repo,
        sync_runtime=runtime,
    )
    workspace = service.create_workspace(tenant_id=_TENANT, name="Docs")
    workspace_id = workspace.workspace_id
    source = service.register_local_folder_source(
        tenant_id=_TENANT,
        workspace_id=workspace_id,
        path=str(tmp_path / "docs"),
    )
    document = WorkspaceDocumentReference(
        document_id="doc-intake-1",
        tenant_id=_TENANT,
        workspace_id=workspace_id,
        source_id=source.source_id,
        source_path=str((tmp_path / "docs" / "a.txt").resolve()),
        file_name="a.txt",
        content_hash="sha256:abc",
        indexed_at=datetime.now(UTC),
    )
    repo.put_document_ref(document)

    async def _execute(task: Task) -> TaskResult:
        assert str(task.task_id).startswith("task_")
        assert not str(task.task_id).startswith("run_")
        validate_task_id(task.task_id)
        return _search_task_result(
            workspace_id=workspace_id,
            source_id=source.source_id,
            document=document,
        )

    executor.execute = AsyncMock(side_effect=_execute)

    with TestClient(app) as client:
        response = client.post(
            f"{_PREFIX}/workspaces/{workspace_id}/search",
            headers={"X-Tenant-Id": _TENANT},
            json={"query": "payment", "limit": 5},
        )

    assert response.status_code == 200, response.text
    executor.execute.assert_awaited_once()
