# © Artur Czarnecki. All rights reserved.

"""Scoped Ask service, HTTP, and propagation tests."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.task.task import Task, TaskResult, TaskState
from local_workspace_application.host.factory import create_local_workspace_backend_app
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.serving.workspace_schemas import WorkspaceSearchHitV1
from local_workspace_application.workspaces.ask_models import AskRunStatus
from local_workspace_application.workspaces.ask_service import WorkspaceAskService
from local_workspace_application.workspaces.knowledge_ask_scope_models import KnowledgeAskScopeV1
from local_workspace_application.workspaces.knowledge_ask_scope_resolver import (
    KnowledgeAskScopeResolver,
)
from local_workspace_application.workspaces.models import WorkspaceDocumentReference
from local_workspace_application.workspaces.search_evidence import SearchEvidenceIncompleteError

pytestmark = pytest.mark.unit

_PREFIX = "/v1/local_workspace"


class RecordingFakeLLM(LLMAdapter):
    provider = "fake"
    model = "fake"

    def __init__(self, *, fixed_text: str) -> None:
        super().__init__()
        self._fixed_text = fixed_text

    @property
    def context_window_tokens(self) -> int:
        return 128_000

    def generate_messages(self, messages, *, temperature=None, max_tokens=None, run_id=None):
        _ = messages, temperature, max_tokens, run_id
        return build_adapter_response(content=self._fixed_text)


def _search_task_result(*, workspace_id: str, evidence: list[dict[str, Any]]) -> TaskResult:
    return TaskResult(
        task_id="search-1",
        run_id="search-1",
        state=TaskState.COMPLETED,
        answer="ok",
        execution_result=AgentExecutionResult(
            agent_id="local_search",
            run_id="search-1",
            status=AgentExecutionStatus.COMPLETED,
            summary="ok",
            structured_data={
                "search_summary": {
                    "query": "q",
                    "workspace_id": workspace_id,
                    "evidence": evidence,
                }
            },
        ),
    )


@pytest.fixture
def scoped_ask_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    store = InMemoryDocumentStore()
    workspace_root = tmp_path / "user_docs"
    workspace_root.mkdir()
    data_home = tmp_path / "lkw-data"
    sqlite_dir = tmp_path / "sqlite"
    shadow_dir = tmp_path / "shadow"
    for path in (data_home, sqlite_dir, shadow_dir):
        path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("LOCAL_WORKSPACE_VECTOR_STORE", "inmemory")
    monkeypatch.setenv("INTERGRAX_ALLOWED_READ_ROOTS", str(workspace_root.resolve()))
    monkeypatch.setenv("LOCAL_WORKSPACE_ENABLE_RAG", "true")
    monkeypatch.setenv("LOCAL_WORKSPACE_ENABLE_RAG_INGEST", "true")
    monkeypatch.setenv("LOCAL_WORKSPACE_INCLUDE_MCP", "false")
    monkeypatch.setenv("DATA_HOME", str(data_home))
    monkeypatch.setenv("LKW_DATA_HOME", str(data_home))
    monkeypatch.setenv("INTERGRAX_SQLITE_DATA_DIR", str(sqlite_dir))
    monkeypatch.setenv("INTERGRAX_SHADOW_ROOT", str(shadow_dir))
    monkeypatch.delenv("INTERGRAX_MONGODB_URI", raising=False)
    monkeypatch.setattr(
        "local_workspace_application.serving.workspace_routes.resolve_managed_workspace_document_store",
        lambda document_store=None: store,
    )

    llm = RecordingFakeLLM(
        fixed_text=json.dumps(
            {
                "status": "completed",
                "answer": "Scoped answer.",
                "used_evidence_ids": ["E1"],
            }
        )
    )
    settings = LocalWorkspaceBackendSettings.from_env()
    app = create_local_workspace_backend_app(settings=settings)
    app.state.lkw_ask_service.llm_adapter = llm

    with TestClient(app) as client:
        yield {"client": client, "app": app, "llm": llm}


def _headers(tenant_id: str) -> dict[str, str]:
    return {"X-Tenant-Id": tenant_id}


def _create_workspace(client: TestClient, tenant: str) -> str:
    created = client.post(
        f"{_PREFIX}/workspaces",
        headers=_headers(tenant),
        json={"name": "Scoped Ask"},
    )
    assert created.status_code == 201, created.text
    return created.json()["workspace_id"]


def test_no_scope_leaves_search_metadata_without_allowed_source_ids(scoped_ask_api) -> None:
    app = scoped_ask_api["app"]
    client = scoped_ask_api["client"]
    tenant = f"tenant-{uuid.uuid4().hex[:8]}"
    workspace_id = _create_workspace(client, tenant)
    captured: dict[str, Any] = {}

    async def _execute(task: Task) -> TaskResult:
        captured.update(dict(task.metadata or {}))
        return _search_task_result(workspace_id=workspace_id, evidence=[])

    app.state.lkw_ask_service._executor.execute = _execute
    response = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/ask",
        headers=_headers(tenant),
        json={"question": "What is the policy?"},
    )
    assert response.status_code == 200, response.text
    assert "allowed_source_ids" not in captured


def test_scoped_ask_passes_allowed_source_ids_to_search_task(scoped_ask_api, monkeypatch) -> None:
    app = scoped_ask_api["app"]
    client = scoped_ask_api["client"]
    tenant = f"tenant-{uuid.uuid4().hex[:8]}"
    workspace_id = _create_workspace(client, tenant)
    item_id = "indexed:binding-a"
    captured: dict[str, Any] = {}

    class _FakeResolver:
        def resolve(self, *, tenant_id: str, workspace_id: str, scope: KnowledgeAskScopeV1):
            _ = tenant_id, workspace_id, scope
            from local_workspace_application.workspaces.knowledge_ask_scope_models import (
                KnowledgeRetrievalScopeV1,
            )

            return KnowledgeRetrievalScopeV1.from_validated_source_ids(("source-b",))

    app.state.lkw_ask_service._scope_resolver = _FakeResolver()

    async def _execute(task: Task) -> TaskResult:
        captured.update(dict(task.metadata or {}))
        return _search_task_result(workspace_id=workspace_id, evidence=[])

    app.state.lkw_ask_service._executor.execute = _execute
    response = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/ask",
        headers=_headers(tenant),
        json={"question": "What is onboarding?", "knowledge_item_ids": [item_id]},
    )
    assert response.status_code == 200, response.text
    assert captured.get("allowed_source_ids") == ["source-b"]


def test_failed_scope_does_not_invoke_retrieval(scoped_ask_api) -> None:
    app = scoped_ask_api["app"]
    client = scoped_ask_api["client"]
    tenant = f"tenant-{uuid.uuid4().hex[:8]}"
    workspace_id = _create_workspace(client, tenant)

    class _FailingResolver:
        def resolve(self, *, tenant_id: str, workspace_id: str, scope: KnowledgeAskScopeV1):
            _ = tenant_id, workspace_id, scope
            from local_workspace_application.workspaces.knowledge_ask_scope_models import (
                KnowledgeAskScopeError,
            )

            raise KnowledgeAskScopeError("knowledge_ask_scope_not_found")

    app.state.lkw_ask_service._scope_resolver = _FailingResolver()
    app.state.lkw_ask_service._executor.execute = AsyncMock()

    response = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/ask",
        headers=_headers(tenant),
        json={"question": "What?", "knowledge_item_ids": ["indexed:missing"]},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "failed"
    assert body["error"]["code"] == "knowledge_ask_scope_not_found"
    app.state.lkw_ask_service._executor.execute.assert_not_called()


def test_out_of_scope_evidence_fails_integrity(scoped_ask_api, monkeypatch) -> None:
    app = scoped_ask_api["app"]
    client = scoped_ask_api["client"]
    tenant = f"tenant-{uuid.uuid4().hex[:8]}"
    workspace_id = _create_workspace(client, tenant)

    class _FakeResolver:
        def resolve(self, *, tenant_id: str, workspace_id: str, scope: KnowledgeAskScopeV1):
            _ = tenant_id, workspace_id, scope
            from local_workspace_application.workspaces.knowledge_ask_scope_models import (
                KnowledgeRetrievalScopeV1,
            )

            return KnowledgeRetrievalScopeV1.from_validated_source_ids(("source-b",))

    app.state.lkw_ask_service._scope_resolver = _FakeResolver()

    async def _execute(task: Task) -> TaskResult:
        _ = task
        return _search_task_result(
            workspace_id=workspace_id,
            evidence=[
                {
                    "document_id": "doc-1",
                    "source_id": "source-a",
                    "workspace_id": workspace_id,
                    "source_path": "/tmp/a.txt",
                    "file_name": "a.txt",
                    "score": 0.9,
                    "snippet": "wrong source",
                }
            ],
        )

    app.state.lkw_ask_service._executor.execute = _execute

    def _fake_map_search_hits(**kwargs: object) -> list[WorkspaceSearchHitV1]:
        _ = kwargs
        return [
            WorkspaceSearchHitV1(
                document_id="doc-1",
                source_id="source-a",
                workspace_id=workspace_id,
                source_path="/tmp/a.txt",
                file_name="a.txt",
                score=0.9,
                snippet="wrong source",
            )
        ]

    monkeypatch.setattr(
        "local_workspace_application.workspaces.ask_service.map_search_hits",
        _fake_map_search_hits,
    )

    response = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/ask",
        headers=_headers(tenant),
        json={"question": "What?", "knowledge_item_ids": ["indexed:binding-a"]},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "failed"
    assert body["error"]["code"] == "knowledge_ask_scope_integrity_failed"


def test_http_rejects_empty_knowledge_item_ids(scoped_ask_api) -> None:
    client = scoped_ask_api["client"]
    tenant = f"tenant-{uuid.uuid4().hex[:8]}"
    workspace_id = _create_workspace(client, tenant)
    response = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/ask",
        headers=_headers(tenant),
        json={"question": "What?", "knowledge_item_ids": []},
    )
    assert response.status_code == 422


def test_validate_scoped_evidence_unit() -> None:
    evidence = [
        WorkspaceSearchHitV1(
            document_id="doc-1",
            source_id="source-a",
            workspace_id="ws-1",
            source_path="/a.txt",
            file_name="a.txt",
            score=0.9,
            snippet="text",
        )
    ]
    with pytest.raises(Exception) as exc:
        WorkspaceAskService._validate_scoped_evidence(
            evidence,
            allowed_source_ids=("source-b",),
        )
    assert exc.value.error_code == "knowledge_ask_scope_integrity_failed"
