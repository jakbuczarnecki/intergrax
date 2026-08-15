# © Artur Czarnecki. All rights reserved.

"""API / contract tests for Trusted Ask Workspace (MVP-2)."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional, Sequence
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.runtime.task.task import TaskResult, TaskState
from local_workspace_application.host.factory import create_local_workspace_backend_app
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.workspaces.ask_models import AskRunStatus
from local_workspace_application.workspaces.ask_repository import WorkspaceAskRepository
from local_workspace_application.workspaces.models import WorkspaceDocumentReference

pytestmark = pytest.mark.unit

_PREFIX = "/v1/local_workspace"


class RecordingFakeLLM(LLMAdapter):
    provider = "fake"
    model = "fake"

    def __init__(self, *, fixed_text: str) -> None:
        super().__init__()
        self._fixed_text = fixed_text
        self.calls = 0
        self.last_messages: list[ChatMessage] = []

    @property
    def context_window_tokens(self) -> int:
        return 128_000

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> LLMAdapterResponse:
        _ = temperature, max_tokens, run_id
        self.calls += 1
        self.last_messages = list(messages)
        return build_adapter_response(content=self._fixed_text)


def _unique_tenant(prefix: str = "tenant") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


def _headers(tenant_id: str) -> dict[str, str]:
    return {"X-Tenant-Id": tenant_id}


def _search_task_result(
    *,
    workspace_id: str,
    evidence: list[dict[str, Any]],
) -> TaskResult:
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
def ask_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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
                "answer": "Payment is due within 14 days.",
                "used_evidence_ids": ["E1"],
            }
        )
    )
    settings = LocalWorkspaceBackendSettings.from_env()
    app = create_local_workspace_backend_app(settings=settings)
    app.state.lkw_ask_service.llm_adapter = llm

    with TestClient(app) as client:
        yield {
            "client": client,
            "store": store,
            "workspace_root": workspace_root,
            "llm": llm,
            "app": app,
        }


def _create_workspace(client: TestClient, tenant: str, name: str = "Ask Case") -> str:
    created = client.post(
        f"{_PREFIX}/workspaces",
        headers=_headers(tenant),
        json={"name": name},
    )
    assert created.status_code == 201, created.text
    return created.json()["workspace_id"]


def _seed_visible_local_document(
    app: Any,
    *,
    tenant_id: str,
    workspace_id: str,
    source_path: Path,
    document_id: str = "doc-1",
) -> WorkspaceDocumentReference:
    managed = app.state.lkw_managed_workspace_service
    source = managed.register_local_folder_source(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        path=str(source_path.parent),
    )
    document_ref = WorkspaceDocumentReference(
        document_id=document_id,
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        source_id=source.source_id,
        source_path=str(source_path.resolve()),
        file_name=source_path.name,
        content_hash="sha256:abc",
        indexed_at=datetime.now(UTC),
    )
    managed.repository.put_document_ref(document_ref)
    return document_ref


def _stub_search(
    app: Any,
    *,
    document_ref: WorkspaceDocumentReference,
    snippet: str = "Payment is due within 14 days.",
    extra_metadata: dict[str, Any] | None = None,
    evidence_workspace_id: str | None = None,
) -> None:
    metadata = {"provider_vector_id": "vec-secret", "slack_channel_id": "C123", "chunk_id": "chk-1"}
    if extra_metadata:
        metadata.update(extra_metadata)
    evidence = [
        {
            "document_id": document_ref.document_id,
            "source_id": document_ref.source_id,
            "workspace_id": evidence_workspace_id or document_ref.workspace_id,
            "source_path": document_ref.source_path,
            "file_name": document_ref.file_name,
            "score": 0.95,
            "snippet": snippet,
            "metadata": metadata,
        }
    ]
    executor = app.state.lkw_ask_service._executor
    executor.execute = AsyncMock(
        return_value=_search_task_result(
            workspace_id=document_ref.workspace_id,
            evidence=evidence,
        )
    )


def test_ask_workspace_search_evidence_to_citations_without_provider_or_slack_leakage(ask_api) -> None:
    client = ask_api["client"]
    root: Path = ask_api["workspace_root"]
    app = ask_api["app"]
    llm: RecordingFakeLLM = ask_api["llm"]

    tenant = _unique_tenant("tenant-a")
    workspace_id = _create_workspace(client, tenant)
    path = root / "invoice.txt"
    path.write_text("Payment is due within 14 days.", encoding="utf-8")
    document_ref = _seed_visible_local_document(
        app,
        tenant_id=tenant,
        workspace_id=workspace_id,
        source_path=path,
    )
    _stub_search(app, document_ref=document_ref)

    response = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/ask",
        headers=_headers(tenant),
        json={"question": "When is payment due?", "limit": 10},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "completed"
    assert body["answer"]
    assert body["run_id"]
    assert body["citations"]
    citation = body["citations"][0]
    assert citation["evidence_id"] == "E1"
    assert citation["document_id"] == "doc-1"
    assert citation["file_name"] == "invoice.txt"
    assert "14 days" in citation["excerpt"]
    assert "provider_vector_id" not in citation
    assert "slack_channel_id" not in citation
    assert "slack" not in json.dumps(body).lower() or "slack" not in str(citation)

    serialized = json.dumps(body)
    assert "vec-secret" not in serialized
    assert "C123" not in serialized
    assert "provider_vector_id" not in serialized

    user_payload = json.loads(llm.last_messages[-1].content or "")
    assert "provider_vector_id" not in json.dumps(user_payload)
    assert "slack_channel_id" not in json.dumps(user_payload)


def test_ask_happy_path_api(ask_api) -> None:
    client = ask_api["client"]
    root: Path = ask_api["workspace_root"]
    app = ask_api["app"]

    tenant = _unique_tenant()
    workspace_id = _create_workspace(client, tenant)
    path = root / "terms.txt"
    path.write_text("Payment is due within 14 days.", encoding="utf-8")
    document_ref = _seed_visible_local_document(
        app,
        tenant_id=tenant,
        workspace_id=workspace_id,
        source_path=path,
    )
    _stub_search(app, document_ref=document_ref)

    response = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/ask",
        headers=_headers(tenant),
        json={"question": "When is payment due?"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "completed"
    assert body["answer"]
    assert body["citations"]
    assert body["run_id"]
    assert body["workspace_id"] == workspace_id


def test_ask_empty_question_returns_422(ask_api) -> None:
    client = ask_api["client"]
    tenant = _unique_tenant()
    workspace_id = _create_workspace(client, tenant)

    empty = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/ask",
        headers=_headers(tenant),
        json={"question": ""},
    )
    assert empty.status_code == 422

    blank = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/ask",
        headers=_headers(tenant),
        json={"question": "   "},
    )
    assert blank.status_code == 422


def test_ask_tenant_isolation(ask_api) -> None:
    client = ask_api["client"]
    root: Path = ask_api["workspace_root"]
    app = ask_api["app"]

    tenant_a = _unique_tenant("tenant-a")
    tenant_b = _unique_tenant("tenant-b")
    workspace_id = _create_workspace(client, tenant_a)
    path = root / "a.txt"
    path.write_text("Payment is due within 14 days.", encoding="utf-8")
    document_ref = _seed_visible_local_document(
        app,
        tenant_id=tenant_a,
        workspace_id=workspace_id,
        source_path=path,
    )
    _stub_search(app, document_ref=document_ref)

    foreign_ask = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/ask",
        headers=_headers(tenant_b),
        json={"question": "When is payment due?"},
    )
    assert foreign_ask.status_code == 404

    own = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/ask",
        headers=_headers(tenant_a),
        json={"question": "When is payment due?"},
    )
    assert own.status_code == 200, own.text
    run_id = own.json()["run_id"]

    foreign_get = client.get(f"{_PREFIX}/asks/{run_id}", headers=_headers(tenant_b))
    assert foreign_get.status_code == 404

    own_get = client.get(f"{_PREFIX}/asks/{run_id}", headers=_headers(tenant_a))
    assert own_get.status_code == 200
    assert own_get.json()["run_id"] == run_id


def test_ask_workspace_isolation_drops_foreign_evidence(ask_api) -> None:
    client = ask_api["client"]
    root: Path = ask_api["workspace_root"]
    app = ask_api["app"]
    llm: RecordingFakeLLM = ask_api["llm"]

    tenant = _unique_tenant()
    workspace_id = _create_workspace(client, tenant)
    path = root / "a.txt"
    path.write_text("secret", encoding="utf-8")
    document_ref = _seed_visible_local_document(
        app,
        tenant_id=tenant,
        workspace_id=workspace_id,
        source_path=path,
    )
    # Evidence claims a different workspace — must be dropped → empty → insufficient.
    _stub_search(
        app,
        document_ref=document_ref,
        evidence_workspace_id="other-workspace",
    )

    response = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/ask",
        headers=_headers(tenant),
        json={"question": "What is secret?"},
    )
    assert response.status_code == 502
    assert response.json()["detail"] == "search_evidence_incomplete"
    assert llm.calls == 0


def test_ask_persisted_completed_run_read(ask_api) -> None:
    client = ask_api["client"]
    root: Path = ask_api["workspace_root"]
    app = ask_api["app"]

    tenant = _unique_tenant()
    workspace_id = _create_workspace(client, tenant)
    path = root / "a.txt"
    path.write_text("Payment is due within 14 days.", encoding="utf-8")
    document_ref = _seed_visible_local_document(
        app,
        tenant_id=tenant,
        workspace_id=workspace_id,
        source_path=path,
    )
    _stub_search(app, document_ref=document_ref)

    posted = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/ask",
        headers=_headers(tenant),
        json={"question": "When is payment due?"},
    )
    assert posted.status_code == 200, posted.text
    posted_body = posted.json()
    run_id = posted_body["run_id"]

    fetched = client.get(f"{_PREFIX}/asks/{run_id}", headers=_headers(tenant))
    assert fetched.status_code == 200
    assert fetched.json() == posted_body


def test_ask_restart_persistence_via_repository_reload(ask_api) -> None:
    client = ask_api["client"]
    store = ask_api["store"]
    root: Path = ask_api["workspace_root"]
    app = ask_api["app"]

    tenant = _unique_tenant()
    workspace_id = _create_workspace(client, tenant)
    path = root / "a.txt"
    path.write_text("Payment is due within 14 days.", encoding="utf-8")
    document_ref = _seed_visible_local_document(
        app,
        tenant_id=tenant,
        workspace_id=workspace_id,
        source_path=path,
    )
    _stub_search(app, document_ref=document_ref)

    posted = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/ask",
        headers=_headers(tenant),
        json={"question": "When is payment due?"},
    )
    assert posted.status_code == 200, posted.text
    run_id = posted.json()["run_id"]
    posted_body = posted.json()

    # Dispose app-facing repository; reload from same durable store without model/search.
    reloaded = WorkspaceAskRepository(store)
    run = reloaded.get_run(tenant_id=tenant, run_id=run_id)
    assert run is not None
    assert run.status == AskRunStatus.COMPLETED
    assert run.answer == posted_body["answer"]
    assert run.citations


def test_ask_assembly_failure_persisted(ask_api) -> None:
    client = ask_api["client"]
    root: Path = ask_api["workspace_root"]
    app = ask_api["app"]
    llm: RecordingFakeLLM = ask_api["llm"]
    llm._fixed_text = json.dumps(
        {
            "status": "completed",
            "answer": "Bad grounding",
            "used_evidence_ids": ["E99"],
        }
    )

    tenant = _unique_tenant()
    workspace_id = _create_workspace(client, tenant)
    path = root / "a.txt"
    path.write_text("Payment is due within 14 days.", encoding="utf-8")
    document_ref = _seed_visible_local_document(
        app,
        tenant_id=tenant,
        workspace_id=workspace_id,
        source_path=path,
    )
    _stub_search(app, document_ref=document_ref)

    response = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/ask",
        headers=_headers(tenant),
        json={"question": "When is payment due?"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "failed"
    assert body["answer"] is None
    assert body["citations"] == []
    assert body["error"] is not None
    assert body["error"]["code"] == "unknown_evidence_reference"
    assert "Traceback" not in json.dumps(body)

    fetched = client.get(f"{_PREFIX}/asks/{body['run_id']}", headers=_headers(tenant))
    assert fetched.status_code == 200
    assert fetched.json()["status"] == "failed"
    assert fetched.json()["error"]["code"] == "unknown_evidence_reference"


def test_ask_insufficient_evidence_skips_model(ask_api) -> None:
    client = ask_api["client"]
    app = ask_api["app"]
    llm: RecordingFakeLLM = ask_api["llm"]

    tenant = _unique_tenant()
    workspace_id = _create_workspace(client, tenant)
    executor = app.state.lkw_ask_service._executor
    executor.execute = AsyncMock(
        return_value=_search_task_result(workspace_id=workspace_id, evidence=[])
    )

    response = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/ask",
        headers=_headers(tenant),
        json={"question": "Anything in the empty workspace?"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "insufficient_evidence"
    assert body["answer"] is None
    assert body["citations"] == []
    assert llm.calls == 0


def test_ask_rejects_slack_fields_in_request(ask_api) -> None:
    client = ask_api["client"]
    tenant = _unique_tenant()
    workspace_id = _create_workspace(client, tenant)
    response = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/ask",
        headers=_headers(tenant),
        json={
            "question": "Hello",
            "slack_channel_id": "C123",
        },
    )
    assert response.status_code == 422


def test_listed_workspace_ask_not_404_same_store(ask_api) -> None:
    """LKW-SLACK-ASK-404-FIX: listed + GET workspace must be Ask-usable (same store)."""
    client = ask_api["client"]
    store = ask_api["store"]
    root: Path = ask_api["workspace_root"]
    app = ask_api["app"]

    tenant_a = _unique_tenant("tenant-a")
    tenant_b = _unique_tenant("tenant-b")
    workspace_id = _create_workspace(client, tenant_a, name="Listed Askable")

    listed = client.get(f"{_PREFIX}/workspaces", headers=_headers(tenant_a))
    assert listed.status_code == 200
    ids = [item["workspace_id"] for item in listed.json()["workspaces"]]
    assert workspace_id in ids

    got = client.get(f"{_PREFIX}/workspaces/{workspace_id}", headers=_headers(tenant_a))
    assert got.status_code == 200

    path = root / "policy.txt"
    path.write_text("Payment is due within 14 days.", encoding="utf-8")
    document_ref = _seed_visible_local_document(
        app,
        tenant_id=tenant_a,
        workspace_id=workspace_id,
        source_path=path,
    )
    _stub_search(app, document_ref=document_ref)

    ask_service = app.state.lkw_ask_service
    managed = app.state.lkw_managed_workspace_service
    assert ask_service._workspaces is managed
    assert ask_service._workspace_repo is managed.repository
    assert ask_service._workspace_repo.document_store is store
    assert ask_service._ask_repo.document_store is store

    own = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/ask",
        headers=_headers(tenant_a),
        json={"question": "When is payment due?"},
    )
    assert own.status_code != 404
    assert own.status_code == 200, own.text
    assert own.json()["status"] in {"completed", "insufficient_evidence", "failed"}

    cross = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/ask",
        headers=_headers(tenant_b),
        json={"question": "When is payment due?"},
    )
    assert cross.status_code == 404
    assert cross.json()["detail"] == "not_found"

    missing = client.post(
        f"{_PREFIX}/workspaces/{uuid.uuid4()}/ask",
        headers=_headers(tenant_a),
        json={"question": "When is payment due?"},
    )
    assert missing.status_code == 404
    assert missing.json()["detail"] == "not_found"


def test_ask_keyerror_during_search_is_not_workspace_404(ask_api) -> None:
    """LookupError subclasses (e.g. KeyError) must not be mapped to workspace 404."""
    client = ask_api["client"]
    app = ask_api["app"]

    tenant = _unique_tenant()
    workspace_id = _create_workspace(client, tenant)
    executor = app.state.lkw_ask_service._executor
    executor.execute = AsyncMock(side_effect=KeyError("synthetic_search_key"))

    response = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/ask",
        headers=_headers(tenant),
        json={"question": "Trigger non-workspace LookupError subclass"},
    )
    assert response.status_code == 502
    assert response.json()["detail"] == "ask_error: KeyError"