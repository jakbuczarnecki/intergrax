# © Artur Czarnecki. All rights reserved.

"""End-to-end WEB_URL intake → indexing → grounded Ask proof."""

from __future__ import annotations

import json
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional, Sequence

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from intergrax.integrations._shared.in_memory_document_store import (
from intergrax.runtime.background_execution.identity_persistence import wire_background_execution_identity_persistence
    InMemoryDocumentStore,
)
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.adapter_response_builders import (
    build_adapter_response,
)
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.queueing.providers.document_store.colocated_worker import (
    DocumentStoreTaskWorker,
)
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.contracts.agent_execution_result import (
    AgentExecutionResult,
    AgentExecutionStatus,
)
from intergrax.runtime.task.task import TaskResult, TaskState
from intergrax.websearch.capture.contracts import (
    CapturedWebContent,
    WebContentCaptureRequest,
)
from intergrax.websearch.capture.url_policy import WebUrlAccessPolicy
from intergrax.applications._shared.harness_host_runtime import (
    build_harness_host_runtime,
)
from intergrax.applications._shared.harness_host_runtime_compat import resolve_harness_host_nexus_loop_legacy
from intergrax.rag.vectorstore.contracts.native_vectorstore import VectorStoreScope
from local_workspace_application.host.environment_profile import (
    build_local_workspace_environment_profile,
)
from local_workspace_application.host.lkw_task_enricher import (
    build_lkw_combined_task_enricher,
)
from local_workspace_application.host.lifecycle import LocalWorkspaceHostLifecycle
from local_workspace_application.host.execution_wiring import build_lkw_host_task_execution
from local_workspace_application.host.task_executor import LocalWorkspaceTaskExecutor
from local_workspace_application.manifest import LOCAL_WORKSPACE_APPLICATION_MANIFEST
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.serving import workspace_routes
from local_workspace_application.serving.workspace_routes import (
    mount_managed_workspace_routes,
)
from local_workspace_application.workspaces.document_indexing import (
    WorkspaceDocumentIndexingResult,
)
from local_workspace_application.workspaces.knowledge_ingestion import (
    register_knowledge_ingestion_worker_handler,
)
from local_workspace_application.workspaces.models import (
    WorkspaceDocumentReference,
    WorkspaceOperationStatus,
    WorkspaceSourceStatus,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository
from local_workspace_application.workspaces.sync_runtime import (
    build_managed_workspace_sync_runtime,
)
from local_workspace_application.workspaces.sync_service import (
    ManagedWorkspaceSyncService,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_PREFIX = "/v1/local_workspace"
FIXTURE_TEXT = "The Oriole warehouse verification code is BLUE-7319."
PUBLIC_IP = "93.184.216.34"
SUBMIT_URL = "https://example.com/oriole?track=secret"
DISPLAY_URL = "https://example.com/oriole"
RETRIEVAL_MARKER = "BLUE-7319"


def _assert_web_url_rag_retrieval_evidence(
    *,
    citation: dict[str, Any],
    source_id: str,
    marker: str = RETRIEVAL_MARKER,
    display_url: str = DISPLAY_URL,
) -> None:
    """Require the public Ask citation excerpt to contain the retrieved fixture marker."""
    assert citation["source_id"] == source_id
    assert citation["file_name"] == display_url
    assert marker in citation["excerpt"], (
        "retrieved evidence excerpt must contain the fixture marker"
    )
    assert "track=secret" not in json.dumps(citation)


async def _resolve_public(_host: str) -> tuple[str, ...]:
    return (PUBLIC_IP,)


class RecordingFakeLLM(LLMAdapter):
    provider = "fake"
    model = "fake"

    def __init__(self, *, fixed_text: str) -> None:
        super().__init__()
        self._fixed_text = fixed_text

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
        _ = temperature, max_tokens, run_id, messages
        return build_adapter_response(content=self._fixed_text)


class FakeWebContentCapture:
    def __init__(
        self, *, policy: WebUrlAccessPolicy | None = None, text: str = FIXTURE_TEXT
    ) -> None:
        self._policy = policy or WebUrlAccessPolicy(dns_resolver=_resolve_public)
        self._text = text

    async def capture(self, request: WebContentCaptureRequest) -> CapturedWebContent:
        canonical = self._policy.canonicalize(request.url)
        now = datetime.now(UTC)
        return CapturedWebContent(
            safe_display_url=canonical.safe_display_url,
            requested_url_fingerprint=canonical.fingerprint,
            final_url_fingerprint="sha256:" + "b" * 64,
            final_host_changed=False,
            title="Oriole",
            text=self._text,
            content_type="text/html",
            content_hash="sha256:" + "c" * 64,
            status_code=200,
            redirect_count=0,
            content_bytes=len(self._text.encode()),
            text_chars=len(self._text),
            capture_mode="http",
            extraction_method="basic",
            fetched_at=now,
        )


class AskIndexingService:
    async def index_one(self, **kwargs: object) -> WorkspaceDocumentIndexingResult:
        tenant_id = str(kwargs["tenant_id"])
        workspace_id = str(kwargs["workspace_id"])
        source_id = str(kwargs["source_id"])
        logical_source_path = str(kwargs["logical_source_path"])
        safe_file_name = str(kwargs["safe_file_name"])
        content_hash = str(kwargs["content_hash"])
        document_id = f"doc-{source_id[-8:]}"
        repo: ManagedWorkspaceRepository = kwargs["repository"]  # type: ignore[assignment]
        repo.put_document_ref(
            WorkspaceDocumentReference(
                document_id=document_id,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                source_id=source_id,
                source_path=logical_source_path,
                file_name=safe_file_name,
                content_hash=content_hash,
                indexed_at=datetime.now(UTC),
            )
        )
        return WorkspaceDocumentIndexingResult(
            indexed=True,
            unchanged=False,
            document_id=document_id,
            documents_indexed=1,
            num_chunks=1,
            reason="ingest_complete",
        )


class _IndexingServiceAdapter:
    def __init__(self, repository: ManagedWorkspaceRepository) -> None:
        self._inner = AskIndexingService()
        self._repository = repository

    async def index_one(self, **kwargs: object) -> WorkspaceDocumentIndexingResult:
        return await self._inner.index_one(repository=self._repository, **kwargs)


def _search_task_result(
    *,
    workspace_id: str,
    source_id: str,
    file_name: str,
    document_id: str,
    source_path: str,
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
                    "query": "verification code",
                    "workspace_id": workspace_id,
                    "evidence": [
                        {
                            "document_id": document_id,
                            "source_id": source_id,
                            "workspace_id": workspace_id,
                            "source_path": source_path,
                            "file_name": file_name,
                            "snippet": FIXTURE_TEXT,
                            "evidence_id": "E1",
                            "score": 0.99,
                        }
                    ],
                }
            },
        ),
    )


class _FakeExecutor:
    async def execute(self, task: object) -> object:
        _ = task
        return type(
            "R",
            (),
            {
                "metadata": {
                    "ingest_summary": {
                        "used": True,
                        "reason": "ingest_complete",
                        "num_chunks": 1,
                    }
                }
            },
        )()


@pytest.fixture
def e2e_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    data_home = tmp_path / "data"
    data_home.mkdir()
    monkeypatch.setenv("DATA_HOME", str(data_home))
    monkeypatch.setenv("INTERGRAX_ALLOWED_READ_ROOTS", str(tmp_path / "docs"))
    (tmp_path / "docs").mkdir()
    settings = replace(
        LocalWorkspaceBackendSettings.from_env(), data_home=str(data_home)
    )
    executor = _FakeExecutor()
    sync = ManagedWorkspaceSyncService(repo, executor)  # type: ignore[arg-type]
    runtime = build_managed_workspace_sync_runtime(
        document_store=store,
        sync_service=sync,
        repository=repo,
    )
    policy = WebUrlAccessPolicy(dns_resolver=_resolve_public)
    capture = FakeWebContentCapture(policy=policy)
    llm = RecordingFakeLLM(
        fixed_text=json.dumps(
            {
                "status": "completed",
                "answer": "BLUE-7319",
                "used_evidence_ids": ["E1"],
            }
        )
    )
    app = FastAPI()
    service = mount_managed_workspace_routes(
        app,
        task_executor=executor,  # type: ignore[arg-type]
        settings=settings,
        repository=repo,
        sync_runtime=runtime,
        llm_adapter=llm,
        web_url_access_policy=policy,
        web_content_capture=capture,
        indexing_service=_IndexingServiceAdapter(repo),  # type: ignore[arg-type]
    )
    ingestion = app.state.lkw_knowledge_ingestion_service
    registry = TaskExecutionRegistry()
    register_knowledge_ingestion_worker_handler(registry, ingestion)
    worker = DocumentStoreTaskWorker(runtime.wiring_context.message_bus, registry, identity_persistence=wire_background_execution_identity_persistence(document_store=store))  # type: ignore[arg-type]

    async def _search_execute(task: Any) -> TaskResult:
        metadata = getattr(task, "metadata", {}) or {}
        workspace_id = str(metadata.get("workspace_id") or "")
        tenant_id = str(
            getattr(task, "tenant_id", "") or metadata.get("tenant_id") or ""
        )
        refs = repo.list_document_refs(tenant_id=tenant_id, workspace_id=workspace_id)
        if not refs:
            return _search_task_result(
                workspace_id=workspace_id,
                source_id="missing",
                file_name=DISPLAY_URL,
                document_id="missing",
                source_path="",
            )
        ref = refs[0]
        return _search_task_result(
            workspace_id=workspace_id,
            source_id=ref.source_id,
            file_name=ref.file_name,
            document_id=ref.document_id,
            source_path=ref.source_path,
        )

    app.state.lkw_ask_service._executor.execute = _search_execute  # type: ignore[method-assign]

    with TestClient(app) as client:
        yield {
            "client": client,
            "worker": worker,
            "repo": repo,
            "service": service,
            "llm": llm,
            "settings": settings,
        }


@pytest.fixture
def rag_e2e_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    store = InMemoryDocumentStore()
    data_home = tmp_path / "lkw-data"
    sqlite_dir = tmp_path / "sqlite"
    shadow_dir = tmp_path / "shadow"
    user_docs = tmp_path / "docs"
    for path in (data_home, sqlite_dir, shadow_dir, user_docs):
        path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("LOCAL_WORKSPACE_VECTOR_STORE", "inmemory")
    monkeypatch.setenv("INTERGRAX_ALLOWED_READ_ROOTS", str(user_docs.resolve()))
    monkeypatch.setenv("LOCAL_WORKSPACE_ENABLE_RAG", "true")
    monkeypatch.setenv("LOCAL_WORKSPACE_ENABLE_RAG_INGEST", "true")
    monkeypatch.setenv("LKW_DATA_HOME", str(data_home))
    monkeypatch.setenv("INTERGRAX_SQLITE_DATA_DIR", str(sqlite_dir))
    monkeypatch.setenv("INTERGRAX_SHADOW_ROOT", str(shadow_dir))
    monkeypatch.delenv("INTERGRAX_MONGODB_URI", raising=False)
    monkeypatch.setattr(
        workspace_routes,
        "resolve_managed_workspace_document_store",
        lambda document_store=None: store,
    )

    settings = LocalWorkspaceBackendSettings.from_env()
    env = build_local_workspace_environment_profile(settings)
    harness_runtime = build_harness_host_runtime(
        LOCAL_WORKSPACE_APPLICATION_MANIFEST,
        env,
        settings=settings,
    )
    lifecycle = LocalWorkspaceHostLifecycle()
    lifecycle.transition_to_ready()
    lifecycle.set_executor_available(True)
    task_enricher = build_lkw_combined_task_enricher(
        env,
        default_capability="local.workspace.search",
        agent_checkpoint_store=harness_runtime.agent_checkpoint_store,
        compensation_queue_store=harness_runtime.compensation_queue_store,
        idempotency_store=harness_runtime.reliability.idempotency_store,
    )
    nexus_loop = resolve_harness_host_nexus_loop_legacy(harness_runtime)
    task_executor = LocalWorkspaceTaskExecutor(
        build_lkw_host_task_execution(nexus_loop, env),
        nexus_loop=nexus_loop,
        task_enricher=task_enricher,
        readiness=lifecycle,
    )
    repo = ManagedWorkspaceRepository(store)
    sync = ManagedWorkspaceSyncService(repo, task_executor)
    runtime = build_managed_workspace_sync_runtime(
        document_store=store,
        sync_service=sync,
        repository=repo,
    )
    policy = WebUrlAccessPolicy(dns_resolver=_resolve_public)
    capture = FakeWebContentCapture(policy=policy)
    llm = RecordingFakeLLM(
        fixed_text=json.dumps(
            {
                "status": "completed",
                "answer": "The verification code is BLUE-7319.",
                "used_evidence_ids": ["E1"],
            }
        )
    )
    app = FastAPI()
    mount_managed_workspace_routes(
        app,
        task_executor=task_executor,
        settings=settings,
        repository=repo,
        sync_runtime=runtime,
        llm_adapter=llm,
        web_url_access_policy=policy,
        web_content_capture=capture,
        vectorstore_manager=harness_runtime.env_wiring.tool_wiring.wiring_context.vectorstore_manager,
    )
    vectorstore_manager = (
        harness_runtime.env_wiring.tool_wiring.wiring_context.vectorstore_manager
    )

    with TestClient(app) as client:
        yield {
            "client": client,
            "repo": repo,
            "runtime": runtime,
            "llm": llm,
            "settings": settings,
            "vectorstore_manager": vectorstore_manager,
            "task_executor": task_executor,
            "harness_runtime": harness_runtime,
        }


def test_web_url_http_worker_and_ask_projection_with_test_doubles(e2e_bundle) -> None:
    client = e2e_bundle["client"]
    worker = e2e_bundle["worker"]
    repo: ManagedWorkspaceRepository = e2e_bundle["repo"]
    tenant = f"tenant-{uuid.uuid4().hex[:8]}"

    created = client.post(
        f"{_PREFIX}/workspaces",
        headers={"X-Tenant-Id": tenant},
        json={"name": "Oriole"},
    )
    assert created.status_code == 201
    workspace_id = created.json()["workspace_id"]

    accepted = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/knowledge/web-urls",
        headers={"X-Tenant-Id": tenant, "Idempotency-Key": "oriole-url"},
        json={"url": SUBMIT_URL},
    )
    assert accepted.status_code == 202, accepted.text
    body = accepted.json()
    source_id = body["source_id"]
    operation_id = body["operation_id"]
    assert body["safe_display_url"] == DISPLAY_URL
    assert "track=secret" not in json.dumps(body)

    assert worker.drain_once() == 1
    op = repo.get_operation(tenant_id=tenant, operation_id=operation_id)
    assert op is not None
    assert op.status is WorkspaceOperationStatus.COMPLETED
    source = repo.get_source(
        tenant_id=tenant, workspace_id=workspace_id, source_id=source_id
    )
    assert source is not None
    assert source.status is WorkspaceSourceStatus.READY
    assert repo.list_document_refs(tenant_id=tenant, workspace_id=workspace_id)

    ask = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/ask",
        headers={"X-Tenant-Id": tenant},
        json={"question": "What is the Oriole warehouse verification code?"},
    )
    assert ask.status_code == 200, ask.text
    answer_body = ask.json()
    assert "BLUE-7319" in answer_body["answer"]
    assert answer_body["citations"]
    citation = answer_body["citations"][0]
    assert citation["source_id"] == source_id
    assert citation["file_name"] == DISPLAY_URL
    assert "track=secret" not in json.dumps(answer_body)


def test_web_url_end_to_end_real_rag_ask_proof(rag_e2e_bundle) -> None:
    client = rag_e2e_bundle["client"]
    repo: ManagedWorkspaceRepository = rag_e2e_bundle["repo"]
    runtime = rag_e2e_bundle["runtime"]
    settings: LocalWorkspaceBackendSettings = rag_e2e_bundle["settings"]
    vectorstore_manager = rag_e2e_bundle["vectorstore_manager"]
    _ = vectorstore_manager
    tenant = f"tenant-{uuid.uuid4().hex[:8]}"

    assert settings.web_url_staging_dir in settings.allowed_read_roots

    created = client.post(
        f"{_PREFIX}/workspaces",
        headers={"X-Tenant-Id": tenant},
        json={"name": "Oriole RAG"},
    )
    assert created.status_code == 201
    workspace_id = created.json()["workspace_id"]

    accepted = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/knowledge/web-urls",
        headers={"X-Tenant-Id": tenant, "Idempotency-Key": "oriole-rag"},
        json={"url": SUBMIT_URL},
    )
    assert accepted.status_code == 202, accepted.text
    body = accepted.json()
    source_id = body["source_id"]
    operation_id = body["operation_id"]
    assert body["safe_display_url"] == DISPLAY_URL
    assert "track=secret" not in json.dumps(body)

    worker = runtime.worker
    assert worker.drain_once() == 1

    op = repo.get_operation(tenant_id=tenant, operation_id=operation_id)
    assert op is not None
    assert op.status is WorkspaceOperationStatus.COMPLETED, (op.error_code, op.error)
    assert op.documents_indexed >= 1

    source = repo.get_source(
        tenant_id=tenant, workspace_id=workspace_id, source_id=source_id
    )
    assert source is not None
    assert source.status is WorkspaceSourceStatus.READY

    refs = repo.list_document_refs(tenant_id=tenant, workspace_id=workspace_id)
    assert refs
    assert refs[0].source_id == source_id

    staging_root = Path(settings.web_url_staging_dir)
    assert not any(staging_root.rglob("web-content.txt"))

    wiring_ctx = rag_e2e_bundle["harness_runtime"].env_wiring.tool_wiring.wiring_context
    tenant_stores = wiring_ctx.extras.get("tenant_vectorstore_managers", {})
    scoped_manager = tenant_stores.get(tenant)
    assert scoped_manager is not None
    assert (
        scoped_manager.count(
            scope=VectorStoreScope(
                tenant_id=tenant,
                workspace_id=workspace_id,
            )
        )
        >= 1
    )

    ask = client.post(
        f"{_PREFIX}/workspaces/{workspace_id}/ask",
        headers={"X-Tenant-Id": tenant},
        json={"question": "What is the Oriole warehouse verification code?"},
    )
    assert ask.status_code == 200, ask.text
    answer_body = ask.json()
    assert RETRIEVAL_MARKER in answer_body["answer"]
    assert answer_body["citations"]
    citation = answer_body["citations"][0]
    _assert_web_url_rag_retrieval_evidence(citation=citation, source_id=source_id)
    assert "track=secret" not in json.dumps(answer_body)


def test_web_url_rag_proof_rejects_answer_only_marker_match() -> None:
    with pytest.raises(AssertionError, match="retrieved evidence excerpt must contain"):
        _assert_web_url_rag_retrieval_evidence(
            citation={
                "source_id": "src-oriole",
                "file_name": DISPLAY_URL,
                "excerpt": "The Oriole warehouse verification code is RED-0000.",
            },
            source_id="src-oriole",
        )
