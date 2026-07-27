# © Artur Czarnecki. All rights reserved.

"""Integration: Slack Source Candidate selection → public API → Knowledge Intake."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.conversation_channel import (
    ConversationActor,
    ConversationAddress,
    ConversationDeliveryReceipt,
    ConversationEventKind,
    InboundConversationEvent,
    OutboundConversationMessage,
)
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.serving.workspace_routes import mount_managed_workspace_routes
from local_workspace_application.slack_companion.ask_client import (
    SlackAskClientConfig,
    WorkspaceAskHttpClient,
)
from local_workspace_application.slack_companion.authorization import SlackCompanionAuthConfig
from local_workspace_application.slack_companion.dedupe_repository import (
    SlackEventDedupeRepository,
)
from local_workspace_application.slack_companion.workflow import (
    SlackAskWorkflow,
    slack_source_candidate_intake_idempotency_key,
)
from local_workspace_application.workspaces.models import (
    KnowledgeInputKind,
    WorkspaceOperationStatus,
    WorkspaceOperationType,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository
from local_workspace_application.workspaces.sync_runtime import (
    build_managed_workspace_sync_runtime,
)
from local_workspace_application.workspaces.sync_service import ManagedWorkspaceSyncService

pytestmark = pytest.mark.unit

_PREFIX = "/v1/local_workspace"


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


def _write_candidates(path: Path, candidates: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "lkw.source_candidates.v1",
                "candidates": candidates,
            }
        ),
        encoding="utf-8",
    )


def _event(*, event_id: str, text: str) -> InboundConversationEvent:
    return InboundConversationEvent(
        event_id=event_id,
        address=ConversationAddress(
            installation_id="T_OK",
            conversation_id="Dchannel",
            thread_id="1712222.000300",
        ),
        actor=ConversationActor(actor_id="U_OK", is_bot=False),
        kind=ConversationEventKind.MESSAGE,
        text=text,
    )


def test_slack_source_candidate_intake_integration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder_a = tmp_path / "contracts"
    folder_a.mkdir()
    (folder_a / "a.txt").write_text("contract body", encoding="utf-8")
    folder_b = tmp_path / "product"
    folder_b.mkdir()
    (folder_b / "b.txt").write_text("product body", encoding="utf-8")

    data_home = tmp_path / "data"
    data_home.mkdir()
    config_file = data_home / "config" / "source_candidates.json"
    _write_candidates(
        config_file,
        [
            {
                "candidate_id": "contracts",
                "tenant_id": "tenant-a",
                "label": "Contracts",
                "description": "Current contract documents",
                "source_type": "local_folder",
                "path": str(folder_a.resolve()),
                "recursive": True,
                "enabled": True,
            },
            {
                "candidate_id": "product_docs",
                "tenant_id": "tenant-a",
                "label": "Product documentation",
                "description": "Approved product materials",
                "source_type": "local_folder",
                "path": str(folder_b.resolve()),
                "recursive": True,
                "enabled": True,
            },
        ],
    )
    monkeypatch.setenv("DATA_HOME", str(data_home))
    monkeypatch.setenv(
        "INTERGRAX_ALLOWED_READ_ROOTS",
        f"{folder_a.resolve()};{folder_b.resolve()}",
    )

    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    settings = replace(
        LocalWorkspaceBackendSettings.from_env(),
        data_home=str(data_home),
        allowed_read_roots=frozenset(
            {str(folder_a.resolve()), str(folder_b.resolve())}
        ),
    )
    executor = _FakeExecutor()
    sync = ManagedWorkspaceSyncService(repo, executor)  # type: ignore[arg-type]
    runtime = build_managed_workspace_sync_runtime(
        document_store=store,
        sync_service=sync,
        repository=repo,
    )
    app = FastAPI()
    mount_managed_workspace_routes(
        app,
        task_executor=executor,  # type: ignore[arg-type]
        settings=settings,
        repository=repo,
        sync_runtime=runtime,
        object_storage=None,
    )

    with TestClient(app) as client:
        created = client.post(
            f"{_PREFIX}/workspaces",
            headers={"X-Tenant-Id": "tenant-a"},
            json={"name": "Docs"},
        )
        assert created.status_code == 201
        workspace_id = created.json()["workspace_id"]

        ask_paths: list[str] = []
        list_paths: list[str] = []
        accept_paths: list[str] = []

        class _CountingTransport(httpx.AsyncBaseTransport):
            def __init__(self, inner: httpx.AsyncBaseTransport) -> None:
                self._inner = inner

            async def handle_async_request(
                self, request: httpx.Request
            ) -> httpx.Response:
                path = request.url.path
                if path.endswith("/ask"):
                    ask_paths.append(path)
                if path.endswith("/source-candidates") and request.method == "GET":
                    list_paths.append(path)
                if (
                    "/knowledge/source-candidates/" in path
                    and request.method == "POST"
                ):
                    accept_paths.append(path)
                return await self._inner.handle_async_request(request)

        outbound: list[str] = []

        async def send(
            message: OutboundConversationMessage,
        ) -> ConversationDeliveryReceipt:
            outbound.append(message.text)
            return ConversationDeliveryReceipt(
                message_id="m1",
                address=message.address,
                delivered_at=datetime.now(UTC),
            )

        ask_client = WorkspaceAskHttpClient(
            SlackAskClientConfig(base_url="http://lkw.test"),
            transport=_CountingTransport(httpx.ASGITransport(app=app)),
        )
        workflow = SlackAskWorkflow(
            auth_config=SlackCompanionAuthConfig(
                approved_team_id="T_OK",
                approved_user_id="U_OK",
                tenant_id="tenant-a",
                active_workspace_id=workspace_id,
            ),
            dedupe=SlackEventDedupeRepository(InMemoryDocumentStore()),
            ask_client=ask_client,
            send=send,
        )

        import local_workspace_application.slack_companion.workflow as workflow_mod

        assert "SourceCandidateRegistry" not in workflow_mod.__dict__
        assert "SourceCandidateIntakeService" not in workflow_mod.__dict__

        async def _run_slack_flow() -> None:
            await workflow.handle(_event(event_id="Ev-list", text="source candidates"))
            await workflow.handle(_event(event_id="Ev-add", text="source add 2"))

        asyncio.run(_run_slack_flow())

        assert ask_paths == []
        assert len(list_paths) == 2
        assert len(accept_paths) == 1
        assert accept_paths[0].endswith(
            f"/workspaces/{workspace_id}/knowledge/source-candidates/product_docs"
        )

        list_text = outbound[0]
        assert "Available source candidates:" in list_text
        assert "1. Contracts" in list_text
        assert "2. Product documentation" in list_text
        assert "product_docs" not in list_text
        assert str(folder_a) not in list_text
        assert str(folder_b) not in list_text
        assert "sha256:" not in list_text
        assert workspace_id not in list_text

        accept_text = outbound[1]
        assert "Source accepted: Product documentation" in accept_text
        assert "Processing continues asynchronously." in accept_text
        assert "product_docs" not in accept_text
        assert str(folder_b) not in accept_text
        assert "sha256:" not in accept_text
        assert "indexed" not in accept_text.casefold()
        assert "completed" not in accept_text.casefold()

        inputs = repo.list_knowledge_inputs(
            tenant_id="tenant-a", workspace_id=workspace_id
        )
        assert len(inputs) == 1
        assert inputs[0].input_kind is KnowledgeInputKind.SOURCE_CANDIDATE
        sources = repo.list_sources(tenant_id="tenant-a", workspace_id=workspace_id)
        assert len(sources) == 1
        assert Path(sources[0].path).resolve() == folder_b.resolve()
        assert inputs[0].input_id not in accept_text
        assert sources[0].source_id not in accept_text

        operations = [
            op
            for op in repo.list_operations(tenant_id="tenant-a")
            if op.operation_type is WorkspaceOperationType.KNOWLEDGE_INGESTION
            and op.workspace_id == workspace_id
        ]
        assert len(operations) == 1
        assert operations[0].queue_task_id
        assert operations[0].operation_id not in accept_text

        assert runtime.worker.drain_once() == 1
        refreshed = repo.get_operation(
            tenant_id="tenant-a",
            operation_id=operations[0].operation_id,
        )
        assert refreshed is not None
        assert refreshed.status is WorkspaceOperationStatus.COMPLETED
        assert refreshed.documents_indexed >= 1

        expected_key = slack_source_candidate_intake_idempotency_key(
            team_id="T_OK",
            event_id="Ev-add",
        )
        again = client.post(
            f"{_PREFIX}/workspaces/{workspace_id}/knowledge/source-candidates/product_docs",
            headers={
                "X-Tenant-Id": "tenant-a",
                "Idempotency-Key": expected_key,
            },
        )
        assert again.status_code == 202
        payload = again.json()
        assert payload["source_id"] == sources[0].source_id
        assert payload["operation_id"] == operations[0].operation_id

        assert (
            len(
                repo.list_knowledge_inputs(
                    tenant_id="tenant-a", workspace_id=workspace_id
                )
            )
            == 1
        )
        assert (
            len(repo.list_sources(tenant_id="tenant-a", workspace_id=workspace_id))
            == 1
        )
        assert ask_paths == []
