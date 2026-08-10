# © Artur Czarnecki. All rights reserved.

"""End-to-end Slack connected source proof through HTTP, sync, Search and Ask."""

from __future__ import annotations

import time
import uuid
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.conversation_channel.slack.integration import (
    SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
)
from intergrax.rag.vectorstore.contracts.native_vectorstore import VectorStoreScope

from .rag_e2e_support import (
    _ATLAS_INVESTIGATION,
    _CONNECTION,
    _LATEST,
    _MARKER_EDIT,
    _MARKER_REPLY,
    _MARKER_ROOT,
    _OLDEST,
    _PREFIX,
    _RecordingFakeLLM,
    _TENANT,
    _UNRELATED_MESSAGE,
    _WORKSPACE,
    _SlackFakeBackend,
)

pytestmark = [pytest.mark.unit]


def _assert_slack_rag_citation(
    *,
    citation: dict[str, Any],
    source_id: str,
    marker: str,
) -> None:
    assert citation["source_id"] == source_id
    assert marker in citation["excerpt"]


def _wait_operation(client: TestClient, operation_id: str) -> dict[str, object]:
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        response = client.get(
            f"{_PREFIX}/operations/{operation_id}",
            headers={"X-Tenant-Id": _TENANT},
        )
        assert response.status_code == 200
        body = response.json()
        if body["status"] in {"completed", "failed"}:
            return body
        time.sleep(0.1)
    raise AssertionError("operation_timeout")


def test_slack_connected_source_http_to_search_and_ask(rag_e2e_env) -> None:
    client = rag_e2e_env["client"]
    backend: _SlackFakeBackend = rag_e2e_env["backend"]
    wiring = rag_e2e_env["wiring"]
    repo: ManagedWorkspaceRepository = rag_e2e_env["repo"]
    integration = rag_e2e_env["integration"]
    runtime = rag_e2e_env["runtime"]
    app: FastAPI = rag_e2e_env["app"]
    source_catalog = rag_e2e_env["source_catalog"]
    live_catalog = rag_e2e_env["live_catalog"]
    harness_runtime = rag_e2e_env["harness_runtime"]
    llm: _RecordingFakeLLM = rag_e2e_env["llm"]

    source_capabilities = source_catalog.list_source_kind_capabilities(
        tenant_id=_TENANT,
        connection_ref=_CONNECTION,
    )
    assert len(source_capabilities) == 1
    assert source_capabilities[0].identity.source_kind == "slack_conversation"
    assert {mode.value for mode in source_capabilities[0].modes} == {
        "DURABLE",
        "INDEXED",
        "LIVE",
    }
    live_capabilities = live_catalog.list_capabilities(
        tenant_id=_TENANT,
        connection_ref=_CONNECTION,
        remote_resource_id=None,
    )
    assert len(live_capabilities) == 3
    assert hasattr(app.state, "lkw_knowledge_inspection_service")
    assert hasattr(app.state, "lkw_knowledge_operations_service")

    discovery = client.get(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/knowledge/connections/{_CONNECTION}/remote-resources",
        headers={"X-Tenant-Id": _TENANT},
        params={"resource_type": "slack_conversation", "limit": 10},
    )
    assert discovery.status_code == 200, discovery.text
    candidate = discovery.json()["items"][0]["opaque_candidate_ref"]
    created = client.post(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/knowledge/indexed-sources",
        headers={
            "X-Tenant-Id": _TENANT,
            "If-Match": "WKC/1",
            "Idempotency-Key": "e2e-create",
        },
        json={
            "connection_ref": _CONNECTION,
            "opaque_candidate_ref": candidate,
            "root_oldest": _OLDEST,
            "root_latest": _LATEST,
        },
    )
    assert created.status_code == 201, created.text
    source_id = created.json()["source_id"]
    sync_accepted = client.post(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/knowledge/indexed-sources/{created.json()['indexed_source_binding_id']}/sync",
        headers={"X-Tenant-Id": _TENANT},
    )
    assert sync_accepted.status_code == 202, sync_accepted.text
    operation_id = sync_accepted.json()["operation_id"]
    worker = runtime.worker
    for _ in range(96):
        worker.drain_once()
        operation = repo.get_operation(tenant_id=_TENANT, operation_id=operation_id)
        if operation is not None and operation.status.value in {"completed", "failed"}:
            break
        time.sleep(0.05)
    completed = _wait_operation(client, operation_id)
    assert completed["status"] == "completed", completed
    assert backend.history_calls == 2, completed
    assert backend.reply_calls == 2, completed
    assert completed["documents_indexed"] >= 1

    wiring_ctx = harness_runtime.env_wiring.tool_wiring.wiring_context
    tenant_stores = wiring_ctx.extras.get("tenant_vectorstore_managers", {})
    scoped_manager = tenant_stores.get(_TENANT) or wiring_ctx.vectorstore_manager
    assert scoped_manager is not None
    assert (
        scoped_manager.count(
            scope=VectorStoreScope(tenant_id=_TENANT, workspace_id=_WORKSPACE)
        )
        >= 1
    )

    refs = repo.list_document_refs(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert len(refs) >= 3
    document_ids = {ref.document_id for ref in refs}
    assert len(document_ids) == len(refs)

    for marker in (_MARKER_ROOT, _MARKER_REPLY, _MARKER_EDIT):
        search = client.post(
            f"{_PREFIX}/workspaces/{_WORKSPACE}/search",
            headers={"X-Tenant-Id": _TENANT},
            json={"query": marker, "limit": 10},
        )
        assert search.status_code == 200, search.text
        results = search.json()["results"]
        assert results
        assert any(marker in (hit.get("snippet") or "") for hit in results)
        assert all(hit.get("source_id") == source_id for hit in results)

    atlas_search = client.post(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/search",
        headers={"X-Tenant-Id": _TENANT},
        json={
            "query": "What caused the Atlas deployment failure and what was decided?",
            "limit": 10,
        },
    )
    assert atlas_search.status_code == 200, atlas_search.text
    atlas_results = atlas_search.json()["results"]
    atlas_snippets = [hit["snippet"] for hit in atlas_results]
    assert any(_MARKER_ROOT in snippet for snippet in atlas_snippets)
    assert any(_ATLAS_INVESTIGATION in snippet for snippet in atlas_snippets)
    assert any(_MARKER_REPLY in snippet for snippet in atlas_snippets)
    assert any(_MARKER_EDIT in snippet for snippet in atlas_snippets)
    assert not any(_UNRELATED_MESSAGE in snippet for snippet in atlas_snippets[:1])
    assert any("Message timestamp:" in snippet for snippet in atlas_snippets)
    assert any("Thread root timestamp:" in snippet for snippet in atlas_snippets)
    assert any("Safe locator: slack://" in snippet for snippet in atlas_snippets)

    other_tenant = client.post(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/search",
        headers={"X-Tenant-Id": "tenant-other"},
        json={"query": _MARKER_ROOT, "limit": 10},
    )
    assert other_tenant.status_code in {403, 404}

    history_before_ask = backend.history_calls
    ask = client.post(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/ask",
        headers={"X-Tenant-Id": _TENANT, "Idempotency-Key": "e2e-ask"},
        json={"question": "What caused the Atlas deployment failure and what was decided?"},
    )
    assert ask.status_code == 200, ask.text
    body = ask.json()
    assert body["citations"]
    _assert_slack_rag_citation(
        citation=body["citations"][0],
        source_id=source_id,
        marker=_MARKER_ROOT,
    )
    assert any(
        _MARKER_ROOT in content
        for message in llm.messages
        for _, content in message
    )
    assert backend.history_calls == history_before_ask
    resolved = wiring.connection_registry.resolve(
        tenant_id=_TENANT,
        connection_ref=_CONNECTION,
        provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
        integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
    )
    assert resolved is integration

    retry_sync = client.post(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/knowledge/indexed-sources/{created.json()['indexed_source_binding_id']}/sync",
        headers={"X-Tenant-Id": _TENANT, "Idempotency-Key": f"retry-{uuid.uuid4().hex}"},
    )
    assert retry_sync.status_code == 202, retry_sync.text
    retry_operation_id = retry_sync.json()["operation_id"]
    for _ in range(48):
        runtime.worker.drain_once()
        operation = repo.get_operation(tenant_id=_TENANT, operation_id=retry_operation_id)
        if operation is not None and operation.status.value in {"completed", "failed"}:
            break
        time.sleep(0.05)
    retry_completed = _wait_operation(client, retry_operation_id)
    assert retry_completed["status"] == "completed", retry_completed
    assert backend.history_calls == 4
    assert backend.reply_calls == 4
    refs_after_retry = repo.list_document_refs(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert len(refs_after_retry) == len(refs)
    repeated_search = client.post(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/search",
        headers={"X-Tenant-Id": _TENANT},
        json={
            "query": "What caused the Atlas deployment failure and what was decided?",
            "limit": 10,
        },
    )
    assert repeated_search.status_code == 200, repeated_search.text
    repeated_ids = sorted(
        hit["document_id"] for hit in repeated_search.json()["results"]
    )
    assert repeated_ids == sorted(hit["document_id"] for hit in atlas_results)

    head = repo.get_knowledge_configuration_head(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    assert head is not None
    disabled = client.delete(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/knowledge/indexed-sources/"
        f"{created.json()['indexed_source_binding_id']}",
        headers={
            "X-Tenant-Id": _TENANT,
            "If-Match": f"WKC/{head.committed_revision}",
            "Idempotency-Key": "e2e-disable",
        },
    )
    assert disabled.status_code == 200, disabled.text
