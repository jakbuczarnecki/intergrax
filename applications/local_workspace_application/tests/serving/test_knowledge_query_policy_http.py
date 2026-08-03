# © Artur Czarnecki. All rights reserved.

"""HTTP tests for workspace Query Policy and knowledge configuration projection."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

from intergrax.fastapi_core.context import RequestContext
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.base import IntegrationCategory
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.serving.knowledge_configuration_http import (
    require_knowledge_configuration_idempotency_key,
)
from local_workspace_application.serving.knowledge_query_policy_routes import (
    mount_knowledge_query_policy_routes,
)
from local_workspace_application.serving.workspace_routes import mount_managed_workspace_routes
from local_workspace_application.workspaces.knowledge_configuration_models import (
    LiveAccessBindingStatusV1,
    LiveResultRetentionV1,
    QueryPolicyModeV1,
    WorkspaceConnectionAttachment,
    WorkspaceConnectionAttachmentStatusV1,
    WorkspaceIndexedSourceBinding,
    WorkspaceIndexedSourceBindingStatusV1,
    WorkspaceKnowledgeConfigurationHead,
    WorkspaceKnowledgeMutationOperationV1,
    WorkspaceLiveAccessBinding,
    WorkspaceQueryPolicy,
)
from local_workspace_application.workspaces.knowledge_configuration_mutation_engine import (
    WorkspaceKnowledgeConfigurationMutationEngine,
)
from local_workspace_application.workspaces.knowledge_configuration_service import (
    WorkspaceKnowledgeConfigurationService,
    WorkspaceKnowledgeConfigurationServiceError,
)
from local_workspace_application.workspaces.knowledge_query_policy_handlers import (
    UpdateQueryPolicyMutationHandler,
)
from local_workspace_application.workspaces.knowledge_query_policy_service import (
    WorkspaceQueryPolicyError,
    WorkspaceQueryPolicyService,
)
from local_workspace_application.workspaces.models import Workspace, WorkspaceStatus
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository
from local_workspace_application.workspaces.service import ManagedWorkspaceService
from local_workspace_application.workspaces.sync_runtime import build_managed_workspace_sync_runtime
from local_workspace_application.workspaces.sync_service import ManagedWorkspaceSyncService

pytestmark = pytest.mark.unit

_PREFIX = "/v1/local_workspace"
_TENANT, _TENANT_B, _WORKSPACE = "tenant-a", "tenant-b", "workspace-1"
_CONN_A, _CAP_A = "conn.a", "cap.read"
_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
_T0 = datetime(2024, 5, 1, 10, 0, 0, tzinfo=UTC)
_T1 = datetime(2024, 5, 2, 10, 0, 0, tzinfo=UTC)
_T2 = datetime(2024, 5, 3, 10, 0, 0, tzinfo=UTC)
_MUTATION = "mutation-1"
_SHA256 = "a" * 64
_FORBIDDEN_SCAN = (
    "credential_ref",
    "mutation_id",
    "idempotency_key",
    "idempotency_key_hash",
    "semantic_identity_hash",
    "stage_manifest_hash",
    "pending_mutation_id",
    "pending_revision",
)


def _workspace(**overrides: object) -> Workspace:
    payload = {
        "workspace_id": _WORKSPACE,
        "tenant_id": _TENANT,
        "name": "Workspace",
        "status": WorkspaceStatus.ACTIVE,
        "created_at": _NOW,
        "updated_at": _NOW,
    }
    payload.update(overrides)
    return Workspace(**payload)


def _indexed_body() -> dict[str, Any]:
    return {"mode": "indexed_only"}


def _live_body() -> dict[str, Any]:
    return {
        "mode": "live_only",
        "allowed_connection_refs": [_CONN_A],
        "allowed_capability_ids": [_CAP_A],
        "max_live_calls": 3,
        "max_total_duration_ms": 60_000,
        "max_result_items": 100,
        "max_result_bytes": 2_097_152,
        "live_result_retention": "receipt_only",
    }


def _headers(
    *,
    tenant: str = _TENANT,
    revision: int = 0,
    idempotency: str = "idem-1",
) -> dict[str, str]:
    return {
        "X-Tenant-Id": tenant,
        "If-Match": f"WKC/{revision}",
        "Idempotency-Key": idempotency,
    }


def _put_path() -> str:
    return f"{_PREFIX}/workspaces/{_WORKSPACE}/query-policy"


def _get_path() -> str:
    return f"{_PREFIX}/workspaces/{_WORKSPACE}/knowledge-configuration"


def _build_stack(
    *,
    workspaces: dict[tuple[str, str], Workspace] | None = None,
    with_context_middleware: bool = False,
) -> tuple[TestClient, ManagedWorkspaceRepository, WorkspaceQueryPolicyService, WorkspaceKnowledgeConfigurationService]:
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    ws_map = {(_TENANT, _WORKSPACE): _workspace()} if workspaces is None else workspaces
    for ws in ws_map.values():
        repo.put_workspace(ws)
    lookup = ManagedWorkspaceService(repo)
    config = WorkspaceKnowledgeConfigurationService(repo, lookup)
    mutation_ids = [f"mutation-{i}" for i in range(1, 30)]
    idx = {"i": 0}

    def _next_id() -> str:
        value = mutation_ids[idx["i"]]
        idx["i"] = min(idx["i"] + 1, len(mutation_ids) - 1)
        return value

    engine = WorkspaceKnowledgeConfigurationMutationEngine(
        repo,
        lookup,
        config,
        {WorkspaceKnowledgeMutationOperationV1.UPDATE_QUERY_POLICY: UpdateQueryPolicyMutationHandler()},
        clock=lambda: _NOW,
        mutation_id_factory=_next_id,
    )
    service = WorkspaceQueryPolicyService(
        repository=repo,
        configuration_service=config,
        mutation_engine=engine,
    )
    app = FastAPI()
    if with_context_middleware:

        class _ContextMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                request.state.context = RequestContext(
                    request_id="test",
                    path=request.url.path,
                    method=request.method,
                    tenant_id=_TENANT,
                    user_id="user",
                    auth=None,
                )
                return await call_next(request)

        app.add_middleware(_ContextMiddleware)
    mount_knowledge_query_policy_routes(
        app,
        query_policy_service=service,
        configuration_service=config,
    )
    return TestClient(app), repo, service, config


def _head_revision(repo: ManagedWorkspaceRepository) -> int:
    head = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    return 0 if head is None else head.committed_revision


def _connection_attachment(**overrides: object) -> WorkspaceConnectionAttachment:
    payload = {
        "attachment_id": "attachment-1",
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "connection_ref": "conn.primary",
        "safe_display_label": "Primary",
        "status": WorkspaceConnectionAttachmentStatusV1.ATTACHED,
        "mutation_id": _MUTATION,
        "effective_revision": 1,
        "created_at": _T0,
        "updated_at": _T0,
    }
    payload.update(overrides)
    return WorkspaceConnectionAttachment(**payload)


def _indexed_source(**overrides: object) -> WorkspaceIndexedSourceBinding:
    payload = {
        "indexed_source_binding_id": "idx-1",
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "knowledge_source_binding_ref": "ksb-1",
        "source_id": "source-1",
        "mutation_id": _MUTATION,
        "effective_revision": 1,
        "semantic_identity_hash": _SHA256,
        "created_at": _T0,
        "updated_at": _T0,
        "cached_safe_display_label": "Docs",
    }
    payload.update(overrides)
    return WorkspaceIndexedSourceBinding(**payload)


def _live_access(**overrides: object) -> WorkspaceLiveAccessBinding:
    payload = {
        "live_access_binding_id": "live-1",
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "connection_ref": "conn.live",
        "allowed_capability_ids": (_CAP_A,),
        "derived_provider_id": "provider-1",
        "derived_integration_kind": IntegrationCategory.WIKI_KNOWLEDGE,
        "derived_safe_display_label": "Wiki",
        "mutation_id": _MUTATION,
        "effective_revision": 1,
        "semantic_identity_hash": _SHA256,
        "created_at": _T0,
        "updated_at": _T0,
        "status": LiveAccessBindingStatusV1.DISABLED,
    }
    payload.update(overrides)
    return WorkspaceLiveAccessBinding(**payload)


def _query_policy(**overrides: object) -> WorkspaceQueryPolicy:
    payload = {
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "mutation_id": _MUTATION,
        "effective_revision": 1,
        "updated_at": _T0,
    }
    payload.update(overrides)
    return WorkspaceQueryPolicy(**payload)


def _seed_full_configuration(repo: ManagedWorkspaceRepository) -> None:
    repo.put_knowledge_configuration_head_if_absent(
        WorkspaceKnowledgeConfigurationHead(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            committed_revision=4,
            updated_at=_NOW,
        )
    )
    repo.put_knowledge_connection_attachment_version_if_absent(
        _connection_attachment(
            attachment_id="att-a",
            connection_ref="conn.b",
            effective_revision=2,
            status=WorkspaceConnectionAttachmentStatusV1.DETACHED,
            updated_at=_T1,
        )
    )
    repo.put_knowledge_connection_attachment_version_if_absent(
        _connection_attachment(
            attachment_id="att-b",
            connection_ref="conn.a",
            effective_revision=4,
            updated_at=_T2,
        )
    )
    repo.put_knowledge_connection_attachment_version_if_absent(
        _connection_attachment(
            attachment_id="att-b",
            connection_ref="conn.a",
            effective_revision=5,
            updated_at=_T2,
        )
    )
    repo.put_knowledge_indexed_source_version_if_absent(
        _indexed_source(indexed_source_binding_id="idx-old", effective_revision=1, updated_at=_T0)
    )
    repo.put_knowledge_indexed_source_version_if_absent(
        _indexed_source(
            indexed_source_binding_id="idx-old",
            effective_revision=3,
            updated_at=_T1,
            status=WorkspaceIndexedSourceBindingStatusV1.DISABLED,
        )
    )
    repo.put_knowledge_indexed_source_version_if_absent(
        _indexed_source(
            indexed_source_binding_id="idx-old",
            effective_revision=5,
            updated_at=_T2,
            status=WorkspaceIndexedSourceBindingStatusV1.DISABLED,
        )
    )
    repo.put_knowledge_live_access_version_if_absent(_live_access(effective_revision=4, updated_at=_T2))
    repo.put_knowledge_live_access_version_if_absent(
        _live_access(live_access_binding_id="live-future", effective_revision=5, updated_at=_T2)
    )
    repo.put_knowledge_query_policy_version_if_absent(
        _query_policy(
            mode=QueryPolicyModeV1.LIVE_ONLY,
            allowed_connection_refs=(_CONN_A,),
            allowed_capability_ids=(_CAP_A,),
            max_live_calls=2,
            live_result_retention=LiveResultRetentionV1.RECEIPT_ONLY,
            effective_revision=4,
            updated_at=_T2,
        )
    )


# --- 5.1 PUT success ---


def test_put_indexed_only_success() -> None:
    client, _, _, _ = _build_stack()
    response = client.put(_put_path(), headers=_headers(), json=_indexed_body())
    assert response.status_code == 200
    body = response.json()
    assert body["workspace_id"] == _WORKSPACE
    assert body["mode"] == "indexed_only"
    assert body["allowed_connection_refs"] == []
    assert body["allowed_capability_ids"] == []
    assert body["max_live_calls"] == 0
    assert body["live_result_retention"] == "ephemeral"
    assert body["effective_revision"] == 1
    assert body["configuration_revision"] == 1
    assert "updated_at" in body


def test_put_live_only_success() -> None:
    client, _, _, _ = _build_stack()
    response = client.put(_put_path(), headers=_headers(), json=_live_body())
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "live_only"
    assert body["allowed_connection_refs"] == [_CONN_A]
    assert body["allowed_capability_ids"] == [_CAP_A]
    assert body["max_live_calls"] == 3
    assert body["live_result_retention"] == "receipt_only"


def test_put_normalizes_allowlists() -> None:
    client, _, _, _ = _build_stack()
    response = client.put(
        _put_path(),
        headers=_headers(),
        json={
            "mode": "live_only",
            "allowed_connection_refs": ["conn.b", "conn.a", "conn.a"],
            "allowed_capability_ids": ["cap.b", "cap.a", "cap.a"],
            "max_live_calls": 1,
            "live_result_retention": "ephemeral",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["allowed_connection_refs"] == ["conn.a", "conn.b"]
    assert body["allowed_capability_ids"] == ["cap.a", "cap.b"]


def test_put_replacement_indexed_to_live_and_back() -> None:
    client, repo, _, _ = _build_stack()
    first = client.put(_put_path(), headers=_headers(), json=_indexed_body())
    rev = first.json()["configuration_revision"]
    live = client.put(
        _put_path(),
        headers=_headers(revision=rev, idempotency="idem-2"),
        json=_live_body(),
    )
    assert live.status_code == 200
    assert live.json()["mode"] == "live_only"
    rev2 = live.json()["configuration_revision"]
    back = client.put(
        _put_path(),
        headers=_headers(revision=rev2, idempotency="idem-3"),
        json=_indexed_body(),
    )
    assert back.status_code == 200
    assert back.json()["mode"] == "indexed_only"
    assert len(repo.list_knowledge_query_policy_versions(tenant_id=_TENANT, workspace_id=_WORKSPACE)) == 3


def test_put_idempotent_replay_and_semantic_noop() -> None:
    client, repo, _, _ = _build_stack()
    first = client.put(_put_path(), headers=_headers(), json=_indexed_body())
    rev = first.json()["configuration_revision"]
    replay = client.put(_put_path(), headers=_headers(revision=rev), json=_indexed_body())
    noop = client.put(
        _put_path(),
        headers=_headers(revision=rev, idempotency="idem-2"),
        json=_indexed_body(),
    )
    assert replay.status_code == 200 and noop.status_code == 200
    assert replay.json()["configuration_revision"] == rev
    assert noop.json()["configuration_revision"] == rev
    assert len(repo.list_knowledge_query_policy_versions(tenant_id=_TENANT, workspace_id=_WORKSPACE)) == 1


# --- 5.2 Headers ---


@pytest.mark.parametrize(
    ("headers", "status_code", "detail"),
    [
        ({"X-Tenant-Id": _TENANT, "Idempotency-Key": "idem-1"}, 428, "knowledge_configuration_if_match_required"),
        ({"X-Tenant-Id": _TENANT, "If-Match": "   ", "Idempotency-Key": "idem-1"}, 428, "knowledge_configuration_if_match_required"),
        ({"X-Tenant-Id": _TENANT, "If-Match": "WKC/0"}, 428, "knowledge_configuration_idempotency_key_required"),
        ({"X-Tenant-Id": _TENANT, "If-Match": "WKC/0", "Idempotency-Key": "   "}, 428, "knowledge_configuration_idempotency_key_required"),
        (_headers(idempotency="bad\x00key"), 400, "knowledge_configuration_idempotency_key_invalid"),
        ({"X-Tenant-Id": _TENANT, "If-Match": "bad", "Idempotency-Key": "idem-1"}, 400, "knowledge_configuration_if_match_invalid"),
    ],
)
def test_put_header_validation(headers, status_code, detail) -> None:
    client, _, _, _ = _build_stack()
    response = client.put(_put_path(), headers=headers, json=_indexed_body())
    assert response.status_code == status_code
    assert response.json()["detail"] == detail


def test_idempotency_control_chars_rejected_by_helper() -> None:
    with pytest.raises(HTTPException) as exc:
        require_knowledge_configuration_idempotency_key("bad\x00key")
    assert exc.value.status_code == 400
    assert exc.value.detail == "knowledge_configuration_idempotency_key_invalid"


def test_raw_idempotency_key_not_in_response() -> None:
    client, _, _, _ = _build_stack()
    raw_key = "super-secret-idem"
    response = client.put(
        _put_path(),
        headers=_headers(idempotency=raw_key),
        json=_indexed_body(),
    )
    assert response.status_code == 200
    assert raw_key not in response.text
    assert hashlib.sha256(raw_key.encode()).hexdigest() not in response.text


# --- 5.3 Request validation ---


@pytest.mark.parametrize("mode", ["hybrid", "automatic"])
def test_unsupported_mode_returns_400(mode: str) -> None:
    client, _, _, _ = _build_stack()
    response = client.put(_put_path(), headers=_headers(), json={"mode": mode})
    assert response.status_code == 400
    assert response.json()["detail"] == "query_policy_mode_unsupported"


def test_invalid_retention_returns_400() -> None:
    client, _, _, _ = _build_stack()
    response = client.put(
        _put_path(),
        headers=_headers(),
        json={"mode": "indexed_only", "live_result_retention": "forever"},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "query_policy_invalid"


@pytest.mark.parametrize(
    "body",
    [
        {"mode": "indexed_only", "allowed_connection_refs": ["conn.a"]},
        {"mode": "indexed_only", "allowed_capability_ids": ["cap.a"]},
        {"mode": "indexed_only", "max_live_calls": 1},
        {"mode": "indexed_only", "live_result_retention": "receipt_only"},
        {"mode": "live_only", "allowed_connection_refs": [], "allowed_capability_ids": [_CAP_A], "max_live_calls": 1},
        {"mode": "live_only", "allowed_connection_refs": [_CONN_A], "allowed_capability_ids": [], "max_live_calls": 1},
        {"mode": "live_only", "allowed_connection_refs": [_CONN_A], "allowed_capability_ids": [_CAP_A], "max_live_calls": 0},
        {"mode": "indexed_only", "max_result_items": 501},
        {"mode": "indexed_only", "max_result_bytes": 0},
    ],
)
def test_invalid_policy_shape_returns_400_and_leaves_state(body: dict[str, Any]) -> None:
    client, repo, _, _ = _build_stack()
    before_head = _head_revision(repo)
    before_versions = len(repo.list_knowledge_query_policy_versions(tenant_id=_TENANT, workspace_id=_WORKSPACE))
    before_mutations = len(repo.list_knowledge_configuration_mutations(tenant_id=_TENANT, workspace_id=_WORKSPACE))
    response = client.put(_put_path(), headers=_headers(), json=body)
    assert response.status_code == 400
    assert response.json()["detail"] == "query_policy_invalid"
    assert _head_revision(repo) == before_head
    assert len(repo.list_knowledge_query_policy_versions(tenant_id=_TENANT, workspace_id=_WORKSPACE)) == before_versions
    assert len(repo.list_knowledge_configuration_mutations(tenant_id=_TENANT, workspace_id=_WORKSPACE)) == before_mutations


def test_extra_body_idempotency_key_returns_422() -> None:
    client, _, _, _ = _build_stack()
    response = client.put(
        _put_path(),
        headers=_headers(),
        json={**_indexed_body(), "idempotency_key": "evil"},
    )
    assert response.status_code == 422


# --- 5.4 Concurrency and idempotency ---


def test_revision_conflict_returns_409() -> None:
    client, _, _, _ = _build_stack()
    client.put(_put_path(), headers=_headers(), json=_indexed_body())
    conflict = client.put(
        _put_path(),
        headers=_headers(revision=0, idempotency="idem-2"),
        json=_live_body(),
    )
    assert conflict.status_code == 409
    assert conflict.json()["detail"] == "configuration_revision_conflict"


def test_idempotency_conflict_returns_409() -> None:
    client, _, _, _ = _build_stack()
    first = client.put(_put_path(), headers=_headers(), json=_indexed_body())
    rev = first.json()["configuration_revision"]
    conflict = client.put(
        _put_path(),
        headers=_headers(revision=rev),
        json=_live_body(),
    )
    assert conflict.status_code == 409
    assert conflict.json()["detail"] == "configuration_idempotency_conflict"


def test_idempotency_conflict_before_stale_if_match() -> None:
    client, _, _, _ = _build_stack()
    client.put(_put_path(), headers=_headers(), json=_indexed_body())
    conflict = client.put(
        _put_path(),
        headers=_headers(revision=0),
        json=_live_body(),
    )
    assert conflict.status_code == 409
    assert conflict.json()["detail"] == "configuration_idempotency_conflict"


def test_committed_replay_returns_historical_policy() -> None:
    client, _, _, _ = _build_stack()
    first = client.put(_put_path(), headers=_headers(idempotency="idem-a"), json=_indexed_body())
    rev = first.json()["configuration_revision"]
    client.put(
        _put_path(),
        headers=_headers(revision=rev, idempotency="idem-b"),
        json=_live_body(),
    )
    replay = client.put(
        _put_path(),
        headers=_headers(revision=2, idempotency="idem-a"),
        json=_indexed_body(),
    )
    assert replay.status_code == 200
    body = replay.json()
    assert body["mode"] == "indexed_only"
    assert body["effective_revision"] == 1
    assert body["configuration_revision"] == 1


# --- 5.5 Tenant isolation ---


def test_unknown_workspace_returns_404() -> None:
    client, _, _, _ = _build_stack(workspaces={})
    response = client.put(_put_path(), headers=_headers(), json=_indexed_body())
    assert response.status_code == 404
    assert response.json()["detail"] == "workspace_not_found"


def test_cross_tenant_workspace_returns_404() -> None:
    client, _, _, _ = _build_stack(
        workspaces={(_TENANT_B, _WORKSPACE): _workspace(tenant_id=_TENANT_B)}
    )
    response = client.put(_put_path(), headers=_headers(), json=_indexed_body())
    assert response.status_code == 404
    assert response.json()["detail"] == "workspace_not_found"
    assert _TENANT_B not in response.text


def test_auth_context_precedence_over_spoofed_header() -> None:
    client, _, _, _ = _build_stack(with_context_middleware=True)
    response = client.put(
        _put_path(),
        headers=_headers(tenant=_TENANT_B),
        json=_indexed_body(),
    )
    assert response.status_code == 200
    assert response.json()["workspace_id"] == _WORKSPACE


def test_get_cross_tenant_returns_404() -> None:
    client, _, _, _ = _build_stack(
        workspaces={(_TENANT_B, _WORKSPACE): _workspace(tenant_id=_TENANT_B)}
    )
    response = client.get(_get_path(), headers={"X-Tenant-Id": _TENANT})
    assert response.status_code == 404
    assert response.json()["detail"] == "workspace_not_found"


# --- 5.6 Complete GET projection ---


def test_get_full_configuration_projection() -> None:
    client, repo, _, _ = _build_stack()
    _seed_full_configuration(repo)
    response = client.get(_get_path(), headers={"X-Tenant-Id": _TENANT})
    assert response.status_code == 200
    body = response.json()
    assert body["tenant_id"] == _TENANT
    assert body["workspace_id"] == _WORKSPACE
    assert body["configuration_revision"] == 4
    assert body["updated_at"] == _T2.isoformat().replace("+00:00", "Z")
    assert len(body["connection_attachments"]) == 2
    assert [item["connection_ref"] for item in body["connection_attachments"]] == ["conn.a", "conn.b"]
    assert body["connection_attachments"][0]["effective_revision"] == 4
    assert body["connection_attachments"][1]["status"] == "detached"
    assert len(body["indexed_sources"]) == 1
    assert body["indexed_sources"][0]["effective_revision"] == 3
    assert body["indexed_sources"][0]["status"] == "disabled"
    assert body["indexed_sources"][0]["cached_safe_display_label"] == "Docs"
    assert len(body["live_access_bindings"]) == 1
    assert body["live_access_bindings"][0]["effective_revision"] == 4
    assert body["live_access_bindings"][0]["status"] == "disabled"
    assert body["query_policy"]["mode"] == "live_only"
    assert body["query_policy"]["effective_revision"] == 4
    serialized = json.dumps(body)
    for token in _FORBIDDEN_SCAN:
        assert token not in serialized


def test_get_query_policy_null_before_first_update() -> None:
    client, repo, _, _ = _build_stack()
    repo.put_knowledge_configuration_head_if_absent(
        WorkspaceKnowledgeConfigurationHead(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            committed_revision=0,
            updated_at=_NOW,
        )
    )
    before = len(repo.list_knowledge_query_policy_versions(tenant_id=_TENANT, workspace_id=_WORKSPACE))
    response = client.get(_get_path(), headers={"X-Tenant-Id": _TENANT})
    after = len(repo.list_knowledge_query_policy_versions(tenant_id=_TENANT, workspace_id=_WORKSPACE))
    assert response.status_code == 200
    assert response.json()["query_policy"] is None
    assert before == after == 0


# --- 5.7 Projection failures ---


def test_get_unknown_workspace_returns_404() -> None:
    client, _, _, _ = _build_stack(workspaces={})
    response = client.get(_get_path(), headers={"X-Tenant-Id": _TENANT})
    assert response.status_code == 404
    assert response.json()["detail"] == "workspace_not_found"


def test_get_configuration_projection_unstable_returns_503() -> None:
    client, _, _, config = _build_stack()
    with patch.object(
        config,
        "get_configuration",
        side_effect=WorkspaceKnowledgeConfigurationServiceError("configuration_projection_unstable"),
    ):
        response = client.get(_get_path(), headers={"X-Tenant-Id": _TENANT})
    assert response.status_code == 503
    assert response.json()["detail"] == "configuration_projection_unstable"
    assert response.json().keys() == {"detail"}


def test_get_configuration_projection_invalid_returns_503() -> None:
    client, _, _, config = _build_stack()
    with patch.object(config, "get_configuration", side_effect=ValueError("revision_zero_forbids_children")):
        response = client.get(_get_path(), headers={"X-Tenant-Id": _TENANT})
    assert response.status_code == 503
    assert response.json()["detail"] == "configuration_projection_invalid"


def test_put_query_policy_projection_incomplete_returns_503() -> None:
    client, _, service, _ = _build_stack()
    with patch.object(
        service,
        "update_query_policy",
        side_effect=WorkspaceQueryPolicyError("query_policy_projection_incomplete"),
    ):
        response = client.put(_put_path(), headers=_headers(), json=_indexed_body())
    assert response.status_code == 503
    assert response.json()["detail"] == "query_policy_projection_incomplete"
    assert response.json().keys() == {"detail"}


# --- 5.8 Host wiring ---


class _FakeExecutor:
    async def execute(self, task: object) -> object:
        _ = task
        return type("R", (), {"metadata": {}})()


def test_host_wiring_mounts_routes_without_provider_dependencies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    data_home = tmp_path / "data"
    data_home.mkdir()
    monkeypatch.setenv("DATA_HOME", str(data_home))
    settings = LocalWorkspaceBackendSettings.from_env()
    app = FastAPI()
    executor = _FakeExecutor()
    sync = ManagedWorkspaceSyncService(repo, executor)  # type: ignore[arg-type]
    runtime = build_managed_workspace_sync_runtime(
        document_store=store,
        sync_service=sync,
        repository=repo,
    )
    mount_managed_workspace_routes(
        app,
        task_executor=executor,  # type: ignore[arg-type]
        settings=settings,
        repository=repo,
        sync_runtime=runtime,
        object_storage=None,
    )
    repo.put_workspace(_workspace())
    with TestClient(app) as client:
        put = client.put(
            _put_path(),
            headers=_headers(),
            json=_indexed_body(),
        )
        assert put.status_code == 200
        get = client.get(_get_path(), headers={"X-Tenant-Id": _TENANT})
        assert get.status_code == 200
        assert get.json()["query_policy"]["mode"] == "indexed_only"
