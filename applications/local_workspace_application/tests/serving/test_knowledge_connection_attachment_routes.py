# © Artur Czarnecki. All rights reserved.

"""HTTP tests for workspace connection attachment routes."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.vendor_knowledge.tenant_connections import (
    SafeTenantConnectionV1,
    TenantConnectionAdministrativeStatus,
)
from local_workspace_application.serving.knowledge_connection_attachment_routes import (
    mount_knowledge_connection_attachment_routes,
)
from local_workspace_application.workspaces.knowledge_configuration_handlers import (
    AttachConnectionMutationHandler,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    WorkspaceKnowledgeConfigurationHead,
    WorkspaceKnowledgeMutationOperationV1,
)
from local_workspace_application.workspaces.knowledge_configuration_mutation_engine import (
    WorkspaceKnowledgeConfigurationMutationEngine,
)
from local_workspace_application.workspaces.knowledge_configuration_service import (
    WorkspaceKnowledgeConfigurationService,
)
from local_workspace_application.workspaces.knowledge_connection_attachment_service import (
    WorkspaceConnectionAttachmentService,
)
from local_workspace_application.workspaces.models import Workspace, WorkspaceStatus
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository

pytestmark = pytest.mark.unit

_PREFIX = "/v1/local_workspace"
_TENANT, _TENANT_B, _WORKSPACE, _CONNECTION = "tenant-a", "tenant-b", "workspace-1", "conn.primary"
_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
_IDEM_HASH = hashlib.sha256(b"idem-1").hexdigest()


class _FakeConnectionPort:
    def __init__(self, connections: dict[tuple[str, str], SafeTenantConnectionV1]) -> None:
        self._connections = connections

    def get_connection(self, *, tenant_id: str, connection_ref: str) -> SafeTenantConnectionV1 | None:
        return self._connections.get((tenant_id.strip(), connection_ref.strip()))

    def list_connections(self, *, tenant_id: str, limit: int = 100, administrative_status=None):
        return tuple(v for (t, _), v in self._connections.items() if t == tenant_id)


def _safe_connection(**overrides: object) -> SafeTenantConnectionV1:
    payload = {
        "connection_ref": _CONNECTION,
        "tenant_id": _TENANT,
        "provider_id": "provider.slack",
        "integration_kind": IntegrationCategory.CONVERSATION_CHANNEL,
        "safe_display_name": "Primary Connection",
        "administrative_status": TenantConnectionAdministrativeStatus.ACTIVE,
        "configuration_version": 1,
        "connected_principal_ref": None,
        "created_at": _NOW,
        "updated_at": _NOW,
    }
    payload.update(overrides)
    return SafeTenantConnectionV1(**payload)


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


class _FakeWorkspaceLookup:
    def __init__(self, workspaces: dict[tuple[str, str], Workspace]) -> None:
        self._workspaces = workspaces

    def require_workspace(self, *, tenant_id: str, workspace_id: str) -> Workspace | None:
        return self._workspaces.get((tenant_id, workspace_id))


def _build_client(
    *,
    workspaces: dict[tuple[str, str], Workspace] | None = None,
    connections: dict[tuple[str, str], SafeTenantConnectionV1] | None = None,
    mount_routes: bool = True,
) -> tuple[TestClient, ManagedWorkspaceRepository, _FakeConnectionPort]:
    repo = ManagedWorkspaceRepository(InMemoryDocumentStore())
    workspace_map = workspaces if workspaces is not None else {(_TENANT, _WORKSPACE): _workspace()}
    for workspace in workspace_map.values():
        repo.put_workspace(workspace)
    lookup = _FakeWorkspaceLookup(workspace_map)
    config_service = WorkspaceKnowledgeConfigurationService(repo, lookup)
    engine = WorkspaceKnowledgeConfigurationMutationEngine(
        repo,
        lookup,
        config_service,
        {WorkspaceKnowledgeMutationOperationV1.ATTACH_CONNECTION: AttachConnectionMutationHandler()},
        clock=lambda: _NOW,
        mutation_id_factory=lambda: "mutation-1",
    )
    port = _FakeConnectionPort(connections or {(_TENANT, _CONNECTION): _safe_connection()})
    app = FastAPI()
    if mount_routes:
        mount_knowledge_connection_attachment_routes(
            app,
            attachment_service=WorkspaceConnectionAttachmentService(
                connection_port=port,
                configuration_service=config_service,
                mutation_engine=engine,
            ),
        )
    return TestClient(app), repo, port


def _headers(*, if_match: str | None = "WKC/0", idempotency: str = "idem-1") -> dict[str, str]:
    headers = {"X-Tenant-Id": _TENANT, "Idempotency-Key": idempotency}
    if if_match is not None:
        headers["If-Match"] = if_match
    return headers


def _path(connection_ref: str = _CONNECTION) -> str:
    return f"{_PREFIX}/workspaces/{_WORKSPACE}/connections/{connection_ref}"


@pytest.fixture
def client_bundle():
    client, repo, port = _build_client()
    with client:
        yield client, repo, port


def test_successful_attach_returns_201(client_bundle) -> None:
    client, repo, _ = client_bundle
    response = client.put(_path(), headers=_headers())
    assert response.status_code == 201
    payload = response.json()
    assert payload["connection_ref"] == _CONNECTION
    assert payload["configuration_revision"] == 1
    assert "credential_ref" not in payload and "mutation_id" not in payload
    mutations = repo.list_knowledge_configuration_mutations(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert mutations[0].idempotency_key_hash == _IDEM_HASH
    assert "idem-1" not in json.dumps(mutations[0].model_dump(mode="json"))


def test_committed_replay_and_semantic_no_op(client_bundle) -> None:
    client, _, _ = client_bundle
    first = client.put(_path(), headers=_headers())
    replay = client.put(_path(), headers=_headers(if_match="WKC/0"))
    noop = client.put(_path(), headers=_headers(if_match="WKC/1", idempotency="idem-2"))
    assert first.status_code == 201 and replay.status_code == 200 and noop.status_code == 200


@pytest.mark.parametrize(
    ("headers", "status_code", "detail"),
    [
        ({"X-Tenant-Id": _TENANT, "Idempotency-Key": "idem-1"}, 428, "knowledge_configuration_if_match_required"),
        ({"X-Tenant-Id": _TENANT, "If-Match": "WKC/0"}, 428, "knowledge_configuration_idempotency_key_required"),
        (_headers(if_match='"WKC/0"'), 400, "knowledge_configuration_if_match_invalid"),
        (_headers(idempotency="bad\x00key"), 400, "knowledge_configuration_idempotency_key_invalid"),
        (_headers(idempotency=""), 428, "knowledge_configuration_idempotency_key_required"),
    ],
)
def test_header_validation(headers, status_code, detail) -> None:
    client, _, _ = _build_client()
    with client:
        response = client.put(_path(), headers=headers)
    assert response.status_code == status_code
    assert response.json()["detail"] == detail


def test_extra_request_body_field_returns_422(client_bundle) -> None:
    client, _, _ = client_bundle
    response = client.put(_path(), headers=_headers(), json={"tenant_id": "evil"})
    assert response.status_code == 422


@pytest.mark.parametrize(
    "workspaces",
    [
        {},
        {(_TENANT_B, _WORKSPACE): _workspace(tenant_id=_TENANT_B)},
    ],
)
def test_workspace_not_found(workspaces) -> None:
    client, _, _ = _build_client(workspaces=workspaces)
    with client:
        response = client.put(_path(), headers=_headers())
    assert response.status_code == 404
    assert response.json()["detail"] == "workspace_not_found"
    assert _CONNECTION not in response.text


def test_connection_missing_returns_404(client_bundle) -> None:
    client, _, port = client_bundle
    port._connections.clear()
    response = client.put(_path(), headers=_headers())
    assert response.status_code == 404 and response.json()["detail"] == "connection_not_found"


@pytest.mark.parametrize(
    "status",
    [TenantConnectionAdministrativeStatus.DISABLED, TenantConnectionAdministrativeStatus.REVOKED],
)
def test_unavailable_connection_returns_503(client_bundle, status) -> None:
    client, _, port = client_bundle
    port._connections[(_TENANT, _CONNECTION)] = _safe_connection(administrative_status=status)
    response = client.put(_path(), headers=_headers())
    assert response.status_code == 503 and response.json()["detail"] == "connection_unavailable"


def test_revision_and_idempotency_conflicts(client_bundle) -> None:
    client, _, port = client_bundle
    port._connections[(_TENANT, "conn.other")] = _safe_connection(connection_ref="conn.other")
    client.put(_path(), headers=_headers())
    revision = client.put(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/connections/conn.other",
        headers=_headers(if_match="WKC/0", idempotency="idem-2"),
    )
    idem = client.put(_path(), headers=_headers(if_match="WKC/0"), json={"safe_display_label": "Alias"})
    assert revision.status_code == 409 and revision.json()["detail"] == "configuration_revision_conflict"
    assert idem.status_code == 409 and idem.json()["detail"] == "configuration_idempotency_conflict"


def test_recovery_required_mapping(client_bundle) -> None:
    client, repo, _ = client_bundle
    repo.put_knowledge_configuration_head_if_absent(
        WorkspaceKnowledgeConfigurationHead(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            committed_revision=0,
            pending_revision=1,
            pending_mutation_id="pending",
            updated_at=_NOW,
        )
    )
    response = client.put(_path(), headers=_headers(idempotency="other"))
    assert response.status_code == 503 and response.json()["detail"] == "configuration_recovery_required"


def test_route_not_mounted_without_connection_port() -> None:
    client, _, _ = _build_client(mount_routes=False)
    with client:
        response = client.put(_path(), headers=_headers())
    assert response.status_code == 404
