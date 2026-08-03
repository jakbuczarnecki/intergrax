# © Artur Czarnecki. All rights reserved.

"""HTTP tests for workspace Live Access Binding routes."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.vendor_knowledge.remote_resource_discovery import (
    RemoteResourceAvailabilityV1,
    RemoteResourceDescriptorV1,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_capabilities import (
    CapabilityEffectV1,
    LiveCapabilityDescriptorV1,
)
from intergrax.runtime.vendor_knowledge.tenant_connections import (
    SafeTenantConnectionV1,
    TenantConnectionAdministrativeStatus,
)
from local_workspace_application.serving.knowledge_configuration_http import (
    hash_knowledge_configuration_idempotency_key,
    require_knowledge_configuration_idempotency_key,
)
from local_workspace_application.serving.knowledge_live_access_routes import (
    mount_knowledge_live_access_routes,
)
from local_workspace_application.workspaces.knowledge_configuration_handlers import (
    AttachConnectionMutationHandler,
)
from local_workspace_application.workspaces.knowledge_live_access_handlers import (
    CreateLiveAccessBindingMutationHandler,
    DisableLiveAccessBindingMutationHandler,
)
from local_workspace_application.workspaces.knowledge_configuration_hashing import (
    live_access_binding_id_from_semantic_hash,
    semantic_identity_hash_for_live_access_binding,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
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
from local_workspace_application.workspaces.knowledge_live_access_service import (
    WorkspaceLiveAccessBindingService,
)
from local_workspace_application.workspaces.models import Workspace, WorkspaceStatus
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository

pytestmark = pytest.mark.unit

_PREFIX = "/v1/local_workspace"
_TENANT, _WORKSPACE, _CONNECTION = "tenant-a", "workspace-1", "conn.primary"
_CAP = "cap.read"
_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
_IDEM_HASH = hashlib.sha256(b"idem-1").hexdigest()
_SEMANTIC = semantic_identity_hash_for_live_access_binding(
    tenant_id=_TENANT,
    workspace_id=_WORKSPACE,
    connection_ref=_CONNECTION,
    normalized_remote_resource_id=None,
    normalized_capability_set=(_CAP,),
)
_BINDING_ID = live_access_binding_id_from_semantic_hash(_SEMANTIC)


class _FakeConnectionPort:
    def get_connection(self, *, tenant_id: str, connection_ref: str) -> SafeTenantConnectionV1 | None:
        if tenant_id == _TENANT and connection_ref == _CONNECTION:
            return SafeTenantConnectionV1(
                connection_ref=_CONNECTION,
                tenant_id=_TENANT,
                provider_id="provider.slack",
                integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
                safe_display_name="Primary Connection",
                administrative_status=TenantConnectionAdministrativeStatus.ACTIVE,
                configuration_version=1,
                connected_principal_ref=None,
                created_at=_NOW,
                updated_at=_NOW,
            )
        return None

    def list_connections(self, *, tenant_id: str, limit: int = 100, administrative_status=None):
        conn = self.get_connection(tenant_id=tenant_id, connection_ref=_CONNECTION)
        return (conn,) if conn is not None else ()


class _FakeCatalog:
    def list_capabilities(self, *, tenant_id: str, connection_ref: str, remote_resource_id: str | None):
        return (
            LiveCapabilityDescriptorV1(
                capability_id=_CAP,
                provider_id="provider.slack",
                integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
                effect=CapabilityEffectV1.READ,
                read_only=True,
                resource_scope_required=False,
                request_schema_ref="schema://req",
                result_schema_ref="schema://res",
                available=True,
            ),
        )


class _FakeLookup:
    async def get_remote_resource(self, *, tenant_id: str, connection_ref: str, remote_resource_id: str):
        return RemoteResourceDescriptorV1(
            remote_resource_id=remote_resource_id,
            resource_type="slack_conversation",
            safe_display_label="General",
            availability=RemoteResourceAvailabilityV1.AVAILABLE,
            supported_capability_ids=(_CAP,),
            connection_ref=connection_ref,
            provider_id="provider.slack",
            integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
            source_kind="slack_conversation",
            discovered_at=_NOW,
            snapshot_version="snap-1",
        )


def _workspace() -> Workspace:
    return Workspace(
        workspace_id=_WORKSPACE,
        tenant_id=_TENANT,
        name="Workspace",
        status=WorkspaceStatus.ACTIVE,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _build_client() -> tuple[TestClient, ManagedWorkspaceRepository]:
    repo = ManagedWorkspaceRepository(InMemoryDocumentStore())
    repo.put_workspace(_workspace())
    lookup = type("Lookup", (), {"require_workspace": lambda self, tenant_id, workspace_id: _workspace()})()
    config = WorkspaceKnowledgeConfigurationService(repo, lookup)
    mutation_ids = [f"mutation-{i}" for i in range(1, 8)]
    idx = {"i": 0}

    def _next_id() -> str:
        value = mutation_ids[idx["i"]]
        idx["i"] = min(idx["i"] + 1, len(mutation_ids) - 1)
        return value

    engine = WorkspaceKnowledgeConfigurationMutationEngine(
        repo,
        lookup,
        config,
        {
            WorkspaceKnowledgeMutationOperationV1.ATTACH_CONNECTION: AttachConnectionMutationHandler(),
            WorkspaceKnowledgeMutationOperationV1.CREATE_LIVE_ACCESS_BINDING: CreateLiveAccessBindingMutationHandler(),
            WorkspaceKnowledgeMutationOperationV1.DISABLE_LIVE_ACCESS_BINDING: DisableLiveAccessBindingMutationHandler(),
        },
        clock=lambda: _NOW,
        mutation_id_factory=_next_id,
    )
    port = _FakeConnectionPort()
    service = WorkspaceLiveAccessBindingService(
        repository=repo,
        configuration_service=config,
        mutation_engine=engine,
        tenant_connection_port=port,
        capability_catalog=_FakeCatalog(),
        remote_resource_lookup_port=_FakeLookup(),
    )
    attachment = WorkspaceConnectionAttachmentService(
        connection_port=port,
        configuration_service=config,
        mutation_engine=engine,
    )
    app = FastAPI()
    mount_knowledge_live_access_routes(app, live_access_service=service, repository=repo)
    return TestClient(app), repo, attachment


def _headers(*, revision: int = 0, idem: str = "idem-1") -> dict[str, str]:
    return {
        "X-Tenant-Id": _TENANT,
        "If-Match": f"WKC/{revision}",
        "Idempotency-Key": idem,
    }


def _attach(attachment, *, rev: int = 0) -> int:
    from local_workspace_application.workspaces.knowledge_connection_attachment_service import (
        AttachWorkspaceConnectionCommand,
    )

    return attachment.attach_connection(
        AttachWorkspaceConnectionCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            connection_ref=_CONNECTION,
            expected_revision=rev,
            idempotency_key_hash=hashlib.sha256(b"attach").hexdigest(),
        )
    ).configuration_revision


def test_create_returns_201_and_safe_payload() -> None:
    client, _, attachment = _build_client()
    rev = _attach(attachment)
    response = client.post(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/knowledge/live-access-bindings",
        headers=_headers(revision=rev),
        json={"connection_ref": _CONNECTION, "allowed_capability_ids": [_CAP]},
    )
    assert response.status_code == 201
    payload = response.json()
    assert payload["live_access_binding_id"] == _BINDING_ID
    assert payload["derived_provider_id"] == "provider.slack"
    assert payload["configuration_revision"] == rev + 1
    assert "credential" not in json.dumps(payload).lower()
    assert "idem-1" not in json.dumps(payload)


def test_create_noop_returns_200() -> None:
    client, _, attachment = _build_client()
    rev = _attach(attachment)
    client.post(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/knowledge/live-access-bindings",
        headers=_headers(revision=rev),
        json={"connection_ref": _CONNECTION, "allowed_capability_ids": [_CAP]},
    )
    response = client.post(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/knowledge/live-access-bindings",
        headers=_headers(revision=rev + 1, idem="idem-2"),
        json={"connection_ref": _CONNECTION, "allowed_capability_ids": [_CAP]},
    )
    assert response.status_code == 200


def test_disable_returns_200() -> None:
    client, _, attachment = _build_client()
    rev = _attach(attachment)
    client.post(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/knowledge/live-access-bindings",
        headers=_headers(revision=rev),
        json={"connection_ref": _CONNECTION, "allowed_capability_ids": [_CAP]},
    )
    response = client.delete(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/knowledge/live-access-bindings/{_BINDING_ID}",
        headers=_headers(revision=rev + 1, idem="disable-1"),
    )
    assert response.status_code == 200
    assert response.json()["status"] == "disabled"


def test_missing_if_match_returns_428() -> None:
    client, _, _ = _build_client()
    response = client.post(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/knowledge/live-access-bindings",
        headers={"X-Tenant-Id": _TENANT, "Idempotency-Key": "idem-1"},
        json={"connection_ref": _CONNECTION, "allowed_capability_ids": [_CAP]},
    )
    assert response.status_code == 428


def test_missing_idempotency_key_returns_428() -> None:
    client, _, _ = _build_client()
    response = client.post(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/knowledge/live-access-bindings",
        headers={"X-Tenant-Id": _TENANT, "If-Match": "WKC/0"},
        json={"connection_ref": _CONNECTION, "allowed_capability_ids": [_CAP]},
    )
    assert response.status_code == 428


def test_invalid_if_match_returns_400() -> None:
    client, _, _ = _build_client()
    response = client.post(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/knowledge/live-access-bindings",
        headers={"X-Tenant-Id": _TENANT, "If-Match": "bad", "Idempotency-Key": "idem-1"},
        json={"connection_ref": _CONNECTION, "allowed_capability_ids": [_CAP]},
    )
    assert response.status_code == 400


def test_body_idempotency_key_rejected() -> None:
    client, _, attachment = _build_client()
    rev = _attach(attachment)
    response = client.post(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/knowledge/live-access-bindings",
        headers=_headers(revision=rev),
        json={
            "connection_ref": _CONNECTION,
            "allowed_capability_ids": [_CAP],
            "idempotency_key": "forbidden",
        },
    )
    assert response.status_code == 422


def test_cross_tenant_returns_404() -> None:
    client, _, attachment = _build_client()
    rev = _attach(attachment)
    response = client.post(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/knowledge/live-access-bindings",
        headers={
            "X-Tenant-Id": "tenant-other",
            "If-Match": f"WKC/{rev}",
            "Idempotency-Key": "idem-1",
        },
        json={"connection_ref": _CONNECTION, "allowed_capability_ids": [_CAP]},
    )
    assert response.status_code == 404


def test_capability_validation_error_returns_400() -> None:
    client, _, attachment = _build_client()
    rev = _attach(attachment)
    response = client.post(
        f"{_PREFIX}/workspaces/{_WORKSPACE}/knowledge/live-access-bindings",
        headers=_headers(revision=rev),
        json={"connection_ref": _CONNECTION, "allowed_capability_ids": ["missing"]},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "capability_not_found"


@pytest.mark.parametrize(
    "bad_key",
    ["\x00bad", "bad\x1fkey", "bad\x7fkey"],
)
def test_idempotency_key_control_characters_rejected(bad_key: str) -> None:
    with pytest.raises(HTTPException) as exc_info:
        require_knowledge_configuration_idempotency_key(bad_key)
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "knowledge_configuration_idempotency_key_invalid"
    assert bad_key not in str(exc_info.value.detail)


def test_idempotency_key_opaque_accepted_and_hashed() -> None:
    key = "opaque-stable-key"
    assert require_knowledge_configuration_idempotency_key(key) == key
    digest = hash_knowledge_configuration_idempotency_key(key)
    assert digest != key
    assert key not in digest
