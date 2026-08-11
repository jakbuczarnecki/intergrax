# © Artur Czarnecki. All rights reserved.

"""Tests for derived workspace setup snapshot (LKW-PRODUCT-3C)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from dataclasses import replace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from local_workspace_application.host.readiness import LocalWorkspaceReadinessSnapshot
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.serving.workspace_routes import mount_managed_workspace_routes
from local_workspace_application.workspaces.knowledge_inspection_operations_service import (
    KnowledgeAccessModeV1,
    KnowledgeInventoryItemV1,
    KnowledgeInventorySummaryV1,
    KnowledgeInventoryV1,
    KnowledgeOperationV1,
    KnowledgeRevisionKindV1,
    indexed_knowledge_item_id,
    live_knowledge_item_id,
)
from local_workspace_application.workspaces.models import (
    WorkspaceOperation,
    WorkspaceOperationStatus,
    WorkspaceOperationType,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository
from local_workspace_application.workspaces.sync_runtime import build_managed_workspace_sync_runtime
from local_workspace_application.workspaces.sync_service import ManagedWorkspaceSyncService
from local_workspace_application.workspaces.workspace_setup_snapshot_service import (
    SetupNextActionV1,
    SetupPhaseV1,
    WorkspaceSetupSnapshotService,
)

pytestmark = pytest.mark.unit

_PREFIX = "/v1/local_workspace"
_NOW = datetime(2026, 8, 8, 10, 0, tzinfo=UTC)
_TENANT = "tenant-a"
_TENANT_B = "tenant-b"
_CONFIRM_SECRET = "confirm-secret"


class _FakeExecutor:
    async def execute(self, task: object) -> object:
        _ = task
        return type("R", (), {"metadata": {}})()


@dataclass(frozen=True)
class _ReadinessStub:
    ready: bool = True
    accepts_new_work: bool = True

    def readiness_snapshot(self) -> LocalWorkspaceReadinessSnapshot:
        return LocalWorkspaceReadinessSnapshot(
            ready=self.ready,
            accepts_new_work=self.accepts_new_work,
            state="ready" if self.accepts_new_work else "degraded",
            detail="",
            rejection_error_id="",
        )


def _headers(tenant: str = _TENANT) -> dict[str, str]:
    return {"X-Tenant-Id": tenant}


def _inventory(*, items: tuple[KnowledgeInventoryItemV1, ...]) -> KnowledgeInventoryV1:
    return KnowledgeInventoryV1(
        tenant_id=_TENANT,
        workspace_id="ws-1",
        items=items,
        summary=KnowledgeInventorySummaryV1(
            total=len(items),
            indexed=sum(item.mode is KnowledgeAccessModeV1.INDEXED for item in items),
            live=sum(item.mode is KnowledgeAccessModeV1.LIVE for item in items),
            active=sum(item.lifecycle_state == "active" for item in items),
            disabled=sum(item.lifecycle_state == "disabled" for item in items),
            attention_required=sum(
                item.lifecycle_state in {"error", "detach_blocked"}
                or (
                    item.runtime_available is False
                    and item.enabled
                    and not item.detached
                )
                for item in items
            ),
        ),
        updated_at=_NOW,
    )


def _indexed_item(
    *,
    lifecycle_state: str = "active",
    sync_state: str = "succeeded",
    error: bool = False,
    label: str = "Indexed Docs",
    binding_id: str = "idx-1",
) -> KnowledgeInventoryItemV1:
    return KnowledgeInventoryItemV1(
        tenant_id=_TENANT,
        workspace_id="ws-1",
        knowledge_item_id=indexed_knowledge_item_id(binding_id),
        mode=KnowledgeAccessModeV1.INDEXED,
        source_id="source-1",
        indexed_source_binding_id=binding_id,
        display_label=label,
        lifecycle_state="error" if error else lifecycle_state,
        enabled=True,
        detached=False,
        sync_state="failed" if error else sync_state,
        last_error_code="sync_failed" if error else None,
        revision=3,
        revision_kind=KnowledgeRevisionKindV1.LIFECYCLE,
        available_actions=(
            KnowledgeOperationV1.SYNC,
            KnowledgeOperationV1.RETRY_SYNC,
            KnowledgeOperationV1.DISABLE,
            KnowledgeOperationV1.DETACH,
        )
        if error
        else (
            KnowledgeOperationV1.SYNC,
            KnowledgeOperationV1.DISABLE,
            KnowledgeOperationV1.DETACH,
        ),
        updated_at=_NOW,
    )


def _live_item(
    *,
    unavailable: bool = False,
    label: str = "Live Wiki",
    binding_id: str = "live-1",
) -> KnowledgeInventoryItemV1:
    return KnowledgeInventoryItemV1(
        tenant_id=_TENANT,
        workspace_id="ws-1",
        knowledge_item_id=live_knowledge_item_id(binding_id),
        mode=KnowledgeAccessModeV1.LIVE,
        live_access_binding_id=binding_id,
        connection_ref="conn-1",
        display_label=label,
        provider_id="provider-1",
        source_kind="wiki",
        capability_ids=("cap.read",),
        lifecycle_state="active",
        enabled=True,
        detached=False,
        runtime_available=not unavailable,
        last_error_code="connection_unavailable" if unavailable else None,
        revision=4,
        revision_kind=KnowledgeRevisionKindV1.CONFIGURATION,
        available_actions=(KnowledgeOperationV1.DISABLE, KnowledgeOperationV1.DETACH),
        updated_at=_NOW,
    )


@pytest.fixture
def api_bundle(tmp_path, monkeypatch: pytest.MonkeyPatch):
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    data_home = tmp_path / "data"
    data_home.mkdir()
    monkeypatch.setenv("DATA_HOME", str(data_home))
    monkeypatch.setenv("INTERGRAX_ALLOWED_READ_ROOTS", str(tmp_path / "docs"))
    (tmp_path / "docs").mkdir()
    settings = replace(
        LocalWorkspaceBackendSettings.from_env(),
        data_home=str(data_home),
        knowledge_admin_confirmation_secret=_CONFIRM_SECRET,
    )
    executor = _FakeExecutor()
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
    with TestClient(app) as client:
        yield client, repo, service, workspace.workspace_id, app


def _snapshot_path(workspace_id: str) -> str:
    return f"{_PREFIX}/workspaces/{workspace_id}/setup-snapshot"


def _mount_inspection(app: FastAPI, inventory: KnowledgeInventoryV1) -> MagicMock:
    inspection = MagicMock()
    inspection.list_items.return_value = inventory
    app.state.lkw_knowledge_inspection_service = inspection
    return inspection


def _mount_readiness(app: FastAPI, *, accepts_new_work: bool = True) -> None:
    app.state.lkw_workspace_setup_snapshot_service = WorkspaceSetupSnapshotService(
        workspace_service=app.state.lkw_managed_workspace_service,
        inspection_service=app.state.lkw_knowledge_inspection_service,
        readiness_provider=_ReadinessStub(accepts_new_work=accepts_new_work),
    )


def test_no_knowledge_snapshot(api_bundle) -> None:
    client, _, _, workspace_id, app = api_bundle
    _mount_inspection(app, _inventory(items=()))
    _mount_readiness(app)

    response = client.get(_snapshot_path(workspace_id), headers=_headers())

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["phase"] == SetupPhaseV1.NO_KNOWLEDGE
    assert body["can_ask"] is False
    assert body["next_action"] == SetupNextActionV1.ADD_SOURCE
    assert body["suggested_question"] is None
    assert body["knowledge_summary"]["total"] == 0


def test_configuring_snapshot(api_bundle) -> None:
    client, _, _, workspace_id, app = api_bundle
    item = _indexed_item(lifecycle_state="ready", sync_state="never_synced")
    _mount_inspection(app, _inventory(items=(item,)))
    _mount_readiness(app)

    response = client.get(_snapshot_path(workspace_id), headers=_headers())

    assert response.status_code == 200
    body = response.json()
    assert body["phase"] == SetupPhaseV1.CONFIGURING
    assert body["can_ask"] is False
    assert body["next_action"] == SetupNextActionV1.WAIT_FOR_SYNC


def test_syncing_from_inventory_state(api_bundle) -> None:
    client, _, _, workspace_id, app = api_bundle
    item = _indexed_item(lifecycle_state="syncing", sync_state="running")
    _mount_inspection(app, _inventory(items=(item,)))
    _mount_readiness(app)

    response = client.get(_snapshot_path(workspace_id), headers=_headers())

    assert response.status_code == 200
    body = response.json()
    assert body["phase"] == SetupPhaseV1.SYNCING
    assert body["sync_in_progress"] is True
    assert body["next_action"] == SetupNextActionV1.WAIT_FOR_SYNC


def test_syncing_from_active_operation(api_bundle, monkeypatch: pytest.MonkeyPatch) -> None:
    client, repo, _, workspace_id, app = api_bundle
    _mount_inspection(app, _inventory(items=()))
    _mount_readiness(app)
    repo.put_operation(
        WorkspaceOperation(
            operation_id="op-running",
            tenant_id=_TENANT,
            workspace_id=workspace_id,
            source_id="source-1",
            operation_type=WorkspaceOperationType.SOURCE_SYNC,
            status=WorkspaceOperationStatus.RUNNING,
            created_at=_NOW,
        )
    )

    response = client.get(_snapshot_path(workspace_id), headers=_headers())

    assert response.status_code == 200
    body = response.json()
    assert body["phase"] == SetupPhaseV1.SYNCING
    assert body["recent_operation"]["operation_id"] == "op-running"


def test_attention_required_snapshot(api_bundle) -> None:
    client, _, _, workspace_id, app = api_bundle
    items = (_indexed_item(error=True),)
    _mount_inspection(app, _inventory(items=items))
    _mount_readiness(app)

    response = client.get(_snapshot_path(workspace_id), headers=_headers())

    assert response.status_code == 200
    body = response.json()
    assert body["phase"] == SetupPhaseV1.ATTENTION_REQUIRED
    assert body["attention_required"] is True
    assert body["attention"]["error_code"] == "sync_failed"
    assert "retry_sync" in body["attention"]["available_actions"]
    assert body["next_action"] == SetupNextActionV1.RETRY_OR_FIX_SOURCE
    assert "exception" not in response.text.casefold()
    assert "token" not in response.text.casefold()


def test_ready_indexed_snapshot(api_bundle) -> None:
    client, _, _, workspace_id, app = api_bundle
    _mount_inspection(app, _inventory(items=(_indexed_item(),)))
    _mount_readiness(app)

    response = client.get(_snapshot_path(workspace_id), headers=_headers())

    assert response.status_code == 200
    body = response.json()
    assert body["phase"] == SetupPhaseV1.READY
    assert body["can_ask"] is True
    assert body["has_usable_knowledge"] is True
    assert body["next_action"] == SetupNextActionV1.ASK_QUESTION
    assert body["suggested_question"] == "What information is available in Indexed Docs?"


def test_ready_live_snapshot(api_bundle) -> None:
    client, _, _, workspace_id, app = api_bundle
    _mount_inspection(app, _inventory(items=(_live_item(),)))
    _mount_readiness(app)

    response = client.get(_snapshot_path(workspace_id), headers=_headers())

    assert response.status_code == 200
    body = response.json()
    assert body["phase"] == SetupPhaseV1.READY
    assert body["can_ask"] is True


def test_ready_but_host_not_ready_blocks_ask(api_bundle) -> None:
    client, _, _, workspace_id, app = api_bundle
    _mount_inspection(app, _inventory(items=(_indexed_item(),)))
    _mount_readiness(app, accepts_new_work=False)

    response = client.get(_snapshot_path(workspace_id), headers=_headers())

    assert response.status_code == 200
    body = response.json()
    assert body["phase"] == SetupPhaseV1.READY
    assert body["host_ready"] is False
    assert body["can_ask"] is False
    assert body["next_action"] == SetupNextActionV1.NONE
    assert body["suggested_question"] == "What information is available in Indexed Docs?"


def test_multiple_items_summary_is_deterministic(api_bundle) -> None:
    client, _, _, workspace_id, app = api_bundle
    items = (
        _indexed_item(binding_id="idx-2", label="Beta"),
        _indexed_item(binding_id="idx-1", label="Alpha"),
        _live_item(binding_id="live-1", label="Gamma"),
    )
    _mount_inspection(app, _inventory(items=items))
    _mount_readiness(app)

    first = client.get(_snapshot_path(workspace_id), headers=_headers()).json()
    second = client.get(_snapshot_path(workspace_id), headers=_headers()).json()

    assert first == second
    assert first["knowledge_summary"]["total"] == 3
    assert first["knowledge_summary"]["usable"] == 3
    assert first["suggested_question"] == "What information is available in Alpha?"


def test_generic_suggested_question_without_label(api_bundle) -> None:
    client, _, _, workspace_id, app = api_bundle
    item = KnowledgeInventoryItemV1(
        tenant_id=_TENANT,
        workspace_id="ws-1",
        knowledge_item_id=indexed_knowledge_item_id("idx-1"),
        mode=KnowledgeAccessModeV1.INDEXED,
        source_id="source-1",
        indexed_source_binding_id="idx-1",
        display_label=None,
        lifecycle_state="active",
        enabled=True,
        detached=False,
        sync_state="succeeded",
        revision=3,
        revision_kind=KnowledgeRevisionKindV1.LIFECYCLE,
        available_actions=(
            KnowledgeOperationV1.SYNC,
            KnowledgeOperationV1.DISABLE,
            KnowledgeOperationV1.DETACH,
        ),
        updated_at=_NOW,
    )
    _mount_inspection(app, _inventory(items=(item,)))
    _mount_readiness(app)

    response = client.get(_snapshot_path(workspace_id), headers=_headers())

    assert response.status_code == 200
    body = response.json()
    assert body["suggested_question"] == (
        "What are the key points in my connected knowledge?"
    )


def test_tenant_isolation_returns_404(api_bundle) -> None:
    client, _, _, workspace_id, app = api_bundle
    _mount_inspection(app, _inventory(items=()))
    _mount_readiness(app)

    foreign = client.get(_snapshot_path(workspace_id), headers=_headers(_TENANT_B))
    missing = client.get(_snapshot_path("missing"), headers=_headers())

    assert foreign.status_code == 404
    assert missing.status_code == 404


def test_repeated_snapshot_has_no_persisted_onboarding_state(api_bundle) -> None:
    client, repo, _, workspace_id, app = api_bundle
    _mount_inspection(app, _inventory(items=(_indexed_item(),)))
    _mount_readiness(app)
    writes: list[str] = []
    for method_name in ("put_workspace", "put_operation", "put_source"):
        original = getattr(repo, method_name)

        def tracked(original=original, method_name=method_name, *args, **kwargs):
            writes.append(method_name)
            return original(*args, **kwargs)

        setattr(repo, method_name, tracked)

    first = client.get(_snapshot_path(workspace_id), headers=_headers()).json()
    second = client.get(_snapshot_path(workspace_id), headers=_headers()).json()

    assert first == second
    assert writes == []


def test_state_transition_no_knowledge_to_syncing_to_ready(api_bundle) -> None:
    client, repo, _, workspace_id, app = api_bundle
    inspection = MagicMock()
    app.state.lkw_knowledge_inspection_service = inspection
    _mount_readiness(app)

    inspection.list_items.return_value = _inventory(items=())
    empty = client.get(_snapshot_path(workspace_id), headers=_headers()).json()
    assert empty["phase"] == SetupPhaseV1.NO_KNOWLEDGE

    inspection.list_items.return_value = _inventory(
        items=(_indexed_item(lifecycle_state="syncing", sync_state="running"),)
    )
    syncing = client.get(_snapshot_path(workspace_id), headers=_headers()).json()
    assert syncing["phase"] == SetupPhaseV1.SYNCING

    inspection.list_items.return_value = _inventory(items=(_indexed_item(),))
    ready = client.get(_snapshot_path(workspace_id), headers=_headers()).json()
    assert ready["phase"] == SetupPhaseV1.READY
    assert repo.list_operations(tenant_id=_TENANT) == []


def test_attention_precedence_over_ready(api_bundle) -> None:
    client, _, _, workspace_id, app = api_bundle
    items = (_indexed_item(), _indexed_item(binding_id="idx-2", error=True))
    _mount_inspection(app, _inventory(items=items))
    _mount_readiness(app)

    response = client.get(_snapshot_path(workspace_id), headers=_headers())

    assert response.status_code == 200
    body = response.json()
    assert body["phase"] == SetupPhaseV1.ATTENTION_REQUIRED
    assert body["has_usable_knowledge"] is True


def test_next_action_never_asks_when_can_ask_false(api_bundle) -> None:
    client, _, _, workspace_id, app = api_bundle
    scenarios = (
        _inventory(items=()),
        _inventory(items=(_indexed_item(lifecycle_state="ready", sync_state="never_synced"),)),
        _inventory(items=(_indexed_item(lifecycle_state="syncing", sync_state="running"),)),
        _inventory(items=(_indexed_item(error=True),)),
        _inventory(items=(_indexed_item(),)),
    )
    for inventory in scenarios:
        _mount_inspection(app, inventory)
        _mount_readiness(app, accepts_new_work=False)
        body = client.get(_snapshot_path(workspace_id), headers=_headers()).json()
        if body["can_ask"] is False:
            assert body["next_action"] != SetupNextActionV1.ASK_QUESTION


def test_derivation_service_does_not_write_repository_state(
    api_bundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, repo, service, workspace_id, app = api_bundle
    inspection = MagicMock()
    inspection.list_items.return_value = _inventory(items=(_indexed_item(),))
    writes: list[str] = []
    for method_name in (
        "put_workspace",
        "put_operation",
        "put_source",
        "delete_workspace",
        "delete_operation",
    ):
        original = getattr(repo, method_name)

        def tracked(original=original, method_name=method_name, *args, **kwargs):
            writes.append(method_name)
            return original(*args, **kwargs)

        monkeypatch.setattr(repo, method_name, tracked)

    snapshot_service = WorkspaceSetupSnapshotService(
        workspace_service=service,
        inspection_service=inspection,
        readiness_provider=_ReadinessStub(),
    )
    snapshot_service.derive_snapshot(tenant_id=_TENANT, workspace_id=workspace_id)

    assert writes == []
