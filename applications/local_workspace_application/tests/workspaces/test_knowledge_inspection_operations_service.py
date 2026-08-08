# © Artur Czarnecki. All rights reserved.

"""Tests for the provider-neutral knowledge inspection and operation boundary."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from local_workspace_application.workspaces.knowledge_indexed_source_lifecycle_service import (
    IndexedSourceLifecycleStateV1,
    IndexedSourceSyncStateV1,
    WorkspaceIndexedSourceLifecycleError,
)
from local_workspace_application.workspaces.knowledge_inspection_operations_service import (
    KnowledgeAccessModeV1,
    KnowledgeInspectionService,
    KnowledgeInventoryError,
    KnowledgeOperationCommandV1,
    KnowledgeOperationError,
    KnowledgeOperationsService,
    KnowledgeOperationV1,
    KnowledgeRevisionKindV1,
    indexed_knowledge_item_id,
    live_knowledge_item_id,
)
from local_workspace_application.workspaces.knowledge_live_access_service import (
    LiveAccessLifecycleStateV1,
    WorkspaceLiveAccessBindingError,
)

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 8, 8, 10, 0, tzinfo=UTC)
_TENANT = "tenant-a"
_WORKSPACE = "workspace-a"
_OTHER_WORKSPACE = "workspace-b"
_INDEXED_ID = "indexed-binding-a"
_LIVE_ID = "live-binding-a"
_HASH = "a" * 64


def _indexed_view(
    state: IndexedSourceLifecycleStateV1 = IndexedSourceLifecycleStateV1.ACTIVE,
    *,
    revision: int = 3,
    sync_state: IndexedSourceSyncStateV1 = IndexedSourceSyncStateV1.SUCCEEDED,
    detached: bool = False,
    enabled: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id="source-a",
        indexed_source_binding_id=_INDEXED_ID,
        knowledge_source_binding_ref="source-binding-a",
        lifecycle_state=state,
        lifecycle_revision=revision,
        enabled=enabled,
        detached=detached,
        sync_state=sync_state,
        last_successful_sync_at=_NOW,
        last_error_code=None,
        updated_at=_NOW,
    )


def _live_view(
    state: LiveAccessLifecycleStateV1 = LiveAccessLifecycleStateV1.ACTIVE,
    *,
    revision: int = 4,
    runtime_available: bool = True,
    detached: bool = False,
    enabled: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        live_access_binding_id=_LIVE_ID,
        connection_ref="connection-a",
        lifecycle_state=state,
        configuration_revision=revision,
        enabled=enabled,
        detached=detached,
        runtime_available=runtime_available,
        last_error_code=None if runtime_available else "connection_unavailable",
        updated_at=_NOW,
    )


def _configuration(*, workspace_id: str = _WORKSPACE) -> SimpleNamespace:
    return SimpleNamespace(
        tenant_id=_TENANT,
        workspace_id=workspace_id,
        indexed_sources=(
            SimpleNamespace(
                indexed_source_binding_id=_INDEXED_ID,
                cached_safe_display_label="Indexed",
            ),
        ),
        live_access_bindings=(
            SimpleNamespace(
                live_access_binding_id=_LIVE_ID,
                derived_safe_display_label="Live",
            ),
        ),
        updated_at=_NOW,
    )


class _ConfigurationService:
    def __init__(self, configuration: SimpleNamespace | None) -> None:
        self.configuration = configuration

    def get_configuration(self, *, tenant_id: str, workspace_id: str):
        if self.configuration is None or workspace_id != self.configuration.workspace_id:
            return None
        return self.configuration


class _IndexedLifecycle:
    def __init__(self, view: SimpleNamespace) -> None:
        self.view = view
        self.calls: list[tuple[str, object]] = []

    def get(self, **kwargs: object):
        self.calls.append(("get", kwargs))
        if kwargs["workspace_id"] != self.view.workspace_id:
            raise WorkspaceIndexedSourceLifecycleError("indexed_source_not_found")
        return self.view

    def request_sync(self, command):
        self.calls.append(("sync", command))
        return SimpleNamespace(operation_id="operation-sync", mutation_id="mutation-sync")

    def retry_sync(self, command):
        self.calls.append(("retry_sync", command))
        return SimpleNamespace(operation_id="operation-retry", mutation_id=None)

    def disable(self, command):
        self.calls.append(("disable", command))
        return SimpleNamespace(operation_id=None, mutation_id="mutation-disable")

    def enable(self, command):
        self.calls.append(("enable", command))
        return SimpleNamespace(operation_id=None, mutation_id="mutation-enable")

    def detach(self, command):
        self.calls.append(("detach", command))
        return SimpleNamespace(operation_id=None, mutation_id=None)

    def resume_detach(self, command):
        self.calls.append(("resume_detach", command))
        return SimpleNamespace(operation_id=None, mutation_id=None)


class _LiveLifecycle:
    def __init__(self, view: SimpleNamespace) -> None:
        self.view = view
        self.calls: list[tuple[str, object]] = []

    def get(self, command):
        self.calls.append(("get", command))
        if command.workspace_id != self.view.workspace_id:
            raise WorkspaceLiveAccessBindingError("live_access_not_found")
        return self.view

    def disable(self, command):
        self.calls.append(("disable", command))
        return SimpleNamespace()

    async def enable(self, command):
        self.calls.append(("enable", command))
        return SimpleNamespace()

    def detach(self, command):
        self.calls.append(("detach", command))
        return SimpleNamespace()


def _services(
    *,
    indexed_view: SimpleNamespace | None = None,
    live_view: SimpleNamespace | None = None,
    workspace_id: str = _WORKSPACE,
) -> tuple[KnowledgeInspectionService, KnowledgeOperationsService, _IndexedLifecycle, _LiveLifecycle]:
    indexed = _IndexedLifecycle(indexed_view or _indexed_view())
    live = _LiveLifecycle(live_view or _live_view())
    inspection = KnowledgeInspectionService(
        configuration_service=_ConfigurationService(
            _configuration(workspace_id=workspace_id)
        ),
        indexed_source_lifecycle_service=indexed,
        live_access_lifecycle_service=live,
    )
    return inspection, KnowledgeOperationsService(
        inspection_service=inspection,
        indexed_source_lifecycle_service=indexed,
        live_access_lifecycle_service=live,
    ), indexed, live


def _command(
    item_id: str,
    operation: KnowledgeOperationV1,
    *,
    revision: int = 3,
    operation_id: str | None = None,
    workspace_id: str = _WORKSPACE,
) -> KnowledgeOperationCommandV1:
    return KnowledgeOperationCommandV1(
        tenant_id=_TENANT,
        workspace_id=workspace_id,
        knowledge_item_id=item_id,
        operation=operation,
        expected_revision=revision,
        idempotency_key_hash=_HASH,
        operation_id=operation_id,
    )


def test_empty_workspace_returns_empty_inventory() -> None:
    config = SimpleNamespace(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        indexed_sources=(),
        live_access_bindings=(),
        updated_at=_NOW,
    )
    inspection = KnowledgeInspectionService(
        configuration_service=_ConfigurationService(config),
        indexed_source_lifecycle_service=_IndexedLifecycle(_indexed_view()),
        live_access_lifecycle_service=_LiveLifecycle(_live_view()),
    )

    inventory = inspection.list_items(tenant_id=_TENANT, workspace_id=_WORKSPACE)

    assert inventory.items == ()
    assert inventory.summary.total == 0


def test_list_items_maps_indexed_lifecycle_failure_to_inventory_error() -> None:
    inspection, _, indexed, _ = _services()

    def failing_get(**kwargs: object):
        raise WorkspaceIndexedSourceLifecycleError("indexed_source_not_found")

    indexed.get = failing_get

    with pytest.raises(KnowledgeInventoryError, match="knowledge_item_not_found"):
        inspection.list_items(tenant_id=_TENANT, workspace_id=_WORKSPACE)


def test_list_items_maps_unexpected_live_lifecycle_failure_to_inventory_unavailable() -> None:
    inspection, _, _, live = _services()

    def failing_get(command: object):
        raise RuntimeError("lifecycle storage unavailable")

    live.get = failing_get

    with pytest.raises(KnowledgeInventoryError, match="knowledge_inventory_unavailable"):
        inspection.list_items(tenant_id=_TENANT, workspace_id=_WORKSPACE)


def test_inventory_projects_both_modes_with_deterministic_order_and_revision_authority() -> None:
    inspection, _, _, _ = _services()

    inventory = inspection.list_items(tenant_id=_TENANT, workspace_id=_WORKSPACE)

    assert [item.mode for item in inventory.items] == [
        KnowledgeAccessModeV1.INDEXED,
        KnowledgeAccessModeV1.LIVE,
    ]
    assert [item.knowledge_item_id for item in inventory.items] == [
        indexed_knowledge_item_id(_INDEXED_ID),
        live_knowledge_item_id(_LIVE_ID),
    ]
    assert inventory.items[0].revision_kind is KnowledgeRevisionKindV1.LIFECYCLE
    assert inventory.items[0].revision == 3
    assert inventory.items[1].revision_kind is KnowledgeRevisionKindV1.CONFIGURATION
    assert inventory.items[1].revision == 4
    assert inventory.summary.indexed == 1
    assert inventory.summary.live == 1


def test_inventory_exposes_detached_item_kept_by_current_configuration() -> None:
    inspection, _, _, _ = _services(
        indexed_view=_indexed_view(
            IndexedSourceLifecycleStateV1.DETACHED,
            detached=True,
            enabled=True,
        )
    )

    item = inspection.get_item(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        knowledge_item_id=indexed_knowledge_item_id(_INDEXED_ID),
    )

    assert item.detached is True
    assert item.available_actions == ()


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (
            IndexedSourceLifecycleStateV1.READY,
            (
                KnowledgeOperationV1.SYNC,
                KnowledgeOperationV1.DISABLE,
                KnowledgeOperationV1.DETACH,
            ),
        ),
        (
            IndexedSourceLifecycleStateV1.SYNCING,
            (KnowledgeOperationV1.DISABLE,),
        ),
        (
            IndexedSourceLifecycleStateV1.ACTIVE,
            (
                KnowledgeOperationV1.SYNC,
                KnowledgeOperationV1.DISABLE,
                KnowledgeOperationV1.DETACH,
            ),
        ),
        (
            IndexedSourceLifecycleStateV1.DISABLED,
            (KnowledgeOperationV1.ENABLE, KnowledgeOperationV1.DETACH),
        ),
        (
            IndexedSourceLifecycleStateV1.DETACHING,
            (KnowledgeOperationV1.RESUME_DETACH,),
        ),
        (
            IndexedSourceLifecycleStateV1.DETACH_BLOCKED,
            (KnowledgeOperationV1.RESUME_DETACH,),
        ),
        (IndexedSourceLifecycleStateV1.DETACHED, ()),
    ],
)
def test_indexed_available_actions(
    state: IndexedSourceLifecycleStateV1,
    expected: tuple[KnowledgeOperationV1, ...],
) -> None:
    inspection, _, _, _ = _services(
        indexed_view=_indexed_view(
            state,
            detached=state is IndexedSourceLifecycleStateV1.DETACHED,
            enabled=state is not IndexedSourceLifecycleStateV1.DISABLED,
        )
    )

    item = inspection.get_item(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        knowledge_item_id=indexed_knowledge_item_id(_INDEXED_ID),
    )

    assert item.available_actions == expected


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (
            LiveAccessLifecycleStateV1.ACTIVE,
            (KnowledgeOperationV1.DISABLE, KnowledgeOperationV1.DETACH),
        ),
        (
            LiveAccessLifecycleStateV1.READY,
            (KnowledgeOperationV1.DISABLE, KnowledgeOperationV1.DETACH),
        ),
        (
            LiveAccessLifecycleStateV1.DISABLED,
            (KnowledgeOperationV1.ENABLE, KnowledgeOperationV1.DETACH),
        ),
        (LiveAccessLifecycleStateV1.DETACHED, ()),
        (
            LiveAccessLifecycleStateV1.ERROR,
            (KnowledgeOperationV1.DISABLE, KnowledgeOperationV1.DETACH),
        ),
    ],
)
def test_live_available_actions(
    state: LiveAccessLifecycleStateV1,
    expected: tuple[KnowledgeOperationV1, ...],
) -> None:
    inspection, _, _, _ = _services(
        live_view=_live_view(
            state,
            detached=state is LiveAccessLifecycleStateV1.DETACHED,
            enabled=state is not LiveAccessLifecycleStateV1.DISABLED,
            runtime_available=state is LiveAccessLifecycleStateV1.ACTIVE,
        )
    )

    item = inspection.get_item(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        knowledge_item_id=live_knowledge_item_id(_LIVE_ID),
    )

    assert item.available_actions == expected


@pytest.mark.asyncio
async def test_operations_delegate_indexed_and_live_without_cross_mode_mutation() -> None:
    inspection, operations, indexed, live = _services()

    indexed_result = await operations.execute(
        _command(
            indexed_knowledge_item_id(_INDEXED_ID),
            KnowledgeOperationV1.SYNC,
            revision=3,
        )
    )
    live_result = await operations.execute(
        _command(
            live_knowledge_item_id(_LIVE_ID),
            KnowledgeOperationV1.DISABLE,
            revision=4,
        )
    )

    assert indexed_result.operation_id == "operation-sync"
    assert live_result.operation is KnowledgeOperationV1.DISABLE
    assert [name for name, _ in indexed.calls if name != "get"] == ["sync"]
    assert [name for name, _ in live.calls if name != "get"] == ["disable"]
    assert inspection.get_item(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        knowledge_item_id=indexed_knowledge_item_id(_INDEXED_ID),
    ).mode is KnowledgeAccessModeV1.INDEXED


@pytest.mark.asyncio
async def test_retry_requires_target_and_delegates_operation_id() -> None:
    _inspection, operations, indexed, _ = _services(
        indexed_view=_indexed_view(
            IndexedSourceLifecycleStateV1.ERROR,
            sync_state=IndexedSourceSyncStateV1.FAILED,
        )
    )

    with pytest.raises(KnowledgeOperationError, match="knowledge_operation_retry_target_required"):
        await operations.execute(
            _command(
                indexed_knowledge_item_id(_INDEXED_ID),
                KnowledgeOperationV1.RETRY_SYNC,
            )
        )

    await operations.execute(
        _command(
            indexed_knowledge_item_id(_INDEXED_ID),
            KnowledgeOperationV1.RETRY_SYNC,
            operation_id="operation-failed",
        )
    )
    retry_commands = [command for name, command in indexed.calls if name == "retry_sync"]
    assert retry_commands[0].operation_id == "operation-failed"


def test_unknown_and_cross_workspace_items_are_not_found() -> None:
    inspection, _, _, _ = _services()

    with pytest.raises(KnowledgeInventoryError, match="knowledge_item_not_found"):
        inspection.get_item(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            knowledge_item_id="live:other-binding",
        )
    with pytest.raises(KnowledgeInventoryError, match="knowledge_item_not_found"):
        inspection.get_item(
            tenant_id=_TENANT,
            workspace_id=_OTHER_WORKSPACE,
            knowledge_item_id=live_knowledge_item_id(_LIVE_ID),
        )


@pytest.mark.asyncio
async def test_stale_revision_maps_to_stable_conflict() -> None:
    _inspection, operations, indexed, _ = _services()

    def stale_disable(command) -> None:
        indexed.calls.append(("disable", command))
        raise WorkspaceIndexedSourceLifecycleError("lifecycle_conflict")

    indexed.disable = stale_disable
    with pytest.raises(KnowledgeOperationError, match="knowledge_operation_conflict"):
        await operations.execute(
            _command(
                indexed_knowledge_item_id(_INDEXED_ID),
                KnowledgeOperationV1.DISABLE,
                revision=2,
            )
        )
    assert [name for name, _ in indexed.calls if name != "get"] == []


@pytest.mark.asyncio
async def test_indexed_stale_revision_wins_over_current_unsupported_action() -> None:
    _inspection, operations, indexed, _ = _services(
        indexed_view=_indexed_view(
            IndexedSourceLifecycleStateV1.DISABLED,
            revision=4,
            enabled=False,
        )
    )

    with pytest.raises(KnowledgeOperationError, match="knowledge_operation_conflict"):
        await operations.execute(
            _command(
                indexed_knowledge_item_id(_INDEXED_ID),
                KnowledgeOperationV1.DISABLE,
                revision=3,
            )
        )

    assert [name for name, _ in indexed.calls if name != "get"] == []


@pytest.mark.asyncio
async def test_current_revision_unsupported_action_maps_to_not_supported() -> None:
    _inspection, operations, indexed, _ = _services(
        indexed_view=_indexed_view(
            IndexedSourceLifecycleStateV1.DISABLED,
            revision=4,
            enabled=False,
        )
    )

    with pytest.raises(KnowledgeOperationError, match="knowledge_operation_not_supported"):
        await operations.execute(
            _command(
                indexed_knowledge_item_id(_INDEXED_ID),
                KnowledgeOperationV1.DISABLE,
                revision=4,
            )
        )

    assert [name for name, _ in indexed.calls if name != "get"] == []


@pytest.mark.asyncio
async def test_live_stale_configuration_revision_maps_to_stable_conflict() -> None:
    _inspection, operations, _, live = _services()

    def stale_disable(command) -> None:
        live.calls.append(("disable", command))
        raise WorkspaceLiveAccessBindingError("configuration_revision_conflict")

    live.disable = stale_disable
    with pytest.raises(KnowledgeOperationError, match="knowledge_operation_conflict"):
        await operations.execute(
            _command(
                live_knowledge_item_id(_LIVE_ID),
                KnowledgeOperationV1.DISABLE,
                revision=3,
            )
        )


@pytest.mark.asyncio
async def test_live_stale_revision_wins_over_current_unsupported_action() -> None:
    _inspection, operations, _, live = _services(
        live_view=_live_view(
            LiveAccessLifecycleStateV1.DISABLED,
            revision=5,
            enabled=False,
        )
    )

    with pytest.raises(KnowledgeOperationError, match="knowledge_operation_conflict"):
        await operations.execute(
            _command(
                live_knowledge_item_id(_LIVE_ID),
                KnowledgeOperationV1.DISABLE,
                revision=4,
            )
        )

    assert [name for name, _ in live.calls if name != "get"] == []


def test_reconstructed_services_rebuild_inventory_without_process_state() -> None:
    first_inspection, _, _, _ = _services()
    first = first_inspection.list_items(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )

    second_inspection, _, _, _ = _services()
    second = second_inspection.list_items(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )

    assert second == first
    assert second_inspection.get_item(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        knowledge_item_id=live_knowledge_item_id(_LIVE_ID),
    ).revision_kind is KnowledgeRevisionKindV1.CONFIGURATION


def test_models_are_frozen_and_forbid_extra_fields() -> None:
    inspection, _, _, _ = _services()
    item = inspection.get_item(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        knowledge_item_id=indexed_knowledge_item_id(_INDEXED_ID),
    )

    with pytest.raises((TypeError, ValueError)):
        item.lifecycle_state = "changed"  # type: ignore[misc]
    with pytest.raises(ValueError):
        KnowledgeOperationCommandV1(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            knowledge_item_id=item.knowledge_item_id,
            operation=KnowledgeOperationV1.SYNC,
            expected_revision=3,
            idempotency_key_hash=_HASH,
            extra_field="forbidden",
        )
