# © Artur Czarnecki. All rights reserved.

"""Tests for KnowledgeAskScopeV1 resolution and validation."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from local_workspace_application.workspaces.knowledge_ask_scope_models import (
    KnowledgeAskScopeError,
    KnowledgeAskScopeV1,
)
from local_workspace_application.workspaces.knowledge_ask_scope_resolver import (
    KnowledgeAskScopeResolver,
)
from local_workspace_application.workspaces.knowledge_indexed_source_lifecycle_service import (
    IndexedSourceLifecycleStateV1,
    IndexedSourceSyncStateV1,
)
from local_workspace_application.workspaces.knowledge_inspection_operations_service import (
    KnowledgeAccessModeV1,
    KnowledgeInspectionService,
    KnowledgeInventoryError,
    indexed_knowledge_item_id,
    live_knowledge_item_id,
)
from local_workspace_application.workspaces.knowledge_live_access_service import (
    LiveAccessLifecycleStateV1,
)

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 8, 11, 10, 0, tzinfo=UTC)
_TENANT = "tenant-a"
_WORKSPACE = "workspace-a"
_INDEXED_ID = "indexed-binding-a"
_LIVE_ID = "live-binding-a"


def _indexed_view(
    *,
    source_id: str = "source-a",
    enabled: bool = True,
    detached: bool = False,
    workspace_id: str = _WORKSPACE,
) -> SimpleNamespace:
    return SimpleNamespace(
        tenant_id=_TENANT,
        workspace_id=workspace_id,
        source_id=source_id,
        indexed_source_binding_id=_INDEXED_ID,
        knowledge_source_binding_ref="source-binding-a",
        lifecycle_state=IndexedSourceLifecycleStateV1.ACTIVE,
        lifecycle_revision=3,
        enabled=enabled,
        detached=detached,
        sync_state=IndexedSourceSyncStateV1.SUCCEEDED,
        last_successful_sync_at=_NOW,
        last_error_code=None,
        updated_at=_NOW,
    )


def _live_view(*, workspace_id: str = _WORKSPACE) -> SimpleNamespace:
    return SimpleNamespace(
        tenant_id=_TENANT,
        workspace_id=workspace_id,
        live_access_binding_id=_LIVE_ID,
        connection_ref="connection-a",
        knowledge_source_binding_ref="live-binding-a",
        lifecycle_state=LiveAccessLifecycleStateV1.ACTIVE,
        configuration_revision=4,
        enabled=True,
        detached=False,
        runtime_available=True,
        last_error_code=None,
        updated_at=_NOW,
    )


class _IndexedLifecycle:
    def get(self, *, tenant_id: str, workspace_id: str, indexed_source_binding_id: str):
        _ = tenant_id, workspace_id
        if indexed_source_binding_id != _INDEXED_ID:
            raise AssertionError("unexpected binding")
        return _indexed_view()


class _LiveLifecycle:
    def get(self, command):
        _ = command
        return _live_view()


class _ConfigurationService:
    def get_configuration(self, *, tenant_id: str, workspace_id: str):
        _ = tenant_id
        if workspace_id != _WORKSPACE:
            return None
        return SimpleNamespace(
            indexed_sources=[
                SimpleNamespace(
                    indexed_source_binding_id=_INDEXED_ID,
                    cached_safe_display_label="Project Drive",
                )
            ],
            live_access_bindings=[
                SimpleNamespace(
                    live_access_binding_id=_LIVE_ID,
                    derived_safe_display_label="HR API",
                    derived_provider_id="hr",
                    derived_resource_type="folder",
                    allowed_capability_ids=(),
                )
            ],
            updated_at=_NOW,
        )


def _resolver(**overrides: object) -> KnowledgeAskScopeResolver:
    indexed = overrides.get("indexed_lifecycle", _IndexedLifecycle())
    live = overrides.get("live_lifecycle", _LiveLifecycle())
    configuration = overrides.get("configuration_service", _ConfigurationService())
    inspection = KnowledgeInspectionService(
        configuration_service=configuration,  # type: ignore[arg-type]
        indexed_source_lifecycle_service=indexed,  # type: ignore[arg-type]
        live_access_lifecycle_service=live,  # type: ignore[arg-type]
    )
    return KnowledgeAskScopeResolver(inspection)


def test_one_indexed_item_resolves_one_source_id() -> None:
    scope = KnowledgeAskScopeV1(
        knowledge_item_ids=(indexed_knowledge_item_id(_INDEXED_ID),)
    )
    resolved = _resolver().resolve(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        scope=scope,
    )
    assert resolved.allowed_source_ids == ("source-a",)


def test_multiple_indexed_items_resolve_multiple_source_ids() -> None:
    class _MultiIndexedLifecycle:
        def get(self, *, tenant_id: str, workspace_id: str, indexed_source_binding_id: str):
            _ = tenant_id, workspace_id
            source_by_binding = {
                "indexed-1": "source-1",
                "indexed-2": "source-2",
            }
            return _indexed_view(source_id=source_by_binding[indexed_source_binding_id])

    class _MultiConfiguration:
        def get_configuration(self, *, tenant_id: str, workspace_id: str):
            _ = tenant_id, workspace_id
            return SimpleNamespace(
                indexed_sources=[
                    SimpleNamespace(
                        indexed_source_binding_id="indexed-1",
                        cached_safe_display_label="A",
                    ),
                    SimpleNamespace(
                        indexed_source_binding_id="indexed-2",
                        cached_safe_display_label="B",
                    ),
                ],
                live_access_bindings=[],
                updated_at=_NOW,
            )

    resolver = _resolver(
        indexed_lifecycle=_MultiIndexedLifecycle(),
        configuration_service=_MultiConfiguration(),
    )
    scope = KnowledgeAskScopeV1(
        knowledge_item_ids=(
            indexed_knowledge_item_id("indexed-2"),
            indexed_knowledge_item_id("indexed-1"),
            indexed_knowledge_item_id("indexed-1"),
        )
    )
    resolved = resolver.resolve(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        scope=scope,
    )
    assert resolved.allowed_source_ids == ("source-1", "source-2")


def test_unknown_item_fails() -> None:
    scope = KnowledgeAskScopeV1(knowledge_item_ids=("indexed:missing",))
    with pytest.raises(KnowledgeAskScopeError) as exc:
        _resolver().resolve(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            scope=scope,
        )
    assert exc.value.error_code == "knowledge_ask_scope_not_found"


def test_cross_workspace_fails() -> None:
    scope = KnowledgeAskScopeV1(
        knowledge_item_ids=(indexed_knowledge_item_id(_INDEXED_ID),)
    )
    with pytest.raises(KnowledgeAskScopeError) as exc:
        _resolver().resolve(
            tenant_id=_TENANT,
            workspace_id="workspace-other",
            scope=scope,
        )
    assert exc.value.error_code == "knowledge_ask_scope_not_found"


def test_live_item_fails() -> None:
    scope = KnowledgeAskScopeV1(knowledge_item_ids=(live_knowledge_item_id(_LIVE_ID),))
    with pytest.raises(KnowledgeAskScopeError) as exc:
        _resolver().resolve(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            scope=scope,
        )
    assert exc.value.error_code == "knowledge_ask_scope_live_unsupported"


def test_mixed_indexed_and_live_fails() -> None:
    scope = KnowledgeAskScopeV1(
        knowledge_item_ids=(
            indexed_knowledge_item_id(_INDEXED_ID),
            live_knowledge_item_id(_LIVE_ID),
        )
    )
    with pytest.raises(KnowledgeAskScopeError) as exc:
        _resolver().resolve(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            scope=scope,
        )
    assert exc.value.error_code == "knowledge_ask_scope_live_unsupported"


def test_disabled_indexed_item_fails() -> None:
    class _DisabledIndexedLifecycle:
        def get(self, *, tenant_id: str, workspace_id: str, indexed_source_binding_id: str):
            _ = tenant_id, workspace_id, indexed_source_binding_id
            return _indexed_view(enabled=False)

    with pytest.raises(KnowledgeAskScopeError) as exc:
        _resolver(indexed_lifecycle=_DisabledIndexedLifecycle()).resolve(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            scope=KnowledgeAskScopeV1(
                knowledge_item_ids=(indexed_knowledge_item_id(_INDEXED_ID),)
            ),
        )
    assert exc.value.error_code == "knowledge_ask_scope_disabled"


def test_detached_indexed_item_fails() -> None:
    class _DetachedIndexedLifecycle:
        def get(self, *, tenant_id: str, workspace_id: str, indexed_source_binding_id: str):
            _ = tenant_id, workspace_id, indexed_source_binding_id
            return _indexed_view(detached=True)

    with pytest.raises(KnowledgeAskScopeError) as exc:
        _resolver(indexed_lifecycle=_DetachedIndexedLifecycle()).resolve(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            scope=KnowledgeAskScopeV1(
                knowledge_item_ids=(indexed_knowledge_item_id(_INDEXED_ID),)
            ),
        )
    assert exc.value.error_code == "knowledge_ask_scope_detached"


def test_missing_source_id_fails() -> None:
    class _MissingSourceLifecycle:
        def get(self, *, tenant_id: str, workspace_id: str, indexed_source_binding_id: str):
            _ = tenant_id, workspace_id, indexed_source_binding_id
            return _indexed_view(source_id="")

    with pytest.raises(KnowledgeAskScopeError) as exc:
        _resolver(indexed_lifecycle=_MissingSourceLifecycle()).resolve(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            scope=KnowledgeAskScopeV1(
                knowledge_item_ids=(indexed_knowledge_item_id(_INDEXED_ID),)
            ),
        )
    assert exc.value.error_code == "knowledge_ask_scope_invalid"


def test_explicit_empty_scope_fails_at_contract() -> None:
    with pytest.raises(ValueError):
        KnowledgeAskScopeV1(knowledge_item_ids=())


def test_inventory_unavailable_maps_to_invalid() -> None:
    class _BrokenConfiguration:
        def get_configuration(self, *, tenant_id: str, workspace_id: str):
            _ = tenant_id, workspace_id
            raise RuntimeError("boom")

    inspection = KnowledgeInspectionService(
        configuration_service=_BrokenConfiguration(),  # type: ignore[arg-type]
        indexed_source_lifecycle_service=_IndexedLifecycle(),  # type: ignore[arg-type]
        live_access_lifecycle_service=_LiveLifecycle(),  # type: ignore[arg-type]
    )
    with pytest.raises(KnowledgeInventoryError):
        inspection.get_item(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            knowledge_item_id=indexed_knowledge_item_id(_INDEXED_ID),
        )
