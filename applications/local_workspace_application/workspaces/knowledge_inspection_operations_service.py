# © Artur Czarnecki. All rights reserved.

"""Provider-neutral knowledge inventory inspection and delegated operations."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from local_workspace_application.workspaces.knowledge_configuration_models import (
    WorkspaceKnowledgeConfigurationV1,
)
from local_workspace_application.workspaces.knowledge_configuration_mutation_engine import (
    WorkspaceKnowledgeConfigurationMutationError,
)
from local_workspace_application.workspaces.knowledge_configuration_service import (
    WorkspaceKnowledgeConfigurationService,
)
from local_workspace_application.workspaces.knowledge_indexed_source_lifecycle_service import (
    IndexedSourceLifecycleCommand,
    IndexedSourceLifecycleService,
    IndexedSourceLifecycleStateV1,
    IndexedSourceLifecycleViewV1,
    IndexedSourceRetryCommand,
    IndexedSourceSyncCommand,
    WorkspaceIndexedSourceLifecycleError,
)
from local_workspace_application.workspaces.knowledge_live_access_service import (
    DetachWorkspaceLiveAccessBindingCommand,
    DisableWorkspaceLiveAccessBindingCommand,
    EnableWorkspaceLiveAccessBindingCommand,
    GetLiveAccessCommand,
    LiveAccessLifecycleService,
    LiveAccessLifecycleStateV1,
    LiveAccessLifecycleViewV1,
    WorkspaceLiveAccessBindingError,
)
from pydantic import BaseModel, ConfigDict, Field


class KnowledgeAccessModeV1(StrEnum):
    INDEXED = "indexed"
    LIVE = "live"


class KnowledgeOperationV1(StrEnum):
    SYNC = "sync"
    RETRY_SYNC = "retry_sync"
    DISABLE = "disable"
    ENABLE = "enable"
    DETACH = "detach"
    RESUME_DETACH = "resume_detach"


class KnowledgeRevisionKindV1(StrEnum):
    LIFECYCLE = "lifecycle"
    CONFIGURATION = "configuration"


class KnowledgeInventoryItemV1(BaseModel):
    """Credential-free, provider-neutral projection of one knowledge binding."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str
    workspace_id: str
    knowledge_item_id: str
    mode: KnowledgeAccessModeV1

    source_id: str | None = None
    indexed_source_binding_id: str | None = None
    live_access_binding_id: str | None = None
    knowledge_source_binding_ref: str | None = None
    connection_ref: str | None = None

    display_label: str | None = None

    lifecycle_state: str
    enabled: bool
    detached: bool

    runtime_available: bool | None = None
    sync_state: str | None = None
    last_successful_sync_at: datetime | None = None
    last_error_code: str | None = None

    revision: int = Field(ge=0)
    revision_kind: KnowledgeRevisionKindV1
    available_actions: tuple[KnowledgeOperationV1, ...]
    updated_at: datetime


class KnowledgeInventorySummaryV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    total: int = Field(ge=0)
    indexed: int = Field(ge=0)
    live: int = Field(ge=0)
    active: int = Field(ge=0)
    disabled: int = Field(ge=0)
    attention_required: int = Field(ge=0)


class KnowledgeInventoryV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str
    workspace_id: str
    items: tuple[KnowledgeInventoryItemV1, ...]
    summary: KnowledgeInventorySummaryV1
    updated_at: datetime


class KnowledgeInventoryError(RuntimeError):
    def __init__(self, error_code: str) -> None:
        super().__init__(error_code)
        self.error_code = error_code


class KnowledgeOperationError(RuntimeError):
    def __init__(self, error_code: str) -> None:
        super().__init__(error_code)
        self.error_code = error_code


class KnowledgeOperationCommandV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str
    workspace_id: str
    knowledge_item_id: str
    operation: KnowledgeOperationV1
    expected_revision: int = Field(ge=0)
    idempotency_key_hash: str = Field(min_length=64, max_length=64)
    operation_id: str | None = None


class KnowledgeOperationResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    item: KnowledgeInventoryItemV1
    operation: KnowledgeOperationV1
    operation_id: str | None = None
    mutation_id: str | None = None


def indexed_knowledge_item_id(indexed_source_binding_id: str) -> str:
    return f"indexed:{indexed_source_binding_id}"


def live_knowledge_item_id(live_access_binding_id: str) -> str:
    return f"live:{live_access_binding_id}"


def _parse_knowledge_item_id(
    knowledge_item_id: str,
) -> tuple[KnowledgeAccessModeV1, str] | None:
    mode, separator, binding_id = knowledge_item_id.partition(":")
    if not separator or not binding_id.strip():
        return None
    try:
        parsed_mode = KnowledgeAccessModeV1(mode)
    except ValueError:
        return None
    return parsed_mode, binding_id


def _value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _indexed_actions(view: IndexedSourceLifecycleViewV1) -> tuple[KnowledgeOperationV1, ...]:
    if view.detached or _value(view.lifecycle_state) == IndexedSourceLifecycleStateV1.DETACHED.value:
        return ()
    state = _value(view.lifecycle_state)
    if state in {
        IndexedSourceLifecycleStateV1.DETACHING.value,
        IndexedSourceLifecycleStateV1.DETACH_BLOCKED.value,
    }:
        return (KnowledgeOperationV1.RESUME_DETACH,)
    if state == IndexedSourceLifecycleStateV1.DISABLED.value:
        return (KnowledgeOperationV1.ENABLE, KnowledgeOperationV1.DETACH)
    if state == IndexedSourceLifecycleStateV1.SYNCING.value:
        return (KnowledgeOperationV1.DISABLE,)
    if state in {
        IndexedSourceLifecycleStateV1.READY.value,
        IndexedSourceLifecycleStateV1.ACTIVE.value,
        IndexedSourceLifecycleStateV1.ERROR.value,
    }:
        actions = [
            KnowledgeOperationV1.SYNC,
            KnowledgeOperationV1.DISABLE,
            KnowledgeOperationV1.DETACH,
        ]
        if _value(view.sync_state) == "failed":
            actions.insert(1, KnowledgeOperationV1.RETRY_SYNC)
        return tuple(actions)
    return ()


def _live_actions(view: LiveAccessLifecycleViewV1) -> tuple[KnowledgeOperationV1, ...]:
    if view.detached or _value(view.lifecycle_state) == LiveAccessLifecycleStateV1.DETACHED.value:
        return ()
    if _value(view.lifecycle_state) == LiveAccessLifecycleStateV1.DISABLED.value:
        return (KnowledgeOperationV1.ENABLE, KnowledgeOperationV1.DETACH)
    return (KnowledgeOperationV1.DISABLE, KnowledgeOperationV1.DETACH)


class KnowledgeInspectionService:
    """Lists current configuration items and delegates state derivation to lifecycles."""

    def __init__(
        self,
        *,
        configuration_service: WorkspaceKnowledgeConfigurationService,
        indexed_source_lifecycle_service: IndexedSourceLifecycleService,
        live_access_lifecycle_service: LiveAccessLifecycleService,
    ) -> None:
        self._configuration_service = configuration_service
        self._indexed_lifecycle = indexed_source_lifecycle_service
        self._live_lifecycle = live_access_lifecycle_service

    def list_items(self, *, tenant_id: str, workspace_id: str) -> KnowledgeInventoryV1:
        configuration = self._configuration(tenant_id=tenant_id, workspace_id=workspace_id)
        items: list[KnowledgeInventoryItemV1] = []
        try:
            for binding in configuration.indexed_sources:
                view = self._indexed_lifecycle.get(
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    indexed_source_binding_id=binding.indexed_source_binding_id,
                )
                items.append(
                    self._indexed_item(
                        view,
                        display_label=binding.cached_safe_display_label,
                    )
                )
            for binding in configuration.live_access_bindings:
                view = self._live_lifecycle.get(
                    GetLiveAccessCommand(
                        tenant_id=tenant_id,
                        workspace_id=workspace_id,
                        live_access_binding_id=binding.live_access_binding_id,
                    )
                )
                items.append(
                    self._live_item(
                        view,
                        display_label=binding.derived_safe_display_label,
                    )
                )
        except (WorkspaceIndexedSourceLifecycleError, WorkspaceLiveAccessBindingError) as exc:
            raise KnowledgeInventoryError(self._map_error(exc.error_code)) from exc
        except Exception as exc:
            raise KnowledgeInventoryError("knowledge_inventory_unavailable") from exc
        ordered = tuple(
            sorted(
                items,
                key=lambda item: (
                    0 if item.mode is KnowledgeAccessModeV1.INDEXED else 1,
                    (item.display_label or "").casefold(),
                    item.knowledge_item_id,
                ),
            )
        )
        return KnowledgeInventoryV1(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            items=ordered,
            summary=self._summary(ordered),
            updated_at=max(
                (item.updated_at for item in ordered),
                default=configuration.updated_at,
            ),
        )

    def get_item(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        knowledge_item_id: str,
    ) -> KnowledgeInventoryItemV1:
        parsed = _parse_knowledge_item_id(knowledge_item_id)
        if parsed is None:
            raise KnowledgeInventoryError("knowledge_item_not_found")
        mode, binding_id = parsed
        configuration = self._configuration(tenant_id=tenant_id, workspace_id=workspace_id)
        try:
            if mode is KnowledgeAccessModeV1.INDEXED:
                binding = next(
                    item
                    for item in configuration.indexed_sources
                    if item.indexed_source_binding_id == binding_id
                )
                view = self._indexed_lifecycle.get(
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    indexed_source_binding_id=binding_id,
                )
                return self._indexed_item(
                    view,
                    display_label=binding.cached_safe_display_label,
                )
            binding = next(
                item
                for item in configuration.live_access_bindings
                if item.live_access_binding_id == binding_id
            )
            view = self._live_lifecycle.get(
                GetLiveAccessCommand(
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    live_access_binding_id=binding_id,
                )
            )
            return self._live_item(
                view,
                display_label=binding.derived_safe_display_label,
            )
        except StopIteration:
            raise KnowledgeInventoryError("knowledge_item_not_found") from None
        except (WorkspaceIndexedSourceLifecycleError, WorkspaceLiveAccessBindingError) as exc:
            raise KnowledgeInventoryError(self._map_error(exc.error_code)) from exc
        except Exception as exc:
            raise KnowledgeInventoryError("knowledge_inventory_unavailable") from exc

    def _configuration(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> WorkspaceKnowledgeConfigurationV1:
        try:
            configuration = self._configuration_service.get_configuration(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
            )
        except Exception as exc:
            raise KnowledgeInventoryError("knowledge_inventory_unavailable") from exc
        if configuration is None:
            raise KnowledgeInventoryError("knowledge_item_not_found")
        return configuration

    @staticmethod
    def _indexed_item(
        view: IndexedSourceLifecycleViewV1,
        *,
        display_label: str | None,
    ) -> KnowledgeInventoryItemV1:
        return KnowledgeInventoryItemV1(
            tenant_id=view.tenant_id,
            workspace_id=view.workspace_id,
            knowledge_item_id=indexed_knowledge_item_id(view.indexed_source_binding_id),
            mode=KnowledgeAccessModeV1.INDEXED,
            source_id=view.source_id,
            indexed_source_binding_id=view.indexed_source_binding_id,
            knowledge_source_binding_ref=view.knowledge_source_binding_ref,
            display_label=display_label,
            lifecycle_state=_value(view.lifecycle_state),
            enabled=view.enabled,
            detached=view.detached,
            sync_state=_value(view.sync_state),
            last_successful_sync_at=view.last_successful_sync_at,
            last_error_code=view.last_error_code,
            revision=view.lifecycle_revision,
            revision_kind=KnowledgeRevisionKindV1.LIFECYCLE,
            available_actions=_indexed_actions(view),
            updated_at=view.updated_at,
        )

    @staticmethod
    def _live_item(
        view: LiveAccessLifecycleViewV1,
        *,
        display_label: str | None,
    ) -> KnowledgeInventoryItemV1:
        return KnowledgeInventoryItemV1(
            tenant_id=view.tenant_id,
            workspace_id=view.workspace_id,
            knowledge_item_id=live_knowledge_item_id(view.live_access_binding_id),
            mode=KnowledgeAccessModeV1.LIVE,
            live_access_binding_id=view.live_access_binding_id,
            connection_ref=view.connection_ref,
            display_label=display_label,
            lifecycle_state=_value(view.lifecycle_state),
            enabled=view.enabled,
            detached=view.detached,
            runtime_available=view.runtime_available,
            last_error_code=view.last_error_code,
            revision=view.configuration_revision,
            revision_kind=KnowledgeRevisionKindV1.CONFIGURATION,
            available_actions=_live_actions(view),
            updated_at=view.updated_at,
        )

    @staticmethod
    def _summary(items: tuple[KnowledgeInventoryItemV1, ...]) -> KnowledgeInventorySummaryV1:
        return KnowledgeInventorySummaryV1(
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
        )

    @staticmethod
    def _map_error(error_code: str) -> str:
        if error_code in {
            "indexed_source_not_found",
            "live_access_not_found",
            "live_access_binding_not_found",
            "workspace_not_found",
        }:
            return "knowledge_item_not_found"
        if error_code in {"lifecycle_conflict", "configuration_idempotency_conflict"}:
            return "knowledge_operation_conflict"
        return error_code


class KnowledgeOperationsService:
    """Provider-neutral command façade; all mutations remain lifecycle-owned."""

    def __init__(
        self,
        *,
        inspection_service: KnowledgeInspectionService,
        indexed_source_lifecycle_service: IndexedSourceLifecycleService,
        live_access_lifecycle_service: LiveAccessLifecycleService,
    ) -> None:
        self._inspection = inspection_service
        self._indexed_lifecycle = indexed_source_lifecycle_service
        self._live_lifecycle = live_access_lifecycle_service

    async def execute(
        self,
        command: KnowledgeOperationCommandV1,
    ) -> KnowledgeOperationResultV1:
        try:
            item = self._inspection.get_item(
                tenant_id=command.tenant_id,
                workspace_id=command.workspace_id,
                knowledge_item_id=command.knowledge_item_id,
            )
        except KnowledgeInventoryError as exc:
            raise KnowledgeOperationError(exc.error_code) from exc
        if command.expected_revision != item.revision:
            raise KnowledgeOperationError("knowledge_operation_conflict")
        if command.operation not in item.available_actions:
            if command.operation is KnowledgeOperationV1.RETRY_SYNC and command.operation_id is None:
                raise KnowledgeOperationError("knowledge_operation_retry_target_required")
            raise KnowledgeOperationError("knowledge_operation_not_supported")
        if (
            command.operation is KnowledgeOperationV1.RETRY_SYNC
            and not command.operation_id
        ):
            raise KnowledgeOperationError("knowledge_operation_retry_target_required")

        parsed = _parse_knowledge_item_id(command.knowledge_item_id)
        if parsed is None:
            raise KnowledgeOperationError("knowledge_item_not_found")
        mode, binding_id = parsed
        try:
            if mode is KnowledgeAccessModeV1.INDEXED:
                result = self._execute_indexed(command, binding_id)
                operation_id = result.operation_id
                mutation_id = result.mutation_id
            else:
                result = await self._execute_live(command, binding_id)
                operation_id = None
                mutation_id = None
        except (
            WorkspaceIndexedSourceLifecycleError,
            WorkspaceLiveAccessBindingError,
            WorkspaceKnowledgeConfigurationMutationError,
        ) as exc:
            raise KnowledgeOperationError(self._map_operation_error(exc.error_code)) from exc
        except KnowledgeOperationError:
            raise
        except Exception as exc:
            raise KnowledgeOperationError("knowledge_operation_unavailable") from exc

        try:
            item = self._inspection.get_item(
                tenant_id=command.tenant_id,
                workspace_id=command.workspace_id,
                knowledge_item_id=command.knowledge_item_id,
            )
        except KnowledgeInventoryError as exc:
            raise KnowledgeOperationError("knowledge_operation_unavailable") from exc
        return KnowledgeOperationResultV1(
            item=item,
            operation=command.operation,
            operation_id=operation_id,
            mutation_id=mutation_id,
        )

    def _execute_indexed(
        self,
        command: KnowledgeOperationCommandV1,
        binding_id: str,
    ):
        common = {
            "tenant_id": command.tenant_id,
            "workspace_id": command.workspace_id,
            "indexed_source_binding_id": binding_id,
            "expected_revision": command.expected_revision,
            "idempotency_key_hash": command.idempotency_key_hash,
        }
        if command.operation is KnowledgeOperationV1.SYNC:
            return self._indexed_lifecycle.request_sync(IndexedSourceSyncCommand(**common))
        if command.operation is KnowledgeOperationV1.RETRY_SYNC:
            return self._indexed_lifecycle.retry_sync(
                IndexedSourceRetryCommand(**common, operation_id=command.operation_id)
            )
        lifecycle_command = IndexedSourceLifecycleCommand(**common)
        if command.operation is KnowledgeOperationV1.DISABLE:
            return self._indexed_lifecycle.disable(lifecycle_command)
        if command.operation is KnowledgeOperationV1.ENABLE:
            return self._indexed_lifecycle.enable(lifecycle_command)
        if command.operation is KnowledgeOperationV1.DETACH:
            return self._indexed_lifecycle.detach(lifecycle_command)
        if command.operation is KnowledgeOperationV1.RESUME_DETACH:
            return self._indexed_lifecycle.resume_detach(lifecycle_command)
        raise KnowledgeOperationError("knowledge_operation_not_supported")

    async def _execute_live(
        self,
        command: KnowledgeOperationCommandV1,
        binding_id: str,
    ):
        if command.operation not in {
            KnowledgeOperationV1.DISABLE,
            KnowledgeOperationV1.ENABLE,
            KnowledgeOperationV1.DETACH,
        }:
            raise KnowledgeOperationError("knowledge_operation_not_supported")
        common = {
            "tenant_id": command.tenant_id,
            "workspace_id": command.workspace_id,
            "live_access_binding_id": binding_id,
            "expected_revision": command.expected_revision,
            "idempotency_key_hash": command.idempotency_key_hash,
        }
        if command.operation is KnowledgeOperationV1.DISABLE:
            return self._live_lifecycle.disable(
                DisableWorkspaceLiveAccessBindingCommand(**common)
            )
        if command.operation is KnowledgeOperationV1.ENABLE:
            return await self._live_lifecycle.enable(
                EnableWorkspaceLiveAccessBindingCommand(**common)
            )
        return self._live_lifecycle.detach(
            DetachWorkspaceLiveAccessBindingCommand(**common)
        )

    @staticmethod
    def _map_operation_error(error_code: str) -> str:
        if error_code in {
            "indexed_source_not_found",
            "live_access_not_found",
            "live_access_binding_not_found",
            "workspace_not_found",
        }:
            return "knowledge_item_not_found"
        if error_code in {
            "lifecycle_conflict",
            "configuration_revision_conflict",
            "configuration_idempotency_conflict",
        }:
            return "knowledge_operation_conflict"
        if error_code in {
            "sync_not_retryable",
            "sync_in_progress",
            "indexed_source_disabled",
            "indexed_source_detached",
            "live_access_detached",
            "detach_purge_unavailable",
            "publication_in_progress",
            "DETACH_BLOCKED_PUBLICATION_IN_PROGRESS",
        }:
            return "knowledge_operation_invalid_state"
        return error_code


KnowledgeInventoryItem = KnowledgeInventoryItemV1
KnowledgeInventory = KnowledgeInventoryV1
KnowledgeOperationCommand = KnowledgeOperationCommandV1
KnowledgeOperationResult = KnowledgeOperationResultV1
