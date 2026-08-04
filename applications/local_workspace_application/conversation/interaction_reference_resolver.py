# © Artur Czarnecki. All rights reserved.

"""Fail-closed workspace reference resolution for interaction execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from local_workspace_application.conversation.interaction_models import (
    ConversationPlanningRequest,
    WorkspaceReference,
    WorkspaceReferenceKind,
)
from local_workspace_application.workspaces.conversation_context_models import (
    ConversationExecutionContextV1,
)
from local_workspace_application.workspaces.service import ManagedWorkspaceService


class ConversationReferenceResolutionError(LookupError):
    """Stable reference failure without exposing authoritative state."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class ResolvedWorkspace:
    workspace_id: str


class ConversationInteractionReferenceResolver:
    """Resolves every workspace reference against the current tenant authority."""

    def __init__(
        self,
        *,
        planning_request: ConversationPlanningRequest,
        execution_context: ConversationExecutionContextV1,
        workspace_service: ManagedWorkspaceService,
    ) -> None:
        self._planning_request = planning_request
        self._execution_context = execution_context
        self._workspace_service = workspace_service
        self._current_active_workspace_id: str | None = (
            execution_context.workspace_id.strip() or None
        )

    @property
    def current_active_workspace_id(self) -> str | None:
        return self._current_active_workspace_id

    def set_active_workspace(self, workspace_id: str) -> None:
        self._current_active_workspace_id = workspace_id

    def clear_active_workspace(self) -> None:
        self._current_active_workspace_id = None

    def resolve_workspace(
        self,
        reference: WorkspaceReference,
        *,
        created_workspace_ids: Mapping[str, str] | None = None,
    ) -> ResolvedWorkspace:
        if reference.kind is WorkspaceReferenceKind.active:
            workspace_id = self._current_active_workspace_id
            if workspace_id is None:
                raise ConversationReferenceResolutionError("active_workspace_required")
        elif reference.kind is WorkspaceReferenceKind.name:
            workspace_id = self._resolve_name(reference.value or "")
        elif reference.kind is WorkspaceReferenceKind.ordinal:
            workspace_id = self._resolve_ordinal(reference.value or "")
        elif reference.kind is WorkspaceReferenceKind.created_by_action:
            workspace_id = (created_workspace_ids or {}).get(reference.value or "")
            if workspace_id is None:
                raise ConversationReferenceResolutionError("workspace_not_found")
        else:
            raise ConversationReferenceResolutionError("workspace_not_found")

        self._revalidate_workspace(workspace_id)
        return ResolvedWorkspace(workspace_id=workspace_id)

    def _resolve_name(self, value: str) -> str:
        normalized = value.strip().casefold()
        try:
            workspaces = self._workspace_service.list_workspaces(
                tenant_id=self._execution_context.tenant_id,
            )
        except Exception as exc:  # noqa: BLE001 - safe boundary normalization
            raise ConversationReferenceResolutionError("workspace_not_found") from exc

        matches = [
            workspace
            for workspace in workspaces
            if str(getattr(workspace, "name", "")).strip().casefold() == normalized
        ]
        if not matches:
            raise ConversationReferenceResolutionError("workspace_not_found")
        if len(matches) > 1:
            raise ConversationReferenceResolutionError("workspace_reference_ambiguous")
        return str(matches[0].workspace_id)

    def _resolve_ordinal(self, value: str) -> str:
        if not value.isdigit():
            raise ConversationReferenceResolutionError("workspace_not_found")
        index = int(value) - 1
        snapshot = self._planning_request.available_workspaces
        if index < 0 or index >= len(snapshot):
            raise ConversationReferenceResolutionError("workspace_not_found")
        return snapshot[index].workspace_id

    def _revalidate_workspace(self, workspace_id: str) -> None:
        try:
            workspace = self._workspace_service.get_workspace(
                tenant_id=self._execution_context.tenant_id,
                workspace_id=workspace_id,
            )
        except Exception as exc:  # noqa: BLE001 - safe boundary normalization
            raise ConversationReferenceResolutionError("workspace_not_found") from exc
        if workspace is None:
            raise ConversationReferenceResolutionError("workspace_not_found")
