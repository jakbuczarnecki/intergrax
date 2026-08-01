# © Artur Czarnecki. All rights reserved.

"""Committed Workspace Knowledge Configuration read projection (LKW-KNOWLEDGE-ACCESS-1B-3)."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from local_workspace_application.workspaces.knowledge_configuration_models import (
    WorkspaceConnectionAttachment,
    WorkspaceIndexedSourceBinding,
    WorkspaceKnowledgeConfigurationV1,
    WorkspaceLiveAccessBinding,
    WorkspaceQueryPolicy,
)
from local_workspace_application.workspaces.models import Workspace, WorkspaceSource
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository


class WorkspaceKnowledgeConfigurationServiceError(RuntimeError):
    def __init__(self, error_code: str) -> None:
        super().__init__(error_code)
        self.error_code = error_code


class WorkspaceKnowledgeWorkspaceLookupPort(Protocol):
    def require_workspace(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> Workspace | None:
        ...


def is_workspace_source_product_visible(
    source: WorkspaceSource,
    *,
    committed_configuration_revision: int,
) -> bool:
    visibility_revision = source.knowledge_configuration_visibility_revision
    if visibility_revision is None:
        return True
    return committed_configuration_revision >= visibility_revision


def _logical_committed_revision(head_committed_revision: int | None) -> int:
    if head_committed_revision is None:
        return 0
    return head_committed_revision


def _select_highest_committed_attachment(
    versions: list[WorkspaceConnectionAttachment],
    committed_revision: int,
) -> list[WorkspaceConnectionAttachment]:
    best_by_id: dict[str, WorkspaceConnectionAttachment] = {}
    for version in versions:
        if version.effective_revision > committed_revision:
            continue
        current = best_by_id.get(version.attachment_id)
        if current is None or version.effective_revision > current.effective_revision:
            best_by_id[version.attachment_id] = version
    return list(best_by_id.values())


def _select_highest_committed_indexed_source(
    versions: list[WorkspaceIndexedSourceBinding],
    committed_revision: int,
) -> list[WorkspaceIndexedSourceBinding]:
    best_by_id: dict[str, WorkspaceIndexedSourceBinding] = {}
    for version in versions:
        if version.effective_revision > committed_revision:
            continue
        current = best_by_id.get(version.indexed_source_binding_id)
        if current is None or version.effective_revision > current.effective_revision:
            best_by_id[version.indexed_source_binding_id] = version
    return list(best_by_id.values())


def _select_highest_committed_live_access(
    versions: list[WorkspaceLiveAccessBinding],
    committed_revision: int,
) -> list[WorkspaceLiveAccessBinding]:
    best_by_id: dict[str, WorkspaceLiveAccessBinding] = {}
    for version in versions:
        if version.effective_revision > committed_revision:
            continue
        current = best_by_id.get(version.live_access_binding_id)
        if current is None or version.effective_revision > current.effective_revision:
            best_by_id[version.live_access_binding_id] = version
    return list(best_by_id.values())


def _select_highest_committed_query_policy(
    versions: list[WorkspaceQueryPolicy],
    committed_revision: int,
) -> WorkspaceQueryPolicy | None:
    best: WorkspaceQueryPolicy | None = None
    for version in versions:
        if version.effective_revision > committed_revision:
            continue
        if best is None or version.effective_revision > best.effective_revision:
            best = version
    return best


def _projection_updated_at(
    workspace: Workspace,
    *,
    connection_attachments: tuple[WorkspaceConnectionAttachment, ...],
    indexed_sources: tuple[WorkspaceIndexedSourceBinding, ...],
    live_access_bindings: tuple[WorkspaceLiveAccessBinding, ...],
    query_policy: WorkspaceQueryPolicy | None,
) -> datetime:
    timestamps: list[datetime] = []
    timestamps.extend(item.updated_at for item in connection_attachments)
    timestamps.extend(item.updated_at for item in indexed_sources)
    timestamps.extend(item.updated_at for item in live_access_bindings)
    if query_policy is not None:
        timestamps.append(query_policy.updated_at)
    if timestamps:
        return max(timestamps)
    return workspace.updated_at


class WorkspaceKnowledgeConfigurationService:
    def __init__(
        self,
        repository: ManagedWorkspaceRepository,
        workspace_lookup: WorkspaceKnowledgeWorkspaceLookupPort,
    ) -> None:
        self._repository = repository
        self._workspace_lookup = workspace_lookup

    def get_configuration(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> WorkspaceKnowledgeConfigurationV1 | None:
        workspace = self._workspace_lookup.require_workspace(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )
        if workspace is None:
            return None

        for attempt in range(2):
            projection, committed_before, committed_after = self._project_once(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                workspace=workspace,
            )
            if committed_after == committed_before:
                return projection
            if attempt == 1:
                raise WorkspaceKnowledgeConfigurationServiceError(
                    "configuration_projection_unstable"
                )

        raise WorkspaceKnowledgeConfigurationServiceError(
            "configuration_projection_unstable"
        )

    def _project_once(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        workspace: Workspace,
    ) -> tuple[WorkspaceKnowledgeConfigurationV1, int, int]:
        head_before = self._repository.get_knowledge_configuration_head(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )
        committed_revision = _logical_committed_revision(
            None if head_before is None else head_before.committed_revision
        )

        attachment_versions = self._repository.list_knowledge_connection_attachment_versions(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )
        indexed_source_versions = self._repository.list_knowledge_indexed_source_versions(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )
        live_access_versions = self._repository.list_knowledge_live_access_versions(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )
        query_policy_versions = self._repository.list_knowledge_query_policy_versions(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )

        selected_attachments = _select_highest_committed_attachment(
            attachment_versions,
            committed_revision,
        )
        selected_indexed_sources = _select_highest_committed_indexed_source(
            indexed_source_versions,
            committed_revision,
        )
        selected_live_access = _select_highest_committed_live_access(
            live_access_versions,
            committed_revision,
        )
        selected_query_policy = _select_highest_committed_query_policy(
            query_policy_versions,
            committed_revision,
        )

        connection_attachments = tuple(
            sorted(selected_attachments, key=lambda item: (item.connection_ref, item.attachment_id))
        )
        indexed_sources = tuple(
            sorted(selected_indexed_sources, key=lambda item: item.indexed_source_binding_id)
        )
        live_access_bindings = tuple(
            sorted(selected_live_access, key=lambda item: item.live_access_binding_id)
        )

        projection = WorkspaceKnowledgeConfigurationV1(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            configuration_revision=committed_revision,
            connection_attachments=connection_attachments,
            indexed_sources=indexed_sources,
            live_access_bindings=live_access_bindings,
            query_policy=selected_query_policy,
            updated_at=_projection_updated_at(
                workspace,
                connection_attachments=connection_attachments,
                indexed_sources=indexed_sources,
                live_access_bindings=live_access_bindings,
                query_policy=selected_query_policy,
            ),
        )

        head_after = self._repository.get_knowledge_configuration_head(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )
        committed_after = _logical_committed_revision(
            None if head_after is None else head_after.committed_revision
        )
        return projection, committed_revision, committed_after
