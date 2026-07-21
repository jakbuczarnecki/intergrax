# © Artur Czarnecki. All rights reserved.

"""Product service for managed workspaces (LKW-PRODUCT-1)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

from local_workspace_application.workspaces.models import (
    Workspace,
    WorkspaceOperation,
    WorkspaceOperationStatus,
    WorkspaceOperationType,
    WorkspaceSource,
    WorkspaceSourceStatus,
    WorkspaceSourceType,
    WorkspaceStatus,
)
from local_workspace_application.workspaces.path_policy import (
    SourcePathPolicyError,
    validate_local_folder_source_path,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository


def _utc_now() -> datetime:
    return datetime.now(UTC)


class ManagedWorkspaceService:
    def __init__(
        self,
        repository: ManagedWorkspaceRepository,
        *,
        allowlist_roots: frozenset[str] | None = None,
        shadow_roots: tuple[Path, ...] = (),
    ) -> None:
        self._repository = repository
        self._allowlist_roots = allowlist_roots
        self._shadow_roots = shadow_roots

    @property
    def repository(self) -> ManagedWorkspaceRepository:
        return self._repository

    def create_workspace(
        self,
        *,
        tenant_id: str,
        name: str,
        description: str = "",
    ) -> Workspace:
        now = _utc_now()
        workspace = Workspace(
            workspace_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            name=name.strip(),
            description=description.strip(),
            status=WorkspaceStatus.ACTIVE,
            created_at=now,
            updated_at=now,
        )
        return self._repository.put_workspace(workspace)

    def list_workspaces(self, *, tenant_id: str) -> list[Workspace]:
        return self._repository.list_workspaces(tenant_id=tenant_id)

    def get_workspace(self, *, tenant_id: str, workspace_id: str) -> Workspace | None:
        return self._repository.get_workspace(tenant_id=tenant_id, workspace_id=workspace_id)

    def require_workspace(self, *, tenant_id: str, workspace_id: str) -> Workspace | None:
        """Return workspace for tenant or None (fail-closed 404 semantics)."""
        return self.get_workspace(tenant_id=tenant_id, workspace_id=workspace_id)

    def register_local_folder_source(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        path: str,
        recursive: bool = True,
    ) -> WorkspaceSource:
        workspace = self.require_workspace(tenant_id=tenant_id, workspace_id=workspace_id)
        if workspace is None:
            raise LookupError("workspace_not_found")

        try:
            resolved = validate_local_folder_source_path(
                path,
                allowlist_roots=self._allowlist_roots,
                shadow_roots=self._shadow_roots,
            )
        except SourcePathPolicyError as exc:
            raise ValueError(exc.reason) from exc

        now = _utc_now()
        source = WorkspaceSource(
            source_id=str(uuid.uuid4()),
            workspace_id=workspace_id,
            tenant_id=tenant_id,
            source_type=WorkspaceSourceType.LOCAL_FOLDER,
            path=str(resolved),
            recursive=recursive,
            status=WorkspaceSourceStatus.REGISTERED,
            created_at=now,
            last_sync_at=None,
        )
        return self._repository.put_source(source)

    def list_sources(self, *, tenant_id: str, workspace_id: str) -> list[WorkspaceSource] | None:
        if self.require_workspace(tenant_id=tenant_id, workspace_id=workspace_id) is None:
            return None
        return self._repository.list_sources(tenant_id=tenant_id, workspace_id=workspace_id)

    def get_source(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
    ) -> WorkspaceSource | None:
        if self.require_workspace(tenant_id=tenant_id, workspace_id=workspace_id) is None:
            return None
        return self._repository.get_source(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            source_id=source_id,
        )

    def create_sync_operation(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
    ) -> WorkspaceOperation:
        workspace = self.require_workspace(tenant_id=tenant_id, workspace_id=workspace_id)
        if workspace is None:
            raise LookupError("workspace_not_found")
        source = self._repository.get_source(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            source_id=source_id,
        )
        if source is None:
            raise LookupError("source_not_found")

        operation = WorkspaceOperation(
            operation_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            source_id=source_id,
            operation_type=WorkspaceOperationType.SOURCE_SYNC,
            status=WorkspaceOperationStatus.QUEUED,
        )
        return self._repository.put_operation(operation)

    def get_operation(self, *, tenant_id: str, operation_id: str) -> WorkspaceOperation | None:
        return self._repository.get_operation(tenant_id=tenant_id, operation_id=operation_id)
