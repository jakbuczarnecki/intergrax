# © Artur Czarnecki. All rights reserved.

"""Product service for managed workspaces (LKW-PRODUCT-1)."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path

from local_workspace_application.workspaces.ask_repository import WorkspaceAskRepository
from local_workspace_application.workspaces.knowledge_configuration_service import (
    is_workspace_source_product_visible,
)
from local_workspace_application.workspaces.managed_files import ManagedFileCleanupPort
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
from local_workspace_application.workspaces.vector_cleanup import (
    WorkspaceVectorCleanupPort,
)

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(UTC)


class ManagedWorkspaceService:
    def __init__(
        self,
        repository: ManagedWorkspaceRepository,
        *,
        allowlist_roots: frozenset[str] | None = None,
        shadow_roots: tuple[Path, ...] = (),
        ask_repository: WorkspaceAskRepository | None = None,
        vector_cleanup: WorkspaceVectorCleanupPort | None = None,
        managed_file_cleanup: ManagedFileCleanupPort | None = None,
    ) -> None:
        self._repository = repository
        self._allowlist_roots = allowlist_roots
        self._shadow_roots = shadow_roots
        self._ask_repository = ask_repository or WorkspaceAskRepository(
            repository.document_store
        )
        self._vector_cleanup = vector_cleanup
        self._managed_file_cleanup = managed_file_cleanup

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

    def delete_workspace(self, *, tenant_id: str, workspace_id: str) -> bool:
        """
        Delete all LKW-owned state for one tenant/workspace.

        Returns False when the workspace is unknown or cross-tenant (caller → 404).
        Local source files are never touched.

        Ask history policy A: remove workspace-owned Ask runs.
        """
        workspace = self.require_workspace(tenant_id=tenant_id, workspace_id=workspace_id)
        if workspace is None:
            return False

        # Vectors first while document refs still describe scope; idempotent if empty.
        # Best-effort: missing Qdrant collection / unsupported lifecycle must not
        # block deleting workspace metadata (sources/docs/ops/ask/workspace).
        if self._vector_cleanup is not None:
            try:
                self._vector_cleanup.delete_workspace_vectors(
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                )
            except RuntimeError:
                logger.warning("workspace_delete vector_cleanup_failed continuing")

        if self._managed_file_cleanup is not None:
            self._managed_file_cleanup.delete_workspace_files(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
            )

        self._repository.delete_web_url_locators_for_workspace(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )

        self._repository.delete_sources_for_workspace(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )
        self._repository.delete_document_refs_for_workspace(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )
        self._repository.delete_operations_for_workspace(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )
        self._repository.delete_knowledge_inputs_for_workspace(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )
        self._ask_repository.delete_runs_for_workspace(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )
        self._repository.delete_workspace(tenant_id=tenant_id, workspace_id=workspace_id)
        return True

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

    def _committed_knowledge_configuration_revision(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> int:
        head = self._repository.get_knowledge_configuration_head(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )
        return 0 if head is None else head.committed_revision

    def list_sources(self, *, tenant_id: str, workspace_id: str) -> list[WorkspaceSource] | None:
        if self.require_workspace(tenant_id=tenant_id, workspace_id=workspace_id) is None:
            return None
        sources = self._repository.list_sources(tenant_id=tenant_id, workspace_id=workspace_id)
        committed_revision = self._committed_knowledge_configuration_revision(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )
        return [
            source
            for source in sources
            if is_workspace_source_product_visible(
                source,
                committed_configuration_revision=committed_revision,
            )
        ]

    def get_source(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
    ) -> WorkspaceSource | None:
        if self.require_workspace(tenant_id=tenant_id, workspace_id=workspace_id) is None:
            return None
        source = self._repository.get_source(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            source_id=source_id,
        )
        if source is None:
            return None
        committed_revision = self._committed_knowledge_configuration_revision(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )
        if not is_workspace_source_product_visible(
            source,
            committed_configuration_revision=committed_revision,
        ):
            return None
        return source

    def create_sync_operation(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
        allow_concurrent: bool = False,
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
        if source.source_type is WorkspaceSourceType.LOCAL_FOLDER or source.source_type is WorkspaceSourceType.CONNECTED_SOURCE:
            pass
        else:
            raise ValueError("source_sync_unsupported_for_source_type")

        if not allow_concurrent:
            active = self._repository.find_active_sync_operation(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                source_id=source_id,
            )
            if active is not None:
                raise ConcurrentSyncError(active)

        operation = WorkspaceOperation(
            operation_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            source_id=source_id,
            operation_type=WorkspaceOperationType.SOURCE_SYNC,
            status=WorkspaceOperationStatus.QUEUED,
            created_at=_utc_now(),
        )
        return self._repository.put_operation(operation)

    def get_operation(self, *, tenant_id: str, operation_id: str) -> WorkspaceOperation | None:
        return self._repository.get_operation(tenant_id=tenant_id, operation_id=operation_id)

    def list_workspace_operations(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        limit: int = 50,
    ) -> list[WorkspaceOperation]:
        bounded_limit = max(1, min(limit, 100))
        return self._repository.list_operations_for_workspace(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            limit=bounded_limit,
        )

    def recover_running_operations_for_tenant(
        self,
        *,
        tenant_id: str,
        error: str = "interrupted_by_host_restart",
    ) -> int:
        return self._repository.mark_running_operations_failed_for_tenant(
            tenant_id=tenant_id,
            error=error,
        )


class ConcurrentSyncError(Exception):
    """Raised when a sync is already queued or running for the same source."""

    def __init__(self, active: WorkspaceOperation) -> None:
        self.active = active
        super().__init__("sync_already_in_progress")
