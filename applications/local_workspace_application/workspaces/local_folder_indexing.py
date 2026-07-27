# © Artur Czarnecki. All rights reserved.

"""Shared local-folder discovery and per-file indexing for sync and candidate intake."""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel, Field

from intergrax.tools.providers.filesystem.allowlist import read_allowlist_roots_from_env
from local_workspace_application.workspaces.discovery import discover_source_files
from local_workspace_application.workspaces.document_indexing import (
    WorkspaceDocumentIndexingError,
    WorkspaceDocumentIndexingService,
)
from local_workspace_application.workspaces.idempotency import (
    content_hash_for_file,
    normalize_source_path,
)
from local_workspace_application.workspaces.models import WorkspaceSource

logger = logging.getLogger(__name__)


class LocalFolderIndexingResult(BaseModel):
    files_discovered: int = Field(default=0, ge=0)
    files_processed: int = Field(default=0, ge=0)
    files_failed: int = Field(default=0, ge=0)
    documents_indexed: int = Field(default=0, ge=0)
    documents_unchanged: int = Field(default=0, ge=0)


class LocalFolderIndexingService:
    """Indexes all discoverable files under a LOCAL_FOLDER WorkspaceSource."""

    def __init__(
        self,
        indexing_service: WorkspaceDocumentIndexingService,
        *,
        allowlist_roots: frozenset[str] | None = None,
    ) -> None:
        self._indexing_service = indexing_service
        self._allowlist_roots = allowlist_roots

    async def index_source(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source: WorkspaceSource,
        operation_id: str,
    ) -> LocalFolderIndexingResult:
        roots = self._allowlist_roots
        if roots is None:
            roots = read_allowlist_roots_from_env()

        root = Path(source.path)
        discovered, skipped = discover_source_files(
            root,
            recursive=source.recursive,
            allowlist_roots=roots,
        )
        files_discovered = len(discovered) + len(skipped)
        files_processed = 0
        files_failed = len(skipped)
        documents_indexed = 0
        documents_unchanged = 0

        for path in discovered:
            files_processed += 1
            normalized = normalize_source_path(path)
            digest = content_hash_for_file(path)
            try:
                index_result = await self._indexing_service.index_one(
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    source_id=source.source_id,
                    operation_id=operation_id,
                    physical_path=path,
                    logical_source_path=normalized,
                    safe_file_name=path.name,
                    content_hash=digest,
                )
            except WorkspaceDocumentIndexingError as exc:
                files_failed += 1
                logger.warning(
                    "local_folder_indexing_file_failed operation_id=%s reason=%s",
                    operation_id,
                    exc.error_code,
                )
                continue

            if index_result.unchanged:
                documents_unchanged += 1
                continue
            if index_result.indexed:
                documents_indexed += 1
            else:
                files_failed += 1

        return LocalFolderIndexingResult(
            files_discovered=files_discovered,
            files_processed=files_processed,
            files_failed=files_failed,
            documents_indexed=documents_indexed,
            documents_unchanged=documents_unchanged,
        )
