# © Artur Czarnecki. All rights reserved.

"""Repository-backed Collaborative Work scoped reference catalog (MP-5F-B4)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.collaborative_work.repository import (
    CollaborativeWorkScopedReferenceCatalog,
    CollaborativeWorkScopedReferenceListing,
    CollaborativeWorkScopedReferenceQuery,
    WorkArtifactRepository,
    WorkArtifactVersionRepository,
    WorkItemRepository,
)
from intergrax.contracts.collaborative_work import WorkArtifact, WorkArtifactVersion, WorkItem

_KIND_WORK_ITEM = "work_item"
_KIND_WORK_ARTIFACT = "work_artifact"
_KIND_WORK_ARTIFACT_VERSION = "work_artifact_version"

__all__ = ["RepositoryBackedCollaborativeWorkReferenceCatalog"]


def _listing_sort_key(
    row: CollaborativeWorkScopedReferenceListing,
) -> tuple[str, str, str, str]:
    return (
        row.entity_kind,
        row.work_item_id,
        row.work_artifact_id or "",
        row.work_artifact_version_id or "",
    )


def _work_item_matches_scope(item: WorkItem, query: CollaborativeWorkScopedReferenceQuery) -> bool:
    if item.tenant_id != query.tenant_id or item.workspace_id != query.workspace_id:
        return False
    if query.work_item_id is not None and item.work_item_id != query.work_item_id:
        return False
    return True


def _artifact_matches_scope(
    artifact: WorkArtifact,
    query: CollaborativeWorkScopedReferenceQuery,
) -> bool:
    if artifact.tenant_id != query.tenant_id or artifact.workspace_id != query.workspace_id:
        return False
    if query.work_item_id is not None and artifact.work_item_id != query.work_item_id:
        return False
    if query.work_artifact_id is not None and artifact.work_artifact_id != query.work_artifact_id:
        return False
    return True


def _version_matches_scope(
    version: WorkArtifactVersion,
    query: CollaborativeWorkScopedReferenceQuery,
    *,
    artifact: WorkArtifact | None,
) -> bool:
    if version.tenant_id != query.tenant_id or version.workspace_id != query.workspace_id:
        return False
    if query.work_item_id is not None and version.work_item_id != query.work_item_id:
        return False
    if query.work_artifact_id is not None and version.work_artifact_id != query.work_artifact_id:
        return False
    if query.work_artifact_version_id is not None and (
        version.work_artifact_version_id != query.work_artifact_version_id
    ):
        return False
    if artifact is not None:
        if version.work_artifact_id != artifact.work_artifact_id:
            return False
        if version.work_item_id != artifact.work_item_id:
            return False
    return True


@dataclass
class RepositoryBackedCollaborativeWorkReferenceCatalog:
    """Enumerate canonical metadata via public repository ports only."""

    work_item_repository: WorkItemRepository
    work_artifact_repository: WorkArtifactRepository
    work_artifact_version_repository: WorkArtifactVersionRepository

    def list_scoped_references(
        self,
        query: CollaborativeWorkScopedReferenceQuery,
    ) -> tuple[CollaborativeWorkScopedReferenceListing, ...]:
        rows: list[CollaborativeWorkScopedReferenceListing] = []

        if query.work_artifact_version_id is not None:
            rows.extend(self._rows_for_exact_version(query))
        elif query.work_artifact_id is not None:
            rows.extend(self._rows_for_exact_artifact(query))
        elif query.work_item_id is not None:
            rows.extend(self._rows_for_work_item(query))
        else:
            rows.extend(self._rows_for_workspace(query))

        rows.sort(key=_listing_sort_key)
        return tuple(rows[: query.limit])

    def _rows_for_exact_version(
        self,
        query: CollaborativeWorkScopedReferenceQuery,
    ) -> list[CollaborativeWorkScopedReferenceListing]:
        version = self.work_artifact_version_repository.get(
            tenant_id=query.tenant_id,
            workspace_id=query.workspace_id,
            work_artifact_version_id=query.work_artifact_version_id or "",
        )
        if version is None:
            return []
        artifact = self.work_artifact_repository.get(
            tenant_id=query.tenant_id,
            workspace_id=query.workspace_id,
            work_artifact_id=version.work_artifact_id,
        )
        if artifact is None or not _version_matches_scope(version, query, artifact=artifact):
            return []
        if not _artifact_matches_scope(artifact, query):
            return []
        return self._project_version_chain(query, artifact=artifact, versions=(version,))

    def _rows_for_exact_artifact(
        self,
        query: CollaborativeWorkScopedReferenceQuery,
    ) -> list[CollaborativeWorkScopedReferenceListing]:
        artifact = self.work_artifact_repository.get(
            tenant_id=query.tenant_id,
            workspace_id=query.workspace_id,
            work_artifact_id=query.work_artifact_id or "",
        )
        if artifact is None or not _artifact_matches_scope(artifact, query):
            return []
        versions = self.work_artifact_version_repository.list_for_artifact(
            tenant_id=query.tenant_id,
            workspace_id=query.workspace_id,
            work_artifact_id=artifact.work_artifact_id,
        )
        return self._project_version_chain(query, artifact=artifact, versions=versions)

    def _rows_for_work_item(
        self,
        query: CollaborativeWorkScopedReferenceQuery,
    ) -> list[CollaborativeWorkScopedReferenceListing]:
        work_item = self.work_item_repository.get(
            tenant_id=query.tenant_id,
            workspace_id=query.workspace_id,
            work_item_id=query.work_item_id or "",
        )
        if work_item is None or not _work_item_matches_scope(work_item, query):
            return []
        rows: list[CollaborativeWorkScopedReferenceListing] = []
        if _KIND_WORK_ITEM in query.entity_kinds:
            rows.append(
                CollaborativeWorkScopedReferenceListing(
                    entity_kind=_KIND_WORK_ITEM,
                    tenant_id=work_item.tenant_id,
                    workspace_id=work_item.workspace_id,
                    work_item_id=work_item.work_item_id,
                    work_item_state=work_item.state,
                )
            )
        if _KIND_WORK_ARTIFACT not in query.entity_kinds and (
            _KIND_WORK_ARTIFACT_VERSION not in query.entity_kinds
        ):
            return rows
        artifacts = self.work_artifact_repository.list_for_work_item(
            tenant_id=query.tenant_id,
            workspace_id=query.workspace_id,
            work_item_id=work_item.work_item_id,
        )
        for artifact in artifacts:
            if not _artifact_matches_scope(artifact, query):
                continue
            rows.extend(
                self._project_version_chain(
                    query,
                    artifact=artifact,
                    versions=self.work_artifact_version_repository.list_for_artifact(
                        tenant_id=query.tenant_id,
                        workspace_id=query.workspace_id,
                        work_artifact_id=artifact.work_artifact_id,
                    ),
                )
            )
        return rows

    def _rows_for_workspace(
        self,
        query: CollaborativeWorkScopedReferenceQuery,
    ) -> list[CollaborativeWorkScopedReferenceListing]:
        rows: list[CollaborativeWorkScopedReferenceListing] = []
        if _KIND_WORK_ITEM in query.entity_kinds:
            for work_item in self.work_item_repository.list_for_workspace(
                tenant_id=query.tenant_id,
                workspace_id=query.workspace_id,
            ):
                if not _work_item_matches_scope(work_item, query):
                    continue
                rows.append(
                    CollaborativeWorkScopedReferenceListing(
                        entity_kind=_KIND_WORK_ITEM,
                        tenant_id=work_item.tenant_id,
                        workspace_id=work_item.workspace_id,
                        work_item_id=work_item.work_item_id,
                        work_item_state=work_item.state,
                    )
                )
        artifact_kinds = query.entity_kinds - {_KIND_WORK_ITEM}
        if artifact_kinds:
            for work_item in self.work_item_repository.list_for_workspace(
                tenant_id=query.tenant_id,
                workspace_id=query.workspace_id,
            ):
                scoped = CollaborativeWorkScopedReferenceQuery(
                    tenant_id=query.tenant_id,
                    workspace_id=query.workspace_id,
                    limit=query.limit,
                    entity_kinds=artifact_kinds,
                    include_historical=query.include_historical,
                    work_item_id=work_item.work_item_id,
                    work_artifact_id=query.work_artifact_id,
                    work_artifact_version_id=query.work_artifact_version_id,
                )
                rows.extend(self._rows_for_work_item(scoped))
        return rows

    def _project_version_chain(
        self,
        query: CollaborativeWorkScopedReferenceQuery,
        *,
        artifact: WorkArtifact,
        versions: tuple[WorkArtifactVersion, ...],
    ) -> list[CollaborativeWorkScopedReferenceListing]:
        rows: list[CollaborativeWorkScopedReferenceListing] = []
        if _KIND_WORK_ARTIFACT in query.entity_kinds:
            rows.append(
                CollaborativeWorkScopedReferenceListing(
                    entity_kind=_KIND_WORK_ARTIFACT,
                    tenant_id=artifact.tenant_id,
                    workspace_id=artifact.workspace_id,
                    work_item_id=artifact.work_item_id,
                    work_artifact_id=artifact.work_artifact_id,
                    current_version_id=artifact.current_version_id,
                )
            )
        if _KIND_WORK_ARTIFACT_VERSION not in query.entity_kinds:
            return rows
        selected_versions = versions
        if not query.include_historical:
            selected_versions = tuple(
                version
                for version in versions
                if version.work_artifact_version_id == artifact.current_version_id
            )
        for version in selected_versions:
            if not _version_matches_scope(version, query, artifact=artifact):
                continue
            rows.append(
                CollaborativeWorkScopedReferenceListing(
                    entity_kind=_KIND_WORK_ARTIFACT_VERSION,
                    tenant_id=version.tenant_id,
                    workspace_id=version.workspace_id,
                    work_item_id=version.work_item_id,
                    work_artifact_id=version.work_artifact_id,
                    work_artifact_version_id=version.work_artifact_version_id,
                )
            )
        return rows
