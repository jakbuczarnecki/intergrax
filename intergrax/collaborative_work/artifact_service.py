# © Artur Czarnecki. All rights reserved.

"""Authoritative WorkArtifact publication service (MP-3C).

Owns WorkArtifact + WorkArtifactVersion publication with fresh MP-1 authority
enforcement and MP-3B atomic publication repository delegation.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Final

from intergrax.collaborative_work._authority_enforcement import _require_collaborative_allow
from intergrax.collaborative_work.enforcement_gate import CollaborativeWorkEnforcementGate
from intergrax.collaborative_work.repository import (
    ArtifactPublicationRepository,
    CreateArtifactWithInitialVersionCommand,
    PublishWorkArtifactVersionCommand,
    PublishedWorkArtifactVersion,
    WorkArtifactNotFound,
    WorkArtifactRepository,
    WorkItemNotFound,
    WorkItemRepository,
)
from intergrax.contracts.collaborative_work import (
    CreateWorkArtifactRequest,
    PublishWorkArtifactVersionRequest,
    work_item_resource_scope,
)

TRUSTED_OPERATION_WORK_ARTIFACT_CREATE: Final = "collaborative_work.work_artifact.create"
TRUSTED_OPERATION_WORK_ARTIFACT_PUBLISH: Final = "collaborative_work.work_artifact.publish"


class CollaborativeWorkArtifactService:
    """Authoritative WorkArtifact publication boundary."""

    def __init__(
        self,
        *,
        work_item_repository: WorkItemRepository,
        work_artifact_repository: WorkArtifactRepository,
        artifact_publication_repository: ArtifactPublicationRepository,
        enforcement_gate: CollaborativeWorkEnforcementGate,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._work_item_repository = work_item_repository
        self._work_artifact_repository = work_artifact_repository
        self._artifact_publication_repository = artifact_publication_repository
        self._enforcement_gate = enforcement_gate
        self._clock = clock or (lambda: datetime.now(UTC))

    def create_artifact(self, request: CreateWorkArtifactRequest) -> PublishedWorkArtifactVersion:
        work_item = self._work_item_repository.get(
            tenant_id=request.tenant_id,
            workspace_id=request.workspace_id,
            work_item_id=request.work_item_id,
        )
        if work_item is None:
            raise WorkItemNotFound("work item was not found")
        resource_scope = work_item_resource_scope(work_item_id=request.work_item_id)
        _require_collaborative_allow(
            enforcement_gate=self._enforcement_gate,
            operation_id=TRUSTED_OPERATION_WORK_ARTIFACT_CREATE,
            request=request,
            resource_scope=resource_scope,
        )
        now = self._require_timezone_aware(self._clock())
        return self._artifact_publication_repository.create_artifact_with_initial_version(
            CreateArtifactWithInitialVersionCommand(
                tenant_id=request.tenant_id,
                workspace_id=request.workspace_id,
                work_item_id=request.work_item_id,
                work_artifact_id=request.work_artifact_id,
                work_artifact_version_id=request.work_artifact_version_id,
                created_by_principal_id=request.acting_principal_id,
                published_by_principal_id=request.acting_principal_id,
                content_ref=request.content_ref,
                artifact_created_at=now,
                artifact_updated_at=now,
                version_created_at=now,
                version_published_at=now,
                execution=None,
                idempotency_key=request.idempotency_key,
            ),
        )

    def publish_version(
        self,
        request: PublishWorkArtifactVersionRequest,
    ) -> PublishedWorkArtifactVersion:
        artifact = self._work_artifact_repository.get(
            tenant_id=request.tenant_id,
            workspace_id=request.workspace_id,
            work_artifact_id=request.work_artifact_id,
        )
        if artifact is None or artifact.work_item_id != request.work_item_id:
            raise WorkArtifactNotFound("work artifact was not found")
        resource_scope = work_item_resource_scope(work_item_id=artifact.work_item_id)
        _require_collaborative_allow(
            enforcement_gate=self._enforcement_gate,
            operation_id=TRUSTED_OPERATION_WORK_ARTIFACT_PUBLISH,
            request=request,
            resource_scope=resource_scope,
        )
        now = self._require_timezone_aware(self._clock())
        return self._artifact_publication_repository.publish_version(
            PublishWorkArtifactVersionCommand(
                tenant_id=request.tenant_id,
                workspace_id=request.workspace_id,
                work_item_id=artifact.work_item_id,
                work_artifact_id=request.work_artifact_id,
                work_artifact_version_id=request.work_artifact_version_id,
                expected_revision=request.expected_revision,
                created_by_principal_id=request.acting_principal_id,
                published_by_principal_id=request.acting_principal_id,
                content_ref=request.content_ref,
                created_at=now,
                published_at=now,
                artifact_updated_at=now,
                execution=None,
                idempotency_key=request.idempotency_key,
            ),
        )

    @staticmethod
    def _require_timezone_aware(value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("clock must return timezone-aware datetimes")
        return value
