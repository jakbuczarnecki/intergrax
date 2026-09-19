# © Artur Czarnecki. All rights reserved.

"""MP-6F — source service decorators publishing through ``CollaborativeActivityPublicationPort``."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from intergrax.collaborative_work.artifact_service import CollaborativeWorkArtifactService
from intergrax.collaborative_work.collaborative_activity_source_mapping import (
    CollaborativeWorkActivitySourceMapper,
    ContextViewActivitySourceMapper,
)
from intergrax.collaborative_work.decision_binding_service import CollaborativeDecisionBindingService
from intergrax.collaborative_work.service import CollaborativeWorkService
from intergrax.contracts.collaborative_activity import (
    CollaborativeActivityPublication,
    CollaborativeActivityPublicationPort,
)
from intergrax.contracts.decision_record import DecisionProposalRef
from intergrax.contracts.collaborative_decision_binding import (
    CollaborativeDecisionBinding,
    CreateCollaborativeDecisionBindingRequest,
)
from intergrax.contracts.collaborative_work import (
    Assignment,
    CreateAssignmentRequest,
    CreateWorkArtifactFromExecutionRequest,
    CreateWorkArtifactRequest,
    CreateWorkItemRequest,
    PublishWorkArtifactVersionFromExecutionRequest,
    PublishWorkArtifactVersionRequest,
    TransitionAssignmentRequest,
    TransitionWorkItemRequest,
    WorkItem,
)
from intergrax.contracts.context_view import ContextView
from intergrax.contracts.context_view_composition import ContextViewCompositionRequest
from intergrax.collaborative_work.repository import PublishedWorkArtifactVersion

_LOGGER = logging.getLogger(__name__)


class CollaborativeActivityPublicationFailurePolicy(StrEnum):
    """Explicit semantics when source mutation succeeded but activity publication fails."""

    RAISE = "raise"
    LOG_AND_CONTINUE = "log_and_continue"


@dataclass(frozen=True, slots=True)
class CollaborativeActivitySourcePublicationSideEffect:
    """Stateless publication orchestration — depends on ``CollaborativeActivityPublicationPort`` only."""

    publication_port: CollaborativeActivityPublicationPort
    failure_policy: CollaborativeActivityPublicationFailurePolicy = (
        CollaborativeActivityPublicationFailurePolicy.RAISE
    )

    def publish_after_source_success(
        self,
        publication: CollaborativeActivityPublication,
    ) -> None:
        try:
            self.publication_port.publish(publication)
        except Exception:
            if self.failure_policy is CollaborativeActivityPublicationFailurePolicy.RAISE:
                raise
            _LOGGER.exception(
                "collaborative activity publication failed after authoritative source success",
            )


class CollaborativeWorkServiceWithActivityPublication:
    """Decorator — preserves ``CollaborativeWorkService`` semantics and results."""

    def __init__(
        self,
        *,
        inner: CollaborativeWorkService,
        side_effect: CollaborativeActivitySourcePublicationSideEffect,
        mapper: CollaborativeWorkActivitySourceMapper,
    ) -> None:
        self._inner = inner
        self._side_effect = side_effect
        self._mapper = mapper

    def create_work_item(self, request: CreateWorkItemRequest) -> WorkItem:
        work_item = self._inner.create_work_item(request)
        publication = self._mapper.map_work_item_created(request=request, work_item=work_item)
        self._side_effect.publish_after_source_success(publication)
        return work_item

    def transition_work_item(self, request: TransitionWorkItemRequest) -> WorkItem:
        work_item = self._inner.transition_work_item(request)
        publication = self._mapper.map_work_item_state_changed(
            request=request,
            work_item=work_item,
        )
        self._side_effect.publish_after_source_success(publication)
        return work_item

    def create_assignment(self, request: CreateAssignmentRequest) -> Assignment:
        assignment = self._inner.create_assignment(request)
        publication = self._mapper.map_assignment_created(
            request=request,
            assignment=assignment,
        )
        self._side_effect.publish_after_source_success(publication)
        return assignment

    def transition_assignment(self, request: TransitionAssignmentRequest) -> Assignment:
        assignment = self._inner.transition_assignment(request)
        publication = self._mapper.map_assignment_state_changed(
            request=request,
            assignment=assignment,
        )
        self._side_effect.publish_after_source_success(publication)
        return assignment


class CollaborativeWorkArtifactServiceWithActivityPublication:
    def __init__(
        self,
        *,
        inner: CollaborativeWorkArtifactService,
        side_effect: CollaborativeActivitySourcePublicationSideEffect,
        mapper: CollaborativeWorkActivitySourceMapper,
    ) -> None:
        self._inner = inner
        self._side_effect = side_effect
        self._mapper = mapper

    def create_artifact(self, request: CreateWorkArtifactRequest) -> PublishedWorkArtifactVersion:
        published = self._inner.create_artifact(request)
        publication = self._mapper.map_work_artifact_created(request=request, published=published)
        self._side_effect.publish_after_source_success(publication)
        return published

    def create_artifact_from_execution(
        self,
        request: CreateWorkArtifactFromExecutionRequest,
    ) -> PublishedWorkArtifactVersion:
        published = self._inner.create_artifact_from_execution(request)
        publication = self._mapper.map_work_artifact_created(request=request, published=published)
        self._side_effect.publish_after_source_success(publication)
        return published

    def publish_version(
        self,
        request: PublishWorkArtifactVersionRequest,
    ) -> PublishedWorkArtifactVersion:
        published = self._inner.publish_version(request)
        publication = self._mapper.map_work_artifact_version_published(
            request=request,
            published=published,
        )
        self._side_effect.publish_after_source_success(publication)
        return published

    def publish_version_from_execution(
        self,
        request: PublishWorkArtifactVersionFromExecutionRequest,
    ) -> PublishedWorkArtifactVersion:
        published = self._inner.publish_version_from_execution(request)
        publication = self._mapper.map_work_artifact_version_published(
            request=request,
            published=published,
        )
        self._side_effect.publish_after_source_success(publication)
        return published


class CollaborativeDecisionBindingServiceWithActivityPublication:
    def __init__(
        self,
        *,
        inner: CollaborativeDecisionBindingService,
        side_effect: CollaborativeActivitySourcePublicationSideEffect,
        mapper: CollaborativeWorkActivitySourceMapper,
    ) -> None:
        self._inner = inner
        self._side_effect = side_effect
        self._mapper = mapper

    def create_binding(
        self,
        request: CreateCollaborativeDecisionBindingRequest,
    ) -> CollaborativeDecisionBinding:
        binding = self._inner.create_binding(request)
        publication = self._mapper.map_decision_binding_created(
            request=request,
            binding=binding,
        )
        self._side_effect.publish_after_source_success(publication)
        return binding

    def get_binding(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        binding_id: str,
    ) -> CollaborativeDecisionBinding | None:
        return self._inner.get_binding(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            binding_id=binding_id,
        )

    def list_bindings_for_work_item(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        work_item_id: str,
    ) -> tuple[CollaborativeDecisionBinding, ...]:
        return self._inner.list_bindings_for_work_item(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            work_item_id=work_item_id,
        )

    def list_bindings_for_decision_proposal(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        decision_proposal: DecisionProposalRef,
    ) -> tuple[CollaborativeDecisionBinding, ...]:
        return self._inner.list_bindings_for_decision_proposal(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            decision_proposal=decision_proposal,
        )


class ContextViewComposerWithActivityPublication:
    """Wraps any composer exposing ``compose`` — typically ``DefaultContextViewComposer``."""

    def __init__(
        self,
        *,
        inner: object,
        side_effect: CollaborativeActivitySourcePublicationSideEffect,
        mapper: ContextViewActivitySourceMapper,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._inner = inner
        self._side_effect = side_effect
        self._mapper = mapper
        self._clock = clock or (lambda: datetime.now(UTC))

    def compose(self, composition_request: ContextViewCompositionRequest) -> ContextView:
        view = self._inner.compose(composition_request)  # type: ignore[attr-defined]
        completed_at = self._require_timezone_aware(self._clock())
        publication = self._mapper.map_context_view_composed(
            composition_request=composition_request,
            view=view,
            composition_completed_at=completed_at,
        )
        self._side_effect.publish_after_source_success(publication)
        return view

    @staticmethod
    def _require_timezone_aware(value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("clock must return timezone-aware datetimes")
        return value


__all__ = [
    "CollaborativeActivityPublicationFailurePolicy",
    "CollaborativeActivitySourcePublicationSideEffect",
    "CollaborativeDecisionBindingServiceWithActivityPublication",
    "CollaborativeWorkArtifactServiceWithActivityPublication",
    "CollaborativeWorkServiceWithActivityPublication",
    "ContextViewComposerWithActivityPublication",
]
