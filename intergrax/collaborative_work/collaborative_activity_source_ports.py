# © Artur Czarnecki. All rights reserved.

"""MP-6F integration-owned source operation contracts (adapter seams only)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

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
    WorkArtifact,
    WorkArtifactVersion,
    WorkItem,
)
from intergrax.contracts.decision_record import DecisionProposalRef


@runtime_checkable
class PublishedWorkArtifactPublicationResult(Protocol):
    """Structural authoritative artifact create/publish result — not repository-owned."""

    artifact: WorkArtifact
    version: WorkArtifactVersion


@runtime_checkable
class CollaborativeWorkActivityMutationPort(Protocol):
    """Mutations required by MP-6F Shared Work activity integration."""

    def create_work_item(self, request: CreateWorkItemRequest) -> WorkItem: ...

    def transition_work_item(self, request: TransitionWorkItemRequest) -> WorkItem: ...

    def create_assignment(self, request: CreateAssignmentRequest) -> Assignment: ...

    def transition_assignment(self, request: TransitionAssignmentRequest) -> Assignment: ...


@runtime_checkable
class CollaborativeWorkArtifactActivityMutationPort(Protocol):
    """Mutations required by MP-6F WorkArtifact activity integration."""

    def create_artifact(
        self,
        request: CreateWorkArtifactRequest,
    ) -> PublishedWorkArtifactPublicationResult: ...

    def create_artifact_from_execution(
        self,
        request: CreateWorkArtifactFromExecutionRequest,
    ) -> PublishedWorkArtifactPublicationResult: ...

    def publish_version(
        self,
        request: PublishWorkArtifactVersionRequest,
    ) -> PublishedWorkArtifactPublicationResult: ...

    def publish_version_from_execution(
        self,
        request: PublishWorkArtifactVersionFromExecutionRequest,
    ) -> PublishedWorkArtifactPublicationResult: ...


@runtime_checkable
class CollaborativeDecisionBindingActivitySourcePort(Protocol):
    """Create + read surface preserved by MP-6F decision-binding decorator."""

    def create_binding(
        self,
        request: CreateCollaborativeDecisionBindingRequest,
    ) -> CollaborativeDecisionBinding: ...

    def get_binding(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        binding_id: str,
    ) -> CollaborativeDecisionBinding | None: ...

    def list_bindings_for_work_item(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        work_item_id: str,
    ) -> tuple[CollaborativeDecisionBinding, ...]: ...

    def list_bindings_for_decision_proposal(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        decision_proposal: DecisionProposalRef,
    ) -> tuple[CollaborativeDecisionBinding, ...]: ...


__all__ = [
    "CollaborativeDecisionBindingActivitySourcePort",
    "CollaborativeWorkActivityMutationPort",
    "CollaborativeWorkArtifactActivityMutationPort",
    "PublishedWorkArtifactPublicationResult",
]
