# © Artur Czarnecki. All rights reserved.

"""Collaborative Work ↔ canonical Decision proposal binding service (MP-4R4)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Final

from intergrax.collaborative_work._authority_enforcement import _require_collaborative_allow
from intergrax.collaborative_work.enforcement_gate import CollaborativeWorkEnforcementGate
from intergrax.collaborative_work.repository import (
    CollaborativeDecisionBindingRepository,
    CreateCollaborativeDecisionBindingCommand,
    WorkArtifactNotFound,
    WorkArtifactVersionRepository,
    WorkItemNotFound,
    WorkItemRepository,
)
from intergrax.contracts.decision_record import DecisionProposalRef
from intergrax.contracts.collaborative_work import work_item_resource_scope
from intergrax.contracts.collaborative_decision_binding import (
    CollaborativeDecisionBinding,
    CollaborativeDecisionBindingReferenceMismatch,
    CollaborativeDecisionBindingScopeMismatch,
    CreateCollaborativeDecisionBindingRequest,
    mint_collaborative_decision_binding_id,
)

TRUSTED_OPERATION_COLLABORATIVE_DECISION_BINDING_CREATE: Final = (
    "collaborative_work.decision_binding.create"
)


class CollaborativeDecisionBindingService:
    """Domain boundary for append-only Collaborative Work decision proposal associations."""

    def __init__(
        self,
        *,
        work_item_repository: WorkItemRepository,
        work_artifact_version_repository: WorkArtifactVersionRepository,
        binding_repository: CollaborativeDecisionBindingRepository,
        enforcement_gate: CollaborativeWorkEnforcementGate,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._work_item_repository = work_item_repository
        self._work_artifact_version_repository = work_artifact_version_repository
        self._binding_repository = binding_repository
        self._enforcement_gate = enforcement_gate
        self._clock = clock or (lambda: datetime.now(UTC))

    def create_binding(
        self,
        request: CreateCollaborativeDecisionBindingRequest,
    ) -> CollaborativeDecisionBinding:
        work_item = self._work_item_repository.get(
            tenant_id=request.tenant_id,
            workspace_id=request.workspace_id,
            work_item_id=request.work_item_id,
        )
        if work_item is None:
            raise WorkItemNotFound("work item was not found")

        if request.decision_proposal.identity.tenant_id != request.tenant_id:
            raise CollaborativeDecisionBindingScopeMismatch(
                "decision proposal tenant_id must match binding tenant_id",
            )

        if request.work_artifact_version is not None:
            self._validate_artifact_version(request)

        resource_scope = work_item_resource_scope(work_item_id=request.work_item_id)
        _require_collaborative_allow(
            enforcement_gate=self._enforcement_gate,
            operation_id=TRUSTED_OPERATION_COLLABORATIVE_DECISION_BINDING_CREATE,
            request=request,
            resource_scope=resource_scope,
        )

        binding_id = mint_collaborative_decision_binding_id(idempotency_key=request.idempotency_key)
        created_at = self._require_timezone_aware(self._clock())
        return self._binding_repository.create(
            CreateCollaborativeDecisionBindingCommand(
                tenant_id=request.tenant_id,
                workspace_id=request.workspace_id,
                binding_id=binding_id,
                work_item_id=request.work_item_id,
                work_artifact_version=request.work_artifact_version,
                decision_proposal=request.decision_proposal,
                created_by_principal_id=request.acting_principal_id,
                created_at=created_at,
                idempotency_key=request.idempotency_key,
            ),
        )

    def get_binding(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        binding_id: str,
    ) -> CollaborativeDecisionBinding | None:
        return self._binding_repository.get(
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
        return self._binding_repository.list_for_work_item(
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
        if type(decision_proposal) is not DecisionProposalRef:
            raise TypeError("decision_proposal must be DecisionProposalRef")
        return self._binding_repository.list_for_decision_proposal(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            decision_proposal=decision_proposal,
        )

    def _validate_artifact_version(self, request: CreateCollaborativeDecisionBindingRequest) -> None:
        assert request.work_artifact_version is not None
        ref = request.work_artifact_version
        version = self._work_artifact_version_repository.get(
            tenant_id=ref.tenant_id,
            workspace_id=ref.workspace_id,
            work_artifact_version_id=ref.work_artifact_version_id,
        )
        if version is None:
            raise WorkArtifactNotFound("work artifact version was not found")
        if version.work_item_id != ref.work_item_id:
            raise CollaborativeDecisionBindingReferenceMismatch(
                "work artifact version work_item_id mismatch",
            )
        if version.work_artifact_id != ref.work_artifact_id:
            raise CollaborativeDecisionBindingReferenceMismatch(
                "work artifact version work_artifact_id mismatch",
            )
        if version.tenant_id != ref.tenant_id or version.workspace_id != ref.workspace_id:
            raise CollaborativeDecisionBindingReferenceMismatch(
                "work artifact version scope mismatch",
            )

    @staticmethod
    def _require_timezone_aware(value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("clock must return timezone-aware datetime")
        return value
