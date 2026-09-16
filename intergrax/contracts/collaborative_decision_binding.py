# © Artur Czarnecki. All rights reserved.

"""Multiplayer-owned association between Collaborative Work and exact Decision proposals (MP-4R4)."""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.contracts.collaborative_work import (
    AuthorityDelegation,
    MembershipResolutionMode,
    WorkspaceMembership,
)
from intergrax.contracts.collaborative_work import WorkArtifactVersionRef
from intergrax.contracts.decision_record import DecisionProposalRef

SCHEMA_COLLABORATIVE_DECISION_BINDING_V1: str = "collaborative_decision_binding.v1"
SCHEMA_CREATE_COLLABORATIVE_DECISION_BINDING_REQUEST_V1: str = (
    "create_collaborative_decision_binding_request.v1"
)

_BINDING_ID_PREFIX: str = "cdb_"


class CollaborativeDecisionBindingNotFound(Exception):
    """Binding was not found for the requested tenant/workspace scope."""


class CollaborativeDecisionBindingAlreadyExists(Exception):
    """Binding already exists for the requested scoped identity."""


class CollaborativeDecisionBindingDuplicateSemantic(Exception):
    """An equivalent semantic association already exists (deduplicated binding)."""


class CollaborativeDecisionBindingIdempotencyConflict(Exception):
    """Idempotency key replayed with a different semantic command."""


class CollaborativeDecisionBindingScopeMismatch(Exception):
    """Binding scope does not align with referenced collaborative or decision scope."""


class CollaborativeDecisionBindingReferenceMismatch(Exception):
    """Referenced WorkArtifactVersion does not belong to the binding WorkItem scope."""


def mint_collaborative_decision_binding_id(*, idempotency_key: str | None = None) -> str:
    """Mint an independent collaborative decision binding identity."""
    if idempotency_key is not None:
        normalized = idempotency_key.strip()
        if not normalized:
            raise ValueError("idempotency_key must be non-empty when provided")
        digest = hashlib.sha256(f"collaborative_decision_binding:{normalized}".encode()).hexdigest()[:32]
        return f"{_BINDING_ID_PREFIX}{digest}"
    return f"{_BINDING_ID_PREFIX}{uuid4().hex}"


class CollaborativeDecisionBinding(BaseModel):
    """Immutable association record — references authorities; never becomes one."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    schema_version: Literal["collaborative_decision_binding.v1"] = SCHEMA_COLLABORATIVE_DECISION_BINDING_V1
    binding_id: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    work_item_id: str = Field(min_length=1)
    work_artifact_version: WorkArtifactVersionRef | None = None
    decision_proposal: DecisionProposalRef
    created_by_principal_id: str = Field(min_length=1)
    created_at: datetime

    @field_validator(
        "binding_id",
        "tenant_id",
        "workspace_id",
        "work_item_id",
        "created_by_principal_id",
    )
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized

    @field_validator("decision_proposal", mode="before")
    @classmethod
    def _validate_decision_proposal(cls, value: object) -> DecisionProposalRef:
        if type(value) is not DecisionProposalRef:
            raise TypeError("decision_proposal must be DecisionProposalRef")
        return value

    @field_validator("created_at")
    @classmethod
    def _timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _validate_artifact_scope(self) -> CollaborativeDecisionBinding:
        if self.work_artifact_version is None:
            return self
        ref = self.work_artifact_version
        if ref.tenant_id != self.tenant_id:
            raise ValueError("work_artifact_version tenant_id must match binding tenant_id")
        if ref.workspace_id != self.workspace_id:
            raise ValueError("work_artifact_version workspace_id must match binding workspace_id")
        if ref.work_item_id != self.work_item_id:
            raise ValueError("work_artifact_version work_item_id must match binding work_item_id")
        return self


class CreateCollaborativeDecisionBindingRequest(BaseModel):
    """Authoritative binding create input for MP-4R4 service mutations."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    schema_version: Literal["create_collaborative_decision_binding_request.v1"] = (
        SCHEMA_CREATE_COLLABORATIVE_DECISION_BINDING_REQUEST_V1
    )
    tenant_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    work_item_id: str = Field(min_length=1)
    work_artifact_version: WorkArtifactVersionRef | None = None
    decision_proposal: DecisionProposalRef
    acting_principal_id: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    delegator_principal_id: str | None = None
    membership: WorkspaceMembership | None = None
    membership_resolution_mode: MembershipResolutionMode = MembershipResolutionMode.LOCATOR
    delegation: AuthorityDelegation | None = None

    @field_validator(
        "tenant_id",
        "workspace_id",
        "work_item_id",
        "acting_principal_id",
        "idempotency_key",
        "delegator_principal_id",
    )
    @classmethod
    def _strip_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty when provided")
        return normalized

    @field_validator("decision_proposal", mode="before")
    @classmethod
    def _validate_decision_proposal(cls, value: object) -> DecisionProposalRef:
        if type(value) is not DecisionProposalRef:
            raise TypeError("decision_proposal must be DecisionProposalRef")
        return value

    @model_validator(mode="after")
    def _validate_authority_locators(self) -> CreateCollaborativeDecisionBindingRequest:
        if (
            self.membership_resolution_mode is MembershipResolutionMode.CANONICAL_PRINCIPAL
            and self.membership is not None
        ):
            raise ValueError(
                "canonical_principal membership resolution must not include an embedded membership locator",
            )
        if self.membership is not None:
            if self.membership.tenant_id != self.tenant_id:
                raise ValueError("membership tenant_id must match request tenant_id")
            if self.membership.workspace_id != self.workspace_id:
                raise ValueError("membership workspace_id must match request workspace_id")
            if self.membership.principal_id != self.acting_principal_id:
                raise ValueError("membership principal_id must match request acting_principal_id")
        if self.delegation is not None:
            if self.delegation.tenant_id != self.tenant_id:
                raise ValueError("delegation tenant_id must match request tenant_id")
            if self.delegation.workspace_id != self.workspace_id:
                raise ValueError("delegation workspace_id must match request workspace_id")
        if self.work_artifact_version is not None:
            ref = self.work_artifact_version
            if ref.tenant_id != self.tenant_id:
                raise ValueError("work_artifact_version tenant_id must match request tenant_id")
            if ref.workspace_id != self.workspace_id:
                raise ValueError("work_artifact_version workspace_id must match request workspace_id")
            if ref.work_item_id != self.work_item_id:
                raise ValueError("work_artifact_version work_item_id must match request work_item_id")
        return self
