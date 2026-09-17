# © Artur Czarnecki. All rights reserved.

"""Principal-scoped ContextView public contracts (Multiplayer MP-5B).

Collaborative Work owns visibility eligibility and reference-first read projection
semantics — not Memory stores, RAG retrieval, UCL lifecycle, or CE assembly.

Distinct from ``SharedContextView``, ``DecisionContextView``, and ``MemoryView``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.contracts.collaborative_work import (
    AuthorityDelegation,
    EffectiveAuthorityRequest,
    MembershipResolutionMode,
    WorkArtifactVersionRef,
    WorkspaceMembership,
)

SCHEMA_CONTEXT_VIEW_SCOPE_V1: Final = "context_view_scope.v1"
SCHEMA_CONTEXT_VIEW_OPERATION_SCOPE_V1: Final = "context_view_operation_scope.v1"
SCHEMA_CONTEXT_VIEW_REQUEST_V1: Final = "context_view_request.v1"
SCHEMA_CONTEXT_VIEW_V1: Final = "context_view.v1"
SCHEMA_CONTEXT_VIEW_ENTRY_V1: Final = "context_view_entry.v1"
SCHEMA_CONTEXT_VIEW_MEMORY_SOURCE_REF_V1: Final = "context_view_memory_source_ref.v1"
SCHEMA_CONTEXT_VIEW_KNOWLEDGE_SOURCE_REF_V1: Final = "context_view_knowledge_source_ref.v1"
SCHEMA_CONTEXT_VIEW_UCL_SOURCE_REF_V1: Final = "context_view_ucl_source_ref.v1"
SCHEMA_CONTEXT_VIEW_COLLABORATIVE_WORK_SOURCE_REF_V1: Final = (
    "context_view_collaborative_work_source_ref.v1"
)

_NON_EMPTY = Field(min_length=1)


class ContextViewVisibilityClass(StrEnum):
    """Frozen MP-5A visibility policy concepts — not magic metadata strings."""

    PRIVATE_TO_PRINCIPAL = "private_to_principal"
    WORKSPACE_SHARED = "workspace_shared"
    WORK_ITEM = "work_item"
    DELEGATED_VISIBLE = "delegated_visible"
    PLATFORM_VISIBLE = "platform_visible"


class ContextViewCategory(StrEnum):
    """Vendor-neutral eligible context categories for principal-scoped views."""

    MEMORY = "memory"
    KNOWLEDGE = "knowledge"
    UCL_CONTEXT_LIFECYCLE = "ucl_context_lifecycle"
    COLLABORATIVE_WORK = "collaborative_work"


class ContextViewOperationScope(BaseModel):
    """Typed operation/resource scope for a view request — not an arbitrary string bag."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["context_view_operation_scope.v1"] = (
        SCHEMA_CONTEXT_VIEW_OPERATION_SCOPE_V1
    )
    operation_id: str = _NON_EMPTY
    resource_scope: str | None = None

    @field_validator("operation_id", "resource_scope")
    @classmethod
    def _strip_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty when provided")
        return normalized


class ContextViewScope(BaseModel):
    """Authoritative collaborative scope for ContextView requests and results."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["context_view_scope.v1"] = SCHEMA_CONTEXT_VIEW_SCOPE_V1
    tenant_id: str = _NON_EMPTY
    workspace_id: str = _NON_EMPTY
    work_item_id: str | None = None
    operation_scope: ContextViewOperationScope | None = None

    @field_validator("tenant_id", "workspace_id", "work_item_id")
    @classmethod
    def _strip_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty when provided")
        return normalized


class ContextViewMemorySourceRef(BaseModel):
    """Memory-domain locator — no memory record payload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["context_view_memory_source_ref.v1"] = (
        SCHEMA_CONTEXT_VIEW_MEMORY_SOURCE_REF_V1
    )
    tenant_id: str = _NON_EMPTY
    record_ref: str = _NON_EMPTY

    @field_validator("tenant_id", "record_ref")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized


class ContextViewKnowledgeSourceRef(BaseModel):
    """Knowledge / RAG domain locator — no retrieval payload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["context_view_knowledge_source_ref.v1"] = (
        SCHEMA_CONTEXT_VIEW_KNOWLEDGE_SOURCE_REF_V1
    )
    tenant_id: str = _NON_EMPTY
    knowledge_ref: str = _NON_EMPTY

    @field_validator("tenant_id", "knowledge_ref")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized


class ContextViewUclSourceRef(BaseModel):
    """UCL lifecycle artifact locator — no revision body."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["context_view_ucl_source_ref.v1"] = SCHEMA_CONTEXT_VIEW_UCL_SOURCE_REF_V1
    tenant_id: str = _NON_EMPTY
    ucl_artifact_ref: str = _NON_EMPTY

    @field_validator("tenant_id", "ucl_artifact_ref")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized


class ContextViewCollaborativeWorkSourceRef(BaseModel):
    """Collaborative-work plane locator — reuses ``WorkArtifactVersionRef`` when needed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["context_view_collaborative_work_source_ref.v1"] = (
        SCHEMA_CONTEXT_VIEW_COLLABORATIVE_WORK_SOURCE_REF_V1
    )
    tenant_id: str = _NON_EMPTY
    workspace_id: str = _NON_EMPTY
    work_item_id: str | None = None
    work_artifact_version: WorkArtifactVersionRef | None = None

    @field_validator("tenant_id", "workspace_id", "work_item_id")
    @classmethod
    def _strip_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty when provided")
        return normalized

    @model_validator(mode="after")
    def _require_locator(self) -> ContextViewCollaborativeWorkSourceRef:
        if self.work_item_id is None and self.work_artifact_version is None:
            raise ValueError(
                "work_item_id or work_artifact_version is required for collaborative source ref",
            )
        return self

    @model_validator(mode="after")
    def _align_artifact_version_scope(self) -> ContextViewCollaborativeWorkSourceRef:
        version = self.work_artifact_version
        if version is None:
            return self
        if version.tenant_id != self.tenant_id:
            raise ValueError("work_artifact_version tenant_id must match collaborative ref tenant_id")
        if version.workspace_id != self.workspace_id:
            raise ValueError(
                "work_artifact_version workspace_id must match collaborative ref workspace_id",
            )
        if self.work_item_id is not None and version.work_item_id != self.work_item_id:
            raise ValueError("work_artifact_version work_item_id must match work_item_id when both set")
        return self


ContextViewEntrySourceRef = Annotated[
    ContextViewMemorySourceRef
    | ContextViewKnowledgeSourceRef
    | ContextViewUclSourceRef
    | ContextViewCollaborativeWorkSourceRef,
    Field(discriminator="schema_version"),
]


class ContextViewEntry(BaseModel):
    """Reference-first admitted context item — no domain-owned payload bodies."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["context_view_entry.v1"] = SCHEMA_CONTEXT_VIEW_ENTRY_V1
    entry_id: str = _NON_EMPTY
    source_ref: ContextViewEntrySourceRef
    visibility: ContextViewVisibilityClass
    entry_scope: ContextViewScope

    @field_validator("entry_id")
    @classmethod
    def _strip_entry_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized


class ContextViewRequest(BaseModel):
    """Principal-scoped view intent — no provider payloads or runtime Nexus state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["context_view_request.v1"] = SCHEMA_CONTEXT_VIEW_REQUEST_V1
    scope: ContextViewScope
    acting_principal_id: str = _NON_EMPTY
    operation_id: str = _NON_EMPTY
    requested_categories: tuple[ContextViewCategory, ...] = Field(min_length=1)
    authority_request: EffectiveAuthorityRequest | None = None
    delegator_principal_id: str | None = None
    membership: WorkspaceMembership | None = None
    membership_resolution_mode: MembershipResolutionMode = MembershipResolutionMode.LOCATOR
    delegation: AuthorityDelegation | None = None

    @field_validator("acting_principal_id", "operation_id", "delegator_principal_id")
    @classmethod
    def _strip_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty when provided")
        return normalized

    @model_validator(mode="after")
    def _validate_scope_and_authority(self) -> ContextViewRequest:
        scope = self.scope
        if self.authority_request is not None:
            auth = self.authority_request
            if auth.tenant_id != scope.tenant_id:
                raise ValueError("authority_request tenant_id must match scope tenant_id")
            if auth.workspace_id != scope.workspace_id:
                raise ValueError("authority_request workspace_id must match scope workspace_id")
            if auth.acting_principal_id != self.acting_principal_id:
                raise ValueError(
                    "authority_request acting_principal_id must match request acting_principal_id",
                )

        if (
            self.membership_resolution_mode is MembershipResolutionMode.CANONICAL_PRINCIPAL
            and self.membership is not None
        ):
            raise ValueError(
                "canonical_principal membership resolution must not include an embedded membership locator",
            )

        if self.membership is not None:
            if self.membership.tenant_id != scope.tenant_id:
                raise ValueError("membership tenant_id must match scope tenant_id")
            if self.membership.workspace_id != scope.workspace_id:
                raise ValueError("membership workspace_id must match scope workspace_id")
            if self.membership.principal_id != self.acting_principal_id:
                raise ValueError("membership principal_id must match acting_principal_id")

        if self.delegation is not None:
            if self.delegation.tenant_id != scope.tenant_id:
                raise ValueError("delegation tenant_id must match scope tenant_id")
            if self.delegation.workspace_id != scope.workspace_id:
                raise ValueError("delegation workspace_id must match scope workspace_id")
            if self.delegation.delegate_principal_id != self.acting_principal_id:
                raise ValueError(
                    "delegation delegate_principal_id must match acting_principal_id",
                )
            if (
                self.delegator_principal_id is not None
                and self.delegation.delegator_principal_id != self.delegator_principal_id
            ):
                raise ValueError(
                    "delegation delegator_principal_id must match delegator_principal_id",
                )

        if scope.operation_scope is not None and scope.operation_scope.operation_id != self.operation_id:
            raise ValueError("scope.operation_scope.operation_id must match request operation_id")

        return self


class ContextView(BaseModel):
    """Immutable principal-scoped read projection — not storage or CE assembly."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["context_view.v1"] = SCHEMA_CONTEXT_VIEW_V1
    view_id: str = _NON_EMPTY
    scope: ContextViewScope
    acting_principal_id: str = _NON_EMPTY
    entries: tuple[ContextViewEntry, ...] = ()

    @field_validator("view_id", "acting_principal_id")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized

    @model_validator(mode="after")
    def _validate_entry_scope_alignment(self) -> ContextView:
        for entry in self.entries:
            if entry.entry_scope.tenant_id != self.scope.tenant_id:
                raise ValueError("entry tenant_id must match view scope tenant_id")
            if entry.entry_scope.workspace_id != self.scope.workspace_id:
                raise ValueError("entry workspace_id must match view scope workspace_id")
            self._validate_source_ref_tenant(entry)
        return self

    @staticmethod
    def _validate_source_ref_tenant(entry: ContextViewEntry) -> None:
        source = entry.source_ref
        tenant_id = entry.entry_scope.tenant_id
        if isinstance(source, ContextViewMemorySourceRef) and source.tenant_id != tenant_id:
            raise ValueError("memory source_ref tenant_id must match entry scope tenant_id")
        if isinstance(source, ContextViewKnowledgeSourceRef) and source.tenant_id != tenant_id:
            raise ValueError("knowledge source_ref tenant_id must match entry scope tenant_id")
        if isinstance(source, ContextViewUclSourceRef) and source.tenant_id != tenant_id:
            raise ValueError("ucl source_ref tenant_id must match entry scope tenant_id")
        if isinstance(source, ContextViewCollaborativeWorkSourceRef):
            if source.tenant_id != tenant_id:
                raise ValueError(
                    "collaborative source_ref tenant_id must match entry scope tenant_id",
                )
            if source.workspace_id != entry.entry_scope.workspace_id:
                raise ValueError(
                    "collaborative source_ref workspace_id must match entry scope workspace_id",
                )


def validate_context_view_matches_request(*, view: ContextView, request: ContextViewRequest) -> None:
    """Ensure a composed view cannot change principal or collaborative scope."""
    if view.scope.tenant_id != request.scope.tenant_id:
        raise ValueError("view scope tenant_id must match request scope tenant_id")
    if view.scope.workspace_id != request.scope.workspace_id:
        raise ValueError("view scope workspace_id must match request scope workspace_id")
    if view.scope.work_item_id != request.scope.work_item_id:
        raise ValueError("view scope work_item_id must match request scope work_item_id")
    if view.acting_principal_id != request.acting_principal_id:
        raise ValueError("view acting_principal_id must match request acting_principal_id")
