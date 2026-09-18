# © Artur Czarnecki. All rights reserved.

"""Principal-scoped ContextView composition contracts (Multiplayer MP-5E).

Consumer-owned replaceable composer seam — policy-approved composition only.
No retrieval, storage, hydration, or domain adapter implementation.
"""

from __future__ import annotations

from typing import Final, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.contracts.agent_run import RequestIdentity, canonical_principal_id_from_request_identity
from intergrax.contracts.context_view import (
    ContextView,
    ContextViewCategory,
    ContextViewCollaborativeWorkSourceRef,
    ContextViewEntrySourceRef,
    ContextViewKnowledgeSourceRef,
    ContextViewMemorySourceRef,
    ContextViewRequest,
    ContextViewScope,
    ContextViewUclSourceRef,
    ContextViewVisibilityClass,
)
from intergrax.contracts.context_view_visibility_policy import (
    ContextViewPolicyDecision,
    ContextViewPolicyOutcome,
)

SCHEMA_CONTEXT_VIEW_COMPOSITION_REQUEST_V1: Final = "context_view_composition_request.v1"
SCHEMA_DEFAULT_CONTEXT_VIEW_COMPOSER_CONFIG_V1: Final = "default_context_view_composer_config.v1"
SCHEMA_CONTEXT_VIEW_COMPOSITION_VALIDATED_CANDIDATE_V1: Final = (
    "context_view_composition_validated_candidate.v1"
)

DEFAULT_CONTEXT_VIEW_COMPOSER_ID: Final = "collaborative_work.context_view.composer.default"


class ContextViewCompositionError(Exception):
    """Base composition failure — fail closed."""


class ContextViewCompositionPolicyDeniedError(ContextViewCompositionError):
    """Policy outcome DENY — no source invocation permitted."""


class ContextViewCompositionRequestAlignmentError(ContextViewCompositionError):
    """Request and policy decision are not aligned for composition."""


class ContextViewCompositionSourceFailureError(ContextViewCompositionError):
    """Source port returned a non-OK outcome or required port is missing."""


class ContextViewCompositionCandidateIsolationError(ContextViewCompositionError):
    """Candidate failed MP-5D isolation validation."""


class ContextViewCompositionInvariantError(ContextViewCompositionError):
    """Composed view violated composition invariants."""


class ContextViewCompositionRequest(BaseModel):
    """Typed composer input — approved policy decision bound to the originating request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["context_view_composition_request.v1"] = (
        SCHEMA_CONTEXT_VIEW_COMPOSITION_REQUEST_V1
    )
    request: ContextViewRequest
    policy_decision: ContextViewPolicyDecision
    principal_identity: RequestIdentity

    @model_validator(mode="after")
    def _align_principal_identity(self) -> ContextViewCompositionRequest:
        request = self.request
        identity = self.principal_identity
        if identity.tenant_id != request.scope.tenant_id:
            raise ValueError("principal_identity tenant_id must match request scope tenant_id")
        canonical_id = canonical_principal_id_from_request_identity(identity)
        if canonical_id != request.acting_principal_id:
            raise ValueError("principal_identity must match request acting_principal_id")
        return self


class DefaultContextViewComposerConfig(BaseModel):
    """Typed immutable configuration for the platform default composer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["default_context_view_composer_config.v1"] = (
        SCHEMA_DEFAULT_CONTEXT_VIEW_COMPOSER_CONFIG_V1
    )
    composer_id: str = Field(default=DEFAULT_CONTEXT_VIEW_COMPOSER_ID, min_length=1)
    knowledge_reference_read_query_text: str | None = None

    @field_validator("knowledge_reference_read_query_text")
    @classmethod
    def _strip_knowledge_query(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class ContextViewCompositionValidatedCandidate(BaseModel):
    """Validated, reference-first candidate ready for ordering and entry materialization."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["context_view_composition_validated_candidate.v1"] = (
        SCHEMA_CONTEXT_VIEW_COMPOSITION_VALIDATED_CANDIDATE_V1
    )
    category: ContextViewCategory
    source_ref: ContextViewEntrySourceRef
    entry_scope: ContextViewScope
    visibility: ContextViewVisibilityClass


def _scopes_collaboratively_compatible(
    request_scope: ContextViewScope,
    candidate_scope: ContextViewScope,
) -> bool:
    if request_scope.tenant_id != candidate_scope.tenant_id:
        return False
    if request_scope.workspace_id != candidate_scope.workspace_id:
        return False
    if request_scope.work_item_id is not None:
        if candidate_scope.work_item_id != request_scope.work_item_id:
            return False
    if request_scope.operation_scope is not None:
        req_op = request_scope.operation_scope
        cand_op = candidate_scope.operation_scope
        if cand_op is None or cand_op.operation_id != req_op.operation_id:
            return False
        if req_op.resource_scope is not None:
            if cand_op.resource_scope != req_op.resource_scope:
                return False
    return True


def effective_scope_within_request_scope(
    *,
    request_scope: ContextViewScope,
    effective_scope: ContextViewScope,
) -> bool:
    """True when policy effective scope is no broader than the originating request scope."""
    if effective_scope.tenant_id != request_scope.tenant_id:
        return False
    if effective_scope.workspace_id != request_scope.workspace_id:
        return False
    if request_scope.work_item_id is not None:
        if effective_scope.work_item_id != request_scope.work_item_id:
            return False
    request_op = request_scope.operation_scope
    effective_op = effective_scope.operation_scope
    if request_op is not None:
        if effective_op is None:
            return False
        if effective_op.operation_id != request_op.operation_id:
            return False
        if request_op.resource_scope is not None:
            if effective_op.resource_scope != request_op.resource_scope:
                return False
    return True


def validate_composition_request_alignment(
    *,
    composition_request: ContextViewCompositionRequest,
) -> None:
    """Fail closed when policy decision does not match the bound ContextViewRequest."""
    request = composition_request.request
    decision = composition_request.policy_decision
    if decision.outcome is not ContextViewPolicyOutcome.ALLOW:
        raise ContextViewCompositionRequestAlignmentError(
            "policy_decision outcome must be ALLOW for composition",
        )
    if not effective_scope_within_request_scope(
        request_scope=request.scope,
        effective_scope=decision.effective_scope,
    ):
        raise ContextViewCompositionRequestAlignmentError(
            "policy effective_scope is not within request scope",
        )
    for category in decision.eligible_categories:
        if category not in request.requested_categories:
            raise ContextViewCompositionRequestAlignmentError(
                "eligible category is not present in requested_categories",
            )


def context_view_source_ref_identity_key(source_ref: ContextViewEntrySourceRef) -> tuple[str, ...]:
    """Deterministic dedupe identity from typed locator fields — not opaque serialization."""
    if isinstance(source_ref, ContextViewMemorySourceRef):
        return (
            source_ref.schema_version,
            ContextViewCategory.MEMORY.value,
            source_ref.tenant_id,
            source_ref.record_ref,
        )
    if isinstance(source_ref, ContextViewKnowledgeSourceRef):
        return (
            source_ref.schema_version,
            ContextViewCategory.KNOWLEDGE.value,
            source_ref.tenant_id,
            source_ref.knowledge_ref,
        )
    if isinstance(source_ref, ContextViewUclSourceRef):
        return (
            source_ref.schema_version,
            ContextViewCategory.UCL_CONTEXT_LIFECYCLE.value,
            source_ref.tenant_id,
            source_ref.ucl_artifact_ref,
        )
    if isinstance(source_ref, ContextViewCollaborativeWorkSourceRef):
        version = source_ref.work_artifact_version
        version_id = ""
        artifact_id = ""
        work_item_id = source_ref.work_item_id or ""
        if version is not None:
            version_id = version.work_artifact_version_id
            artifact_id = version.work_artifact_id
            if not work_item_id:
                work_item_id = version.work_item_id
        return (
            source_ref.schema_version,
            ContextViewCategory.COLLABORATIVE_WORK.value,
            source_ref.tenant_id,
            source_ref.workspace_id,
            work_item_id,
            artifact_id,
            version_id,
        )
    raise ContextViewCompositionInvariantError("unsupported source_ref type for identity key")


def validate_context_view_matches_composition_request(
    *,
    view: ContextView,
    composition_request: ContextViewCompositionRequest,
) -> None:
    """Ensure composed view preserves least-context scope and principal from the approved decision."""
    request = composition_request.request
    decision = composition_request.policy_decision
    if view.acting_principal_id != request.acting_principal_id:
        raise ValueError("view acting_principal_id must match request acting_principal_id")
    if view.scope != decision.effective_scope:
        raise ValueError("view scope must equal policy_decision effective_scope")
    eligible_visibility = decision.eligible_visibility_classes
    for entry in view.entries:
        if entry.visibility not in eligible_visibility:
            raise ValueError("entry visibility must be in policy eligible_visibility_classes")
        if entry.entry_scope.tenant_id != view.scope.tenant_id:
            raise ValueError("entry tenant_id must match view scope tenant_id")
        if entry.entry_scope.workspace_id != view.scope.workspace_id:
            raise ValueError("entry workspace_id must match view scope workspace_id")


@runtime_checkable
class ContextViewCandidateOrderingStrategy(Protocol):
    """Replaceable ordering for validated candidates before entry materialization."""

    def order(
        self,
        *,
        eligible_category_order: tuple[ContextViewCategory, ...],
        candidates: tuple[ContextViewCompositionValidatedCandidate, ...],
    ) -> tuple[ContextViewCompositionValidatedCandidate, ...]:
        """Return candidates in stable composition order."""


@runtime_checkable
class ContextViewEntryIdentityStrategy(Protocol):
    """Replaceable deterministic entry_id generation."""

    def entry_id_for_candidate(
        self,
        *,
        candidate: ContextViewCompositionValidatedCandidate,
        composition_request: ContextViewCompositionRequest,
    ) -> str:
        """Produce a non-empty entry_id for one admitted candidate."""


@runtime_checkable
class ContextViewIdentityStrategy(Protocol):
    """Replaceable view_id generation for a composed ContextView."""

    def view_id_for_composition(
        self,
        *,
        composition_request: ContextViewCompositionRequest,
        entry_ids: tuple[str, ...],
    ) -> str:
        """Produce a non-empty view_id for the composed result."""


@runtime_checkable
class ContextViewComposer(Protocol):
    """Replaceable principal-scoped ContextView composer — MP-5E extension seam."""

    def compose(self, composition_request: ContextViewCompositionRequest) -> ContextView:
        """Compose a reference-first ContextView from an approved policy decision and source ports."""
