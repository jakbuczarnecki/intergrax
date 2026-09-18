# © Artur Czarnecki. All rights reserved.

"""Principal-scoped ContextView source composition ports (Multiplayer MP-5D).

Consumer-owned replaceable ports — reference-first candidates only.
No retrieval, storage, hydration, or composer implementation in this module.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.contracts.agent_run import RequestIdentity, canonical_principal_id_from_request_identity
from intergrax.contracts.context_view import (
    ContextViewCategory,
    ContextViewCollaborativeWorkSourceRef,
    ContextViewKnowledgeSourceRef,
    ContextViewMemorySourceRef,
    ContextViewScope,
    ContextViewUclSourceRef,
    ContextViewVisibilityClass,
)

SCHEMA_CONTEXT_VIEW_MEMORY_SOURCE_REQUEST_V1: Final = "context_view_memory_source_request.v1"
SCHEMA_CONTEXT_VIEW_KNOWLEDGE_SOURCE_REQUEST_V1: Final = (
    "context_view_knowledge_source_request.v1"
)
SCHEMA_CONTEXT_VIEW_UCL_SOURCE_REQUEST_V1: Final = "context_view_ucl_source_request.v1"
SCHEMA_CONTEXT_VIEW_COLLABORATIVE_WORK_SOURCE_REQUEST_V1: Final = (
    "context_view_collaborative_work_source_request.v1"
)
SCHEMA_CONTEXT_VIEW_MEMORY_SOURCE_CANDIDATE_V1: Final = "context_view_memory_source_candidate.v1"
SCHEMA_CONTEXT_VIEW_KNOWLEDGE_SOURCE_CANDIDATE_V1: Final = (
    "context_view_knowledge_source_candidate.v1"
)
SCHEMA_CONTEXT_VIEW_UCL_SOURCE_CANDIDATE_V1: Final = "context_view_ucl_source_candidate.v1"
SCHEMA_CONTEXT_VIEW_COLLABORATIVE_WORK_SOURCE_CANDIDATE_V1: Final = (
    "context_view_collaborative_work_source_candidate.v1"
)
SCHEMA_CONTEXT_VIEW_MEMORY_SOURCE_CANDIDATES_V1: Final = (
    "context_view_memory_source_candidates.v1"
)
SCHEMA_CONTEXT_VIEW_KNOWLEDGE_SOURCE_CANDIDATES_V1: Final = (
    "context_view_knowledge_source_candidates.v1"
)
SCHEMA_CONTEXT_VIEW_UCL_SOURCE_CANDIDATES_V1: Final = "context_view_ucl_source_candidates.v1"
SCHEMA_CONTEXT_VIEW_COLLABORATIVE_WORK_SOURCE_CANDIDATES_V1: Final = (
    "context_view_collaborative_work_source_candidates.v1"
)

_NON_EMPTY = Field(min_length=1)


class ContextViewSourceOutcome(StrEnum):
    """Typed source port completion — not governance HITL."""

    OK = "ok"
    SOURCE_UNAVAILABLE = "source_unavailable"
    SCOPE_REJECTED = "scope_rejected"
    INVALID_REQUEST = "invalid_request"


@runtime_checkable
class ContextViewSourceRequestIdentityView(Protocol):
    """Structural contract for shared MP-5D source request identity fields."""

    scope: ContextViewScope
    acting_principal_id: str
    principal_identity: RequestIdentity
    eligible_visibility_classes: tuple[ContextViewVisibilityClass, ...]


class _ContextViewSourceRequestBase(BaseModel):
    """Shared tenant-scoped source query input — not a policy decision carrier."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scope: ContextViewScope
    acting_principal_id: str = _NON_EMPTY
    principal_identity: RequestIdentity
    eligible_visibility_classes: tuple[ContextViewVisibilityClass, ...] = Field(min_length=1)

    @field_validator("acting_principal_id")
    @classmethod
    def _strip_acting_principal(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized

    @model_validator(mode="after")
    def _align_principal_identity(self) -> _ContextViewSourceRequestBase:
        if self.principal_identity.tenant_id != self.scope.tenant_id:
            raise ValueError("principal_identity tenant_id must match scope tenant_id")
        canonical_id = canonical_principal_id_from_request_identity(self.principal_identity)
        if canonical_id != self.acting_principal_id:
            raise ValueError("acting_principal_id must match principal_identity")
        return self


class ContextViewMemorySourceRequest(_ContextViewSourceRequestBase):
    """Memory-domain source query — MEMORY category only."""

    schema_version: Literal["context_view_memory_source_request.v1"] = (
        SCHEMA_CONTEXT_VIEW_MEMORY_SOURCE_REQUEST_V1
    )
    category: Literal[ContextViewCategory.MEMORY] = ContextViewCategory.MEMORY


class ContextViewKnowledgeSourceRequest(_ContextViewSourceRequestBase):
    """Knowledge / RAG source query — KNOWLEDGE category only."""

    schema_version: Literal["context_view_knowledge_source_request.v1"] = (
        SCHEMA_CONTEXT_VIEW_KNOWLEDGE_SOURCE_REQUEST_V1
    )
    category: Literal[ContextViewCategory.KNOWLEDGE] = ContextViewCategory.KNOWLEDGE
    reference_read_query_text: str = _NON_EMPTY

    @field_validator("reference_read_query_text")
    @classmethod
    def _strip_query_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized


class ContextViewUclSourceRequest(_ContextViewSourceRequestBase):
    """UCL lifecycle source query — UCL_CONTEXT_LIFECYCLE category only."""

    schema_version: Literal["context_view_ucl_source_request.v1"] = (
        SCHEMA_CONTEXT_VIEW_UCL_SOURCE_REQUEST_V1
    )
    category: Literal[ContextViewCategory.UCL_CONTEXT_LIFECYCLE] = (
        ContextViewCategory.UCL_CONTEXT_LIFECYCLE
    )


class ContextViewCollaborativeWorkSourceRequest(_ContextViewSourceRequestBase):
    """Collaborative-work artifact source query — COLLABORATIVE_WORK category only."""

    schema_version: Literal["context_view_collaborative_work_source_request.v1"] = (
        SCHEMA_CONTEXT_VIEW_COLLABORATIVE_WORK_SOURCE_REQUEST_V1
    )
    category: Literal[ContextViewCategory.COLLABORATIVE_WORK] = ContextViewCategory.COLLABORATIVE_WORK


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


def _visibility_allowed(
    *,
    eligible: tuple[ContextViewVisibilityClass, ...],
    suggested: ContextViewVisibilityClass,
) -> bool:
    return suggested in eligible


def _nested_collaborative_ref_work_item_id(
    ref: ContextViewCollaborativeWorkSourceRef,
) -> str | None:
    version = ref.work_artifact_version
    if version is None:
        return None
    return version.work_item_id


def _require_collaborative_work_item_locators_aligned(
    *,
    ref: ContextViewCollaborativeWorkSourceRef,
    expected_work_item_id: str | None,
    scope_mismatch_message: str,
) -> None:
    """Fail closed when wrapper and nested locators disagree or breach expected scope."""
    wrapper = ref.work_item_id
    nested = _nested_collaborative_ref_work_item_id(ref)
    if wrapper is not None and nested is not None and wrapper != nested:
        raise ValueError(
            "work_artifact_version work_item_id must match work_item_id when both set",
        )
    if expected_work_item_id is None:
        return
    if wrapper is not None and wrapper != expected_work_item_id:
        raise ValueError(scope_mismatch_message)
    if nested is not None and nested != expected_work_item_id:
        raise ValueError(scope_mismatch_message)


class ContextViewMemorySourceCandidate(BaseModel):
    """Transient memory candidate — locator only."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["context_view_memory_source_candidate.v1"] = (
        SCHEMA_CONTEXT_VIEW_MEMORY_SOURCE_CANDIDATE_V1
    )
    category: Literal[ContextViewCategory.MEMORY] = ContextViewCategory.MEMORY
    source_ref: ContextViewMemorySourceRef
    candidate_scope: ContextViewScope
    suggested_visibility: ContextViewVisibilityClass

    @model_validator(mode="after")
    def _align_ref_tenant(self) -> ContextViewMemorySourceCandidate:
        if self.source_ref.tenant_id != self.candidate_scope.tenant_id:
            raise ValueError("memory source_ref tenant_id must match candidate_scope tenant_id")
        return self


class ContextViewKnowledgeSourceCandidate(BaseModel):
    """Transient knowledge candidate — locator only."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["context_view_knowledge_source_candidate.v1"] = (
        SCHEMA_CONTEXT_VIEW_KNOWLEDGE_SOURCE_CANDIDATE_V1
    )
    category: Literal[ContextViewCategory.KNOWLEDGE] = ContextViewCategory.KNOWLEDGE
    source_ref: ContextViewKnowledgeSourceRef
    candidate_scope: ContextViewScope
    suggested_visibility: ContextViewVisibilityClass

    @model_validator(mode="after")
    def _align_ref_tenant(self) -> ContextViewKnowledgeSourceCandidate:
        if self.source_ref.tenant_id != self.candidate_scope.tenant_id:
            raise ValueError("knowledge source_ref tenant_id must match candidate_scope tenant_id")
        return self


class ContextViewUclSourceCandidate(BaseModel):
    """Transient UCL lifecycle candidate — locator only."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["context_view_ucl_source_candidate.v1"] = (
        SCHEMA_CONTEXT_VIEW_UCL_SOURCE_CANDIDATE_V1
    )
    category: Literal[ContextViewCategory.UCL_CONTEXT_LIFECYCLE] = (
        ContextViewCategory.UCL_CONTEXT_LIFECYCLE
    )
    source_ref: ContextViewUclSourceRef
    candidate_scope: ContextViewScope
    suggested_visibility: ContextViewVisibilityClass

    @model_validator(mode="after")
    def _align_ref_tenant(self) -> ContextViewUclSourceCandidate:
        if self.source_ref.tenant_id != self.candidate_scope.tenant_id:
            raise ValueError("ucl source_ref tenant_id must match candidate_scope tenant_id")
        return self


class ContextViewCollaborativeWorkSourceCandidate(BaseModel):
    """Transient collaborative-work candidate — locator only."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["context_view_collaborative_work_source_candidate.v1"] = (
        SCHEMA_CONTEXT_VIEW_COLLABORATIVE_WORK_SOURCE_CANDIDATE_V1
    )
    category: Literal[ContextViewCategory.COLLABORATIVE_WORK] = ContextViewCategory.COLLABORATIVE_WORK
    source_ref: ContextViewCollaborativeWorkSourceRef
    candidate_scope: ContextViewScope
    suggested_visibility: ContextViewVisibilityClass

    @model_validator(mode="after")
    def _align_collaborative_ref_scope(self) -> ContextViewCollaborativeWorkSourceCandidate:
        scope = self.candidate_scope
        ref = self.source_ref
        if ref.tenant_id != scope.tenant_id:
            raise ValueError("collaborative source_ref tenant_id must match candidate_scope tenant_id")
        if ref.workspace_id != scope.workspace_id:
            raise ValueError(
                "collaborative source_ref workspace_id must match candidate_scope workspace_id",
            )
        _require_collaborative_work_item_locators_aligned(
            ref=ref,
            expected_work_item_id=scope.work_item_id,
            scope_mismatch_message="collaborative source_ref work_item_id must match candidate_scope",
        )
        return self


class _ContextViewSourceCandidatesResultBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    outcome: ContextViewSourceOutcome

    @model_validator(mode="after")
    def _align_outcome_candidates(self) -> _ContextViewSourceCandidatesResultBase:
        if self.outcome is ContextViewSourceOutcome.OK:
            return self
        if self._candidate_count() > 0:
            raise ValueError("candidates must be empty when outcome is not ok")
        return self

    def _candidate_count(self) -> int:
        raise NotImplementedError


class ContextViewMemorySourceCandidatesResult(_ContextViewSourceCandidatesResultBase):
    schema_version: Literal["context_view_memory_source_candidates.v1"] = (
        SCHEMA_CONTEXT_VIEW_MEMORY_SOURCE_CANDIDATES_V1
    )
    category: Literal[ContextViewCategory.MEMORY] = ContextViewCategory.MEMORY
    candidates: tuple[ContextViewMemorySourceCandidate, ...] = ()

    def _candidate_count(self) -> int:
        return len(self.candidates)


class ContextViewKnowledgeSourceCandidatesResult(_ContextViewSourceCandidatesResultBase):
    schema_version: Literal["context_view_knowledge_source_candidates.v1"] = (
        SCHEMA_CONTEXT_VIEW_KNOWLEDGE_SOURCE_CANDIDATES_V1
    )
    category: Literal[ContextViewCategory.KNOWLEDGE] = ContextViewCategory.KNOWLEDGE
    candidates: tuple[ContextViewKnowledgeSourceCandidate, ...] = ()

    def _candidate_count(self) -> int:
        return len(self.candidates)


class ContextViewUclSourceCandidatesResult(_ContextViewSourceCandidatesResultBase):
    schema_version: Literal["context_view_ucl_source_candidates.v1"] = (
        SCHEMA_CONTEXT_VIEW_UCL_SOURCE_CANDIDATES_V1
    )
    category: Literal[ContextViewCategory.UCL_CONTEXT_LIFECYCLE] = (
        ContextViewCategory.UCL_CONTEXT_LIFECYCLE
    )
    candidates: tuple[ContextViewUclSourceCandidate, ...] = ()

    def _candidate_count(self) -> int:
        return len(self.candidates)


class ContextViewCollaborativeWorkSourceCandidatesResult(_ContextViewSourceCandidatesResultBase):
    schema_version: Literal["context_view_collaborative_work_source_candidates.v1"] = (
        SCHEMA_CONTEXT_VIEW_COLLABORATIVE_WORK_SOURCE_CANDIDATES_V1
    )
    category: Literal[ContextViewCategory.COLLABORATIVE_WORK] = ContextViewCategory.COLLABORATIVE_WORK
    candidates: tuple[ContextViewCollaborativeWorkSourceCandidate, ...] = ()

    def _candidate_count(self) -> int:
        return len(self.candidates)


def validate_memory_source_candidate_isolation(
    *,
    request: ContextViewMemorySourceRequest,
    candidate: ContextViewMemorySourceCandidate,
) -> None:
    """Fail closed on cross-tenant scope or visibility escalation."""
    if candidate.category != request.category:
        raise ValueError("memory candidate category must match request category")
    if candidate.source_ref.tenant_id != request.scope.tenant_id:
        raise ValueError("memory source_ref tenant_id must match request scope tenant_id")
    if not _visibility_allowed(
        eligible=request.eligible_visibility_classes,
        suggested=candidate.suggested_visibility,
    ):
        raise ValueError("memory candidate visibility is not in eligible_visibility_classes")


def validate_knowledge_source_candidate_isolation(
    *,
    request: ContextViewKnowledgeSourceRequest,
    candidate: ContextViewKnowledgeSourceCandidate,
) -> None:
    if candidate.category != request.category:
        raise ValueError("knowledge candidate category must match request category")
    if candidate.source_ref.tenant_id != request.scope.tenant_id:
        raise ValueError("knowledge source_ref tenant_id must match request scope tenant_id")
    if not _visibility_allowed(
        eligible=request.eligible_visibility_classes,
        suggested=candidate.suggested_visibility,
    ):
        raise ValueError("knowledge candidate visibility is not in eligible_visibility_classes")


def validate_ucl_source_candidate_isolation(
    *,
    request: ContextViewUclSourceRequest,
    candidate: ContextViewUclSourceCandidate,
) -> None:
    if candidate.category != request.category:
        raise ValueError("ucl candidate category must match request category")
    if candidate.source_ref.tenant_id != request.scope.tenant_id:
        raise ValueError("ucl source_ref tenant_id must match request scope tenant_id")
    if not _visibility_allowed(
        eligible=request.eligible_visibility_classes,
        suggested=candidate.suggested_visibility,
    ):
        raise ValueError("ucl candidate visibility is not in eligible_visibility_classes")


def validate_collaborative_work_source_candidate_isolation(
    *,
    request: ContextViewCollaborativeWorkSourceRequest,
    candidate: ContextViewCollaborativeWorkSourceCandidate,
) -> None:
    if candidate.category != request.category:
        raise ValueError("collaborative candidate category must match request category")
    ref = candidate.source_ref
    if ref.tenant_id != request.scope.tenant_id:
        raise ValueError("collaborative source_ref tenant_id must match request scope tenant_id")
    if ref.workspace_id != request.scope.workspace_id:
        raise ValueError("collaborative source_ref workspace_id must match request scope workspace_id")
    _require_collaborative_work_item_locators_aligned(
        ref=ref,
        expected_work_item_id=request.scope.work_item_id,
        scope_mismatch_message="collaborative source_ref work_item_id must match request work_item_id",
    )
    if not _visibility_allowed(
        eligible=request.eligible_visibility_classes,
        suggested=candidate.suggested_visibility,
    ):
        raise ValueError(
            "collaborative candidate visibility is not in eligible_visibility_classes",
        )


class MemoryContextSourcePort(Protocol):
    """Replaceable Memory source — eligible MEMORY reference candidates only."""

    def list_candidates(
        self,
        request: ContextViewMemorySourceRequest,
    ) -> ContextViewMemorySourceCandidatesResult:
        """Return zero or more memory locator candidates for the approved scope."""
        ...


class KnowledgeContextSourcePort(Protocol):
    """Replaceable Knowledge / RAG source — KNOWLEDGE reference candidates only."""

    def list_candidates(
        self,
        request: ContextViewKnowledgeSourceRequest,
    ) -> ContextViewKnowledgeSourceCandidatesResult:
        """Return zero or more knowledge locator candidates for the approved scope."""
        ...


class UclContextSourcePort(Protocol):
    """Replaceable UCL lifecycle source — UCL_CONTEXT_LIFECYCLE candidates only."""

    def list_candidates(
        self,
        request: ContextViewUclSourceRequest,
    ) -> ContextViewUclSourceCandidatesResult:
        """Return zero or more UCL locator candidates for the approved scope."""
        ...


class CollaborativeWorkContextSourcePort(Protocol):
    """Replaceable Collaborative Work source — COLLABORATIVE_WORK candidates only."""

    def list_candidates(
        self,
        request: ContextViewCollaborativeWorkSourceRequest,
    ) -> ContextViewCollaborativeWorkSourceCandidatesResult:
        """Return zero or more collaborative-work locator candidates for the approved scope."""
        ...
