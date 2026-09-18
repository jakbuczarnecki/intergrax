# © Artur Czarnecki. All rights reserved.

"""Scoped reference-first Knowledge read port (MP-5F-B2).

Knowledge/RAG owns retrieval, ranking, and indexing semantics. This module
exposes canonical knowledge references only — no payload hydration and no
ContextView / MP-5 types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

from intergrax.contracts.agent_run import RequestIdentity

__all__ = [
    "KNOWLEDGE_REFERENCE_READ_DEFAULT_LIMIT",
    "KNOWLEDGE_REFERENCE_READ_MAX_LIMIT",
    "KNOWLEDGE_REFERENCE_READ_MAX_QUERY_CHARS",
    "KnowledgeChunkCanonicalRef",
    "KnowledgeReferenceReadOutcome",
    "KnowledgeReferenceReadPort",
    "KnowledgeReferenceReadQuery",
    "KnowledgeReferenceReadRequest",
    "KnowledgeReferenceReadResult",
    "KnowledgeReferenceReadScope",
    "KnowledgeReferenceReadScopeError",
    "KnowledgeReferenceProjectionError",
    "KnowledgeScopedResourceRef",
    "validate_knowledge_reference_read_request",
]

KNOWLEDGE_REFERENCE_READ_DEFAULT_LIMIT = 20
KNOWLEDGE_REFERENCE_READ_MAX_LIMIT = 100
KNOWLEDGE_REFERENCE_READ_MAX_QUERY_CHARS = 4096


class KnowledgeReferenceReadOutcome(str, Enum):
    OK = "ok"
    ACCESS_DENIED = "access_denied"
    SCOPE_REJECTED = "scope_rejected"
    INVALID_REQUEST = "invalid_request"
    UNAVAILABLE = "unavailable"


class KnowledgeReferenceReadScopeError(ValueError):
    """Request scope or fields violate Knowledge reference-read invariants."""


class KnowledgeReferenceProjectionError(ValueError):
    """A retrieval hit cannot be projected to a canonical knowledge reference."""

    def __init__(self, reason: str) -> None:
        cleaned = (reason or "").strip() or "projection_failed"
        self.reason = cleaned
        super().__init__(cleaned)


@dataclass(frozen=True, slots=True)
class KnowledgeScopedResourceRef:
    """Optional document/source binding within a scoped retrieval request."""

    source_id: str | None = None
    document_id: str | None = None

    def __post_init__(self) -> None:
        source = self.source_id.strip() if self.source_id else None
        document = self.document_id.strip() if self.document_id else None
        if source is None and document is None:
            raise KnowledgeReferenceReadScopeError(
                "resource scope requires source_id and/or document_id"
            )
        if source is not None and not source:
            raise KnowledgeReferenceReadScopeError("source_id must be non-empty when set")
        if document is not None and not document:
            raise KnowledgeReferenceReadScopeError(
                "document_id must be non-empty when set"
            )
        object.__setattr__(self, "source_id", source)
        object.__setattr__(self, "document_id", document)


@dataclass(frozen=True, slots=True)
class KnowledgeReferenceReadScope:
    """Knowledge-owned least-context boundary aligned with ``KnowledgeDocumentScope``."""

    tenant_id: str
    workspace_id: str
    namespace: str | None = None
    resource: KnowledgeScopedResourceRef | None = None

    def __post_init__(self) -> None:
        tenant = (self.tenant_id or "").strip()
        workspace = (self.workspace_id or "").strip()
        if not tenant:
            raise KnowledgeReferenceReadScopeError("tenant_id must be non-empty")
        if not workspace:
            raise KnowledgeReferenceReadScopeError("workspace_id must be non-empty")
        namespace = self.namespace
        if namespace is not None:
            namespace = namespace.strip()
            if not namespace:
                raise KnowledgeReferenceReadScopeError(
                    "namespace when set must be non-empty"
                )
        object.__setattr__(self, "tenant_id", tenant)
        object.__setattr__(self, "workspace_id", workspace)
        object.__setattr__(self, "namespace", namespace)


@dataclass(frozen=True, slots=True)
class KnowledgeReferenceReadQuery:
    """Bounded retrieval intent — ranking strategy remains domain-owned."""

    query_text: str
    limit: int = KNOWLEDGE_REFERENCE_READ_DEFAULT_LIMIT

    def __post_init__(self) -> None:
        text = (self.query_text or "").strip()
        if not text:
            raise KnowledgeReferenceReadScopeError("query_text must be non-empty")
        if len(text) > KNOWLEDGE_REFERENCE_READ_MAX_QUERY_CHARS:
            raise KnowledgeReferenceReadScopeError(
                f"query_text must be <= {KNOWLEDGE_REFERENCE_READ_MAX_QUERY_CHARS} chars"
            )
        if self.limit < 1:
            raise KnowledgeReferenceReadScopeError("limit must be >= 1")
        if self.limit > KNOWLEDGE_REFERENCE_READ_MAX_LIMIT:
            raise KnowledgeReferenceReadScopeError(
                f"limit must be <= {KNOWLEDGE_REFERENCE_READ_MAX_LIMIT}"
            )
        object.__setattr__(self, "query_text", text)


@dataclass(frozen=True, slots=True)
class KnowledgeReferenceReadRequest:
    scope: KnowledgeReferenceReadScope
    query: KnowledgeReferenceReadQuery


@dataclass(frozen=True, slots=True)
class KnowledgeChunkCanonicalRef:
    """Canonical reference for one retrievable indexed knowledge item.

    ``knowledge_ref`` is the provider-neutral logical ``vector_id`` persisted by
    the native vector store (one indexed chunk / vector record). It must not
    alias ``document_id``, which is the canonical owning document identity
    (``KnowledgeDocumentIdentity.root_document_id`` when lineage is present).
    """

    tenant_id: str
    knowledge_ref: str
    document_id: str
    source_id: str | None = None
    rank: int = 0
    relevance_score: float | None = None

    def __post_init__(self) -> None:
        tenant = (self.tenant_id or "").strip()
        knowledge_ref = (self.knowledge_ref or "").strip()
        document_id = (self.document_id or "").strip()
        if not tenant:
            raise KnowledgeReferenceReadScopeError("tenant_id must be non-empty")
        if not knowledge_ref:
            raise KnowledgeReferenceReadScopeError("knowledge_ref must be non-empty")
        if not document_id:
            raise KnowledgeReferenceReadScopeError("document_id must be non-empty")
        if self.rank < 0:
            raise KnowledgeReferenceReadScopeError("rank must be >= 0")
        score = self.relevance_score
        if score is not None and not (0.0 <= score <= 1.0):
            raise KnowledgeReferenceReadScopeError(
                "relevance_score when set must be in [0.0, 1.0]"
            )
        source = self.source_id
        if source is not None:
            source = source.strip()
            if not source:
                raise KnowledgeReferenceReadScopeError(
                    "source_id when set must be non-empty"
                )
        object.__setattr__(self, "tenant_id", tenant)
        object.__setattr__(self, "knowledge_ref", knowledge_ref)
        object.__setattr__(self, "document_id", document_id)
        object.__setattr__(self, "source_id", source)


@dataclass(frozen=True, slots=True)
class KnowledgeReferenceReadResult:
    outcome: KnowledgeReferenceReadOutcome
    references: tuple[KnowledgeChunkCanonicalRef, ...] = ()
    evaluated_scope: KnowledgeReferenceReadScope | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        if self.outcome is not KnowledgeReferenceReadOutcome.OK and self.references:
            raise KnowledgeReferenceReadScopeError(
                "non-OK outcomes must not carry references"
            )
        if self.outcome is KnowledgeReferenceReadOutcome.OK:
            if self.evaluated_scope is None:
                raise KnowledgeReferenceReadScopeError(
                    "OK outcome requires authoritative evaluated_scope"
                )
        elif self.evaluated_scope is not None:
            raise KnowledgeReferenceReadScopeError(
                "evaluated_scope must be omitted when outcome is not OK"
            )


def validate_knowledge_reference_read_request(
    identity: RequestIdentity,
    request: KnowledgeReferenceReadRequest,
) -> KnowledgeReferenceReadOutcome | None:
    """Return a failure outcome when the request is invalid for the identity; else None."""
    tenant = (identity.tenant_id or "").strip()
    if not tenant:
        return KnowledgeReferenceReadOutcome.INVALID_REQUEST
    if tenant != request.scope.tenant_id:
        return KnowledgeReferenceReadOutcome.SCOPE_REJECTED
    return None


@runtime_checkable
class KnowledgeReferenceReadPort(Protocol):
    """Provider-neutral, reference-first Knowledge retrieval capability."""

    def read_references(
        self,
        identity: RequestIdentity,
        request: KnowledgeReferenceReadRequest,
    ) -> KnowledgeReferenceReadResult: ...
