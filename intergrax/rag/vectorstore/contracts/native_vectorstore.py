# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Native, immutable vector-store provider contracts.

These records, scopes and hits are the provider-facing ABI and deliberately
do not import LangChain.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from intergrax.knowledge.contracts import KnowledgeDocument, KnowledgeDocumentScope
from intergrax.knowledge.contracts.validation import (
    JsonValue,
    assert_safe_mapping,
    freeze_knowledge_metadata,
    require_non_empty_str,
)

_ROUTING_KEYS = frozenset({"tenant_id", "namespace", "workspace_id"})


class VectorStoreContractError(ValueError):
    """Raised when a native vector-store contract is malformed."""


def _copy_document(document: KnowledgeDocument) -> KnowledgeDocument:
    if not isinstance(document, KnowledgeDocument):
        raise TypeError("document must be a KnowledgeDocument")
    try:
        return KnowledgeDocument.model_validate(document.model_dump(mode="python"))
    except Exception as exc:
        raise VectorStoreContractError("document failed full revalidation") from exc


def _copy_vector(value: object, *, field_name: str) -> NDArray[np.float32]:
    try:
        vector = np.array(value, dtype=np.float32, copy=True)
    except (TypeError, ValueError) as exc:
        raise VectorStoreContractError(f"{field_name} must be a numeric vector") from exc
    if vector.ndim != 1:
        raise VectorStoreContractError(f"{field_name} must be exactly 1D")
    if vector.size == 0:
        raise VectorStoreContractError(f"{field_name} must have a positive dimension")
    if not np.isfinite(vector).all():
        raise VectorStoreContractError(f"{field_name} must contain only finite values")
    vector.setflags(write=False)
    return vector


def _validate_score(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VectorStoreContractError("similarity_score must be a finite number")
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise VectorStoreContractError("similarity_score must be finite and in [0.0, 1.0]")
    return score


def _validate_rank(value: object) -> int:
    if type(value) is not int or value < 0:
        raise VectorStoreContractError("rank must be an exact non-negative int")
    return value


@dataclass(frozen=True)
class VectorStoreScope:
    """Authoritative tenant and routing scope for one vector-store operation."""

    tenant_id: str
    namespace: str | None = None
    workspace_id: str | None = None

    def __post_init__(self) -> None:
        try:
            validated = KnowledgeDocumentScope.model_validate(
                {
                    "tenant_id": self.tenant_id,
                    "namespace": self.namespace,
                    "workspace_id": self.workspace_id,
                }
            )
        except Exception as exc:
            raise VectorStoreContractError("invalid vector-store scope") from exc
        object.__setattr__(self, "tenant_id", validated.tenant_id)
        object.__setattr__(self, "namespace", validated.namespace)
        object.__setattr__(self, "workspace_id", validated.workspace_id)

    @classmethod
    def from_document(cls, document: KnowledgeDocument) -> VectorStoreScope:
        if not isinstance(document, KnowledgeDocument):
            raise TypeError("document must be a KnowledgeDocument")
        return cls(
            tenant_id=document.scope.tenant_id,
            namespace=document.scope.namespace,
            workspace_id=document.scope.workspace_id,
        )

    def matches_document(self, document: KnowledgeDocument) -> bool:
        return (
            isinstance(document, KnowledgeDocument)
            and document.scope.tenant_id == self.tenant_id
            and document.scope.namespace == self.namespace
            and document.scope.workspace_id == self.workspace_id
        )

    def matches(self, other: VectorStoreScope) -> bool:
        return (
            isinstance(other, VectorStoreScope)
            and self.tenant_id == other.tenant_id
            and self.namespace == other.namespace
            and self.workspace_id == other.workspace_id
        )


@dataclass(frozen=True)
class MetadataFilter:
    """Immutable provider-neutral equality conditions.

    Routing keys are created only by :meth:`for_scope`; callers cannot provide
    them as ordinary user conditions.
    """

    conditions: Mapping[str, JsonValue]
    _allow_routing_keys: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        try:
            normalized = assert_safe_mapping(self.conditions, field_name="conditions")
        except (TypeError, ValueError) as exc:
            raise VectorStoreContractError("conditions must be JSON-compatible") from exc
        if not self._allow_routing_keys:
            reserved = _ROUTING_KEYS.intersection(normalized)
            if reserved:
                key = sorted(reserved)[0]
                raise VectorStoreContractError(
                    f"conditions must not contain reserved routing key '{key}'"
                )
        object.__setattr__(self, "conditions", freeze_knowledge_metadata(normalized))

    @classmethod
    def for_scope(
        cls,
        scope: VectorStoreScope,
        user_filter: MetadataFilter | None = None,
    ) -> MetadataFilter:
        if not isinstance(scope, VectorStoreScope):
            raise TypeError("scope must be a VectorStoreScope")
        conditions: dict[str, JsonValue] = {}
        if user_filter is not None:
            if not isinstance(user_filter, MetadataFilter):
                raise TypeError("metadata_filter must be a MetadataFilter")
            conditions.update(user_filter.conditions)
        conditions["tenant_id"] = scope.tenant_id
        if scope.namespace is not None:
            conditions["namespace"] = scope.namespace
        if scope.workspace_id is not None:
            conditions["workspace_id"] = scope.workspace_id
        normalized = assert_safe_mapping(conditions, field_name="conditions")
        instance = object.__new__(cls)
        object.__setattr__(instance, "conditions", freeze_knowledge_metadata(normalized))
        object.__setattr__(instance, "_allow_routing_keys", True)
        return instance


@dataclass(frozen=True)
class VectorStoreRecord:
    """Immutable native document plus exactly one vector.

    ``vector_id`` is the provider record ID. It may differ from the canonical
    ``document.identity.document_id`` only for existing call sites that need
    an external storage ID; the two roles remain explicit.
    """

    document: KnowledgeDocument
    embedding: NDArray[np.float32]
    vector_id: str

    def __post_init__(self) -> None:
        document = _copy_document(self.document)
        vector_id = require_non_empty_str(self.vector_id, field_name="vector_id")
        embedding = _copy_vector(self.embedding, field_name="embedding")
        object.__setattr__(self, "document", document)
        object.__setattr__(self, "embedding", embedding)
        object.__setattr__(self, "vector_id", vector_id)

    @property
    def document_id(self) -> str:
        return self.document.identity.document_id


@dataclass(frozen=True)
class VectorStoreHit:
    """Immutable native vector-store result containing a KnowledgeDocument."""

    vector_id: str
    document: KnowledgeDocument
    similarity_score: float
    rank: int
    embedding: NDArray[np.float32] | None = None

    def __post_init__(self) -> None:
        document = _copy_document(self.document)
        vector_id = require_non_empty_str(self.vector_id, field_name="vector_id")
        score = _validate_score(self.similarity_score)
        rank = _validate_rank(self.rank)
        embedding = (
            None
            if self.embedding is None
            else _copy_vector(self.embedding, field_name="embedding")
        )
        object.__setattr__(self, "document", document)
        object.__setattr__(self, "vector_id", vector_id)
        object.__setattr__(self, "similarity_score", score)
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "embedding", embedding)

    @property
    def id(self) -> str:
        return self.vector_id

    @property
    def content(self) -> str:
        return self.document.content

    @property
    def metadata(self) -> dict[str, JsonValue]:
        return dict(self.document.metadata)
