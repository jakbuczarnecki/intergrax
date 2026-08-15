# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Trusted internal transport for PRODUCT-4C retrieval source membership scope."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterator

SOURCE_SCOPE_EMPTY = "source_scope_empty"
SOURCE_SCOPE_BLANK_ONLY = "source_scope_blank_only"
SOURCE_SCOPE_MALFORMED = "source_scope_malformed"

_MAX_SOURCE_IDS = 20
_MAX_SOURCE_ID_LEN = 256
_RECOGNIZED_INVALID_REASONS = frozenset(
    {
        SOURCE_SCOPE_EMPTY,
        SOURCE_SCOPE_BLANK_ONLY,
        SOURCE_SCOPE_MALFORMED,
    }
)


class _ScopePresence(str, Enum):
    ABSENT = "absent"
    PRESENT = "present"


@dataclass(frozen=True, slots=True)
class RagRetrievalSourceScopeState:
    """Trusted retrieval source scope for one rag.retrieve invocation."""

    presence: _ScopePresence
    allowed_source_ids: tuple[str, ...] = ()
    error_reason: str | None = None

    @property
    def is_absent(self) -> bool:
        return self.presence is _ScopePresence.ABSENT

    @property
    def is_present(self) -> bool:
        return self.presence is _ScopePresence.PRESENT

    @property
    def is_invalid(self) -> bool:
        return self.is_present and self.error_reason is not None

    def __post_init__(self) -> None:
        if self.presence is _ScopePresence.ABSENT:
            if self.allowed_source_ids:
                raise ValueError("ABSENT scope must not include allowed_source_ids")
            if self.error_reason is not None:
                raise ValueError("ABSENT scope must not include error_reason")
            return

        if self.error_reason is not None:
            if self.allowed_source_ids:
                raise ValueError("INVALID PRESENT scope must not include allowed_source_ids")
            if self.error_reason not in _RECOGNIZED_INVALID_REASONS:
                raise ValueError("INVALID PRESENT scope requires a recognized error_reason")
            return

        if not self.allowed_source_ids:
            raise ValueError("PRESENT VALID scope requires non-empty allowed_source_ids")
        if len(self.allowed_source_ids) > _MAX_SOURCE_IDS:
            raise ValueError("PRESENT VALID scope exceeds maximum source count")
        for source_id in self.allowed_source_ids:
            if not isinstance(source_id, str):
                raise ValueError("PRESENT VALID scope requires string source IDs")
            if not source_id or source_id != source_id.strip():
                raise ValueError("PRESENT VALID scope requires normalized non-blank source IDs")
            if len(source_id) > _MAX_SOURCE_ID_LEN:
                raise ValueError("PRESENT VALID scope source ID exceeds maximum length")


_RAG_RETRIEVAL_SOURCE_SCOPE: ContextVar[RagRetrievalSourceScopeState | None] = ContextVar(
    "rag_retrieval_source_scope",
    default=None,
)


def absent_rag_retrieval_source_scope() -> RagRetrievalSourceScopeState:
    return RagRetrievalSourceScopeState(presence=_ScopePresence.ABSENT)


def _validate_present_source_ids(
    allowed_source_ids: tuple[str, ...],
) -> tuple[str, ...] | str:
    if not isinstance(allowed_source_ids, tuple):
        return SOURCE_SCOPE_MALFORMED
    if len(allowed_source_ids) == 0:
        return SOURCE_SCOPE_EMPTY
    if len(allowed_source_ids) > _MAX_SOURCE_IDS:
        return SOURCE_SCOPE_MALFORMED
    normalized: list[str] = []
    for item in allowed_source_ids:
        if not isinstance(item, str):
            return SOURCE_SCOPE_MALFORMED
        stripped = item.strip()
        if not stripped:
            return SOURCE_SCOPE_BLANK_ONLY
        if len(stripped) > _MAX_SOURCE_ID_LEN:
            return SOURCE_SCOPE_MALFORMED
        normalized.append(stripped)
    if not normalized:
        return SOURCE_SCOPE_BLANK_ONLY
    return tuple(dict.fromkeys(normalized))


def validated_rag_retrieval_source_scope(
    allowed_source_ids: tuple[str, ...],
) -> RagRetrievalSourceScopeState:
    validated = _validate_present_source_ids(allowed_source_ids)
    if isinstance(validated, str):
        return invalid_rag_retrieval_source_scope(validated)
    return RagRetrievalSourceScopeState(
        presence=_ScopePresence.PRESENT,
        allowed_source_ids=validated,
    )


def invalid_rag_retrieval_source_scope(reason: str) -> RagRetrievalSourceScopeState:
    return RagRetrievalSourceScopeState(
        presence=_ScopePresence.PRESENT,
        error_reason=reason,
    )


def parse_trusted_allowed_source_ids(raw: Any) -> RagRetrievalSourceScopeState:
    if raw is None:
        return invalid_rag_retrieval_source_scope(SOURCE_SCOPE_EMPTY)
    if not isinstance(raw, (list, tuple)):
        return invalid_rag_retrieval_source_scope(SOURCE_SCOPE_MALFORMED)
    if len(raw) == 0:
        return invalid_rag_retrieval_source_scope(SOURCE_SCOPE_EMPTY)
    normalized: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            return invalid_rag_retrieval_source_scope(SOURCE_SCOPE_MALFORMED)
        stripped = item.strip()
        if not stripped:
            continue
        normalized.append(stripped)
    if not normalized:
        return invalid_rag_retrieval_source_scope(SOURCE_SCOPE_BLANK_ONLY)
    return validated_rag_retrieval_source_scope(tuple(dict.fromkeys(normalized)))


def parse_task_metadata_allowed_source_ids(
    metadata: dict[str, Any],
    *,
    key: str = "allowed_source_ids",
) -> RagRetrievalSourceScopeState:
    if key not in metadata:
        return absent_rag_retrieval_source_scope()
    return parse_trusted_allowed_source_ids(metadata.get(key))


def current_rag_retrieval_source_scope() -> RagRetrievalSourceScopeState:
    current = _RAG_RETRIEVAL_SOURCE_SCOPE.get()
    if current is None:
        return absent_rag_retrieval_source_scope()
    return current


def bind_rag_retrieval_source_scope(
    state: RagRetrievalSourceScopeState,
) -> Token[RagRetrievalSourceScopeState | None]:
    return _RAG_RETRIEVAL_SOURCE_SCOPE.set(state)


def reset_rag_retrieval_source_scope(
    token: Token[RagRetrievalSourceScopeState | None],
) -> None:
    _RAG_RETRIEVAL_SOURCE_SCOPE.reset(token)


@contextmanager
def rag_retrieval_source_scope(
    state: RagRetrievalSourceScopeState,
) -> Iterator[None]:
    token = bind_rag_retrieval_source_scope(state)
    try:
        yield
    finally:
        reset_rag_retrieval_source_scope(token)
