# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Structured retrieval error taxonomy (M-RAG.28)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence, Tuple


class RetrievalErrorKind(str, Enum):
    RETRIEVER_NOT_FOUND = "retriever_not_found"
    RETRIEVER_EXHAUSTED = "retriever_exhausted"
    VECTOR_BACKEND_FAILURE = "vector_backend_failure"
    EMBEDDING_FAILURE = "embedding_failure"
    CIRCUIT_OPEN = "circuit_open"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RetrievalError(Exception):
    """Typed retrieval failure with retry / fallback hints."""

    kind: RetrievalErrorKind
    message: str
    retriever_id: str = ""
    attempted_retriever_ids: Tuple[str, ...] = ()
    retryable: bool = False
    cause: Optional[BaseException] = None

    def __str__(self) -> str:
        return self.message


_RETRYABLE_EXCEPTION_TYPES = (
    TimeoutError,
    ConnectionError,
    OSError,
)


def classify_retrieval_exception(
    exc: BaseException,
    *,
    retriever_id: str,
    attempted_retriever_ids: Sequence[str] = (),
) -> RetrievalError:
    if isinstance(exc, RetrievalError):
        return exc

    from intergrax.integrations.contracts.base import IntegrationDependencyError

    if isinstance(exc, IntegrationDependencyError):
        return RetrievalError(
            kind=RetrievalErrorKind.CIRCUIT_OPEN,
            message=str(exc),
            retriever_id=retriever_id,
            attempted_retriever_ids=tuple(attempted_retriever_ids),
            retryable=True,
            cause=exc,
        )

    if isinstance(exc, RuntimeError) and "not registered" in str(exc).lower():
        return RetrievalError(
            kind=RetrievalErrorKind.RETRIEVER_NOT_FOUND,
            message=str(exc),
            retriever_id=retriever_id,
            attempted_retriever_ids=tuple(attempted_retriever_ids),
            retryable=False,
            cause=exc,
        )

    message = str(exc) or exc.__class__.__name__
    lowered = message.lower()
    if "embed" in lowered:
        kind = RetrievalErrorKind.EMBEDDING_FAILURE
    elif any(token in lowered for token in ("vector", "qdrant", "weaviate", "chroma", "pinecone")):
        kind = RetrievalErrorKind.VECTOR_BACKEND_FAILURE
    else:
        kind = RetrievalErrorKind.UNKNOWN

    retryable = isinstance(exc, _RETRYABLE_EXCEPTION_TYPES)
    return RetrievalError(
        kind=kind,
        message=message,
        retriever_id=retriever_id,
        attempted_retriever_ids=tuple(attempted_retriever_ids),
        retryable=retryable,
        cause=exc,
    )
