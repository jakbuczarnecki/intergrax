# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Re-export of the neutral retrieval source scope contract (relocated, not changed)."""

from __future__ import annotations

from intergrax.contracts.rag_retrieval_source_scope import (
    _RAG_RETRIEVAL_SOURCE_SCOPE,
    SOURCE_SCOPE_BLANK_ONLY,
    SOURCE_SCOPE_EMPTY,
    SOURCE_SCOPE_MALFORMED,
    RagRetrievalSourceScopeState,
    _ScopePresence,
    absent_rag_retrieval_source_scope,
    bind_rag_retrieval_source_scope,
    current_rag_retrieval_source_scope,
    invalid_rag_retrieval_source_scope,
    parse_task_metadata_allowed_source_ids,
    parse_trusted_allowed_source_ids,
    rag_retrieval_source_scope,
    reset_rag_retrieval_source_scope,
    validated_rag_retrieval_source_scope,
)

__all__ = [
    "_RAG_RETRIEVAL_SOURCE_SCOPE",
    "_ScopePresence",
    "RagRetrievalSourceScopeState",
    "SOURCE_SCOPE_BLANK_ONLY",
    "SOURCE_SCOPE_EMPTY",
    "SOURCE_SCOPE_MALFORMED",
    "absent_rag_retrieval_source_scope",
    "bind_rag_retrieval_source_scope",
    "current_rag_retrieval_source_scope",
    "invalid_rag_retrieval_source_scope",
    "parse_task_metadata_allowed_source_ids",
    "parse_trusted_allowed_source_ids",
    "rag_retrieval_source_scope",
    "reset_rag_retrieval_source_scope",
    "validated_rag_retrieval_source_scope",
]
