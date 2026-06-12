# © Artur Czarnecki. All rights reserved.

"""Bridge ContextCandidate (as-built) and ContextFragment (CE-1 contracts)."""

from __future__ import annotations

from intergrax.context.contracts import ContextFragment, ContextFragmentSource
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.context.context_compiler_models import (
    ContextCandidate,
    ContextCandidateSource,
)

_CANDIDATE_TO_FRAGMENT_SOURCE: dict[ContextCandidateSource, ContextFragmentSource] = {
    ContextCandidateSource.SYSTEM_INSTRUCTIONS: ContextFragmentSource.SYSTEM_INSTRUCTIONS,
    ContextCandidateSource.SESSION_HISTORY: ContextFragmentSource.SESSION_HISTORY,
    ContextCandidateSource.LONGTERM_MEMORY: ContextFragmentSource.LONGTERM_MEMORY,
    ContextCandidateSource.RAG: ContextFragmentSource.RAG,
    ContextCandidateSource.WEBSEARCH: ContextFragmentSource.WEBSEARCH,
    ContextCandidateSource.ATTACHMENTS: ContextFragmentSource.ATTACHMENT,
    ContextCandidateSource.TOOLS: ContextFragmentSource.TOOL_OUTPUT,
    ContextCandidateSource.USER_TURN: ContextFragmentSource.TASK_MESSAGE,
    ContextCandidateSource.OTHER: ContextFragmentSource.CUSTOM,
}

_FRAGMENT_TO_CANDIDATE_SOURCE: dict[ContextFragmentSource, ContextCandidateSource] = {
    ContextFragmentSource.SYSTEM_INSTRUCTIONS: ContextCandidateSource.SYSTEM_INSTRUCTIONS,
    ContextFragmentSource.SESSION_HISTORY: ContextCandidateSource.SESSION_HISTORY,
    ContextFragmentSource.LONGTERM_MEMORY: ContextCandidateSource.LONGTERM_MEMORY,
    ContextFragmentSource.RAG: ContextCandidateSource.RAG,
    ContextFragmentSource.WEBSEARCH: ContextCandidateSource.WEBSEARCH,
    ContextFragmentSource.ATTACHMENT: ContextCandidateSource.ATTACHMENTS,
    ContextFragmentSource.TOOL_OUTPUT: ContextCandidateSource.TOOLS,
    ContextFragmentSource.TASK_MESSAGE: ContextCandidateSource.USER_TURN,
    ContextFragmentSource.CUSTOM: ContextCandidateSource.OTHER,
}


def fragment_source_from_candidate(source: ContextCandidateSource) -> ContextFragmentSource:
    return _CANDIDATE_TO_FRAGMENT_SOURCE.get(source, ContextFragmentSource.CUSTOM)


def candidate_source_from_fragment(source: ContextFragmentSource) -> ContextCandidateSource:
    return _FRAGMENT_TO_CANDIDATE_SOURCE.get(source, ContextCandidateSource.OTHER)


def fragment_from_candidate(
    candidate: ContextCandidate,
    message: ChatMessage,
    *,
    fragment_id: str | None = None,
) -> ContextFragment:
    content = message.content or ""
    return ContextFragment(
        fragment_id=fragment_id or f"msg-{candidate.message_index}",
        source=fragment_source_from_candidate(candidate.source),
        source_id=str(candidate.message_index),
        content=content,
        token_estimate=candidate.token_estimate,
        relevance_score=candidate.score,
        freshness_score=candidate.score,
        confidence_score=candidate.score,
        mandatory=candidate.mandatory,
        metadata=dict(message.metadata or {}),
    )


def candidate_from_fragment(
    fragment: ContextFragment,
    *,
    message_index: int,
) -> ContextCandidate:
    return ContextCandidate(
        source=candidate_source_from_fragment(fragment.source),
        message_index=message_index,
        score=fragment.relevance_score,
        token_estimate=fragment.token_estimate,
        mandatory=fragment.mandatory,
    )
