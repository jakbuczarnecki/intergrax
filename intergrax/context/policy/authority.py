# © Artur Czarnecki. All rights reserved.

"""Trusted authority mapping for context fragments (MEM-XINT-5)."""

from __future__ import annotations

from intergrax.context.contracts import (
    ContextAuthorityClass,
    ContextFragment,
    ContextFragmentSource,
    PROVIDER_FORBIDDEN_AUTHORITY_CLASSES,
    replace_context_fragment,
)
from intergrax.context.errors import ContextProviderContractViolationError
from intergrax.context.contracts import ContextProviderDescriptor

AUTHORITY_BY_SOURCE: dict[ContextFragmentSource, ContextAuthorityClass] = {
    ContextFragmentSource.SYSTEM_INSTRUCTIONS: ContextAuthorityClass.SYSTEM_CONTEXT,
    ContextFragmentSource.POLICY_OVERLAY: ContextAuthorityClass.SYSTEM_CONTEXT,
    ContextFragmentSource.LONGTERM_MEMORY: ContextAuthorityClass.CANONICAL_MEMORY,
    ContextFragmentSource.RAG: ContextAuthorityClass.RAG_EVIDENCE,
    ContextFragmentSource.WEBSEARCH: ContextAuthorityClass.RAG_EVIDENCE,
    ContextFragmentSource.TOOL_OUTPUT: ContextAuthorityClass.TOOL_OBSERVATION,
    ContextFragmentSource.SESSION_HISTORY: ContextAuthorityClass.SESSION_EPISODIC,
    ContextFragmentSource.SESSION_HISTORY_SEMANTIC: ContextAuthorityClass.SESSION_EPISODIC,
    ContextFragmentSource.TASK_MESSAGE: ContextAuthorityClass.DERIVED_MEMORY,
    ContextFragmentSource.GRAPH_PRIOR: ContextAuthorityClass.DERIVED_MEMORY,
    ContextFragmentSource.SHARED_CONTEXT: ContextAuthorityClass.DERIVED_MEMORY,
    ContextFragmentSource.ATTACHMENT: ContextAuthorityClass.DERIVED_MEMORY,
    ContextFragmentSource.WORKSPACE: ContextAuthorityClass.DERIVED_MEMORY,
    ContextFragmentSource.CUSTOM: ContextAuthorityClass.DERIVED_MEMORY,
}

_AUTHORITY_RANK: dict[ContextAuthorityClass, int] = {
    ContextAuthorityClass.SYSTEM_CONTEXT: 100,
    ContextAuthorityClass.CANONICAL_MEMORY: 90,
    ContextAuthorityClass.DERIVED_MEMORY: 70,
    ContextAuthorityClass.RAG_EVIDENCE: 60,
    ContextAuthorityClass.TOOL_OBSERVATION: 55,
    ContextAuthorityClass.SESSION_EPISODIC: 50,
    ContextAuthorityClass.UNASSIGNED: 0,
}


def authority_rank(authority: ContextAuthorityClass) -> int:
    return _AUTHORITY_RANK.get(authority, 0)


def resolve_authority_for_source(source: ContextFragmentSource) -> ContextAuthorityClass:
    return AUTHORITY_BY_SOURCE.get(source, ContextAuthorityClass.DERIVED_MEMORY)


def enforce_provider_authority(
    fragment: ContextFragment,
    *,
    descriptor: ContextProviderDescriptor | None = None,
) -> ContextFragment:
    mapped = resolve_authority_for_source(fragment.source)
    declared = fragment.authority_class
    if declared in PROVIDER_FORBIDDEN_AUTHORITY_CLASSES and declared != mapped:
        if descriptor is None:
            raise ValueError("provider self-elevated authority class")
        raise ContextProviderContractViolationError(
            descriptor=descriptor,
            reason_code="provider.contract_violation",
            detail="forbidden authority self-assignment",
        )
    if declared is ContextAuthorityClass.UNASSIGNED:
        return replace_context_fragment(fragment, authority_class=mapped)
    return fragment
