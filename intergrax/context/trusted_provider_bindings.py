# © Artur Czarnecki. All rights reserved.

"""Trusted provider-id authority bindings (MEM-XINT-5-R) — not source inference."""

from __future__ import annotations

from intergrax.context.contracts import ContextAuthorityClass

BUILTIN_TRUSTED_PROVIDER_AUTHORITY: dict[str, ContextAuthorityClass] = {
    "builtin.system_instructions": ContextAuthorityClass.SYSTEM_CONTEXT,
    "builtin.policy_overlay": ContextAuthorityClass.SYSTEM_CONTEXT,
    "builtin.longterm_memory": ContextAuthorityClass.CANONICAL_MEMORY,
    "builtin.rag": ContextAuthorityClass.RAG_EVIDENCE,
    "builtin.websearch": ContextAuthorityClass.RAG_EVIDENCE,
    "builtin.tool_output": ContextAuthorityClass.TOOL_OBSERVATION,
    "builtin.session_history": ContextAuthorityClass.SESSION_EPISODIC,
    "builtin.session_history_semantic": ContextAuthorityClass.SESSION_EPISODIC,
    "builtin.task_message": ContextAuthorityClass.DERIVED_MEMORY,
    "builtin.graph_prior": ContextAuthorityClass.DERIVED_MEMORY,
    "builtin.shared_context": ContextAuthorityClass.DERIVED_MEMORY,
    "builtin.attachments": ContextAuthorityClass.DERIVED_MEMORY,
    "builtin.workspace": ContextAuthorityClass.DERIVED_MEMORY,
}


def trusted_authority_for_provider_id(provider_id: str) -> ContextAuthorityClass | None:
    normalized = provider_id.strip().lower()
    return BUILTIN_TRUSTED_PROVIDER_AUTHORITY.get(normalized)
