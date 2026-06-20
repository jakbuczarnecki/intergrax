# © Artur Czarnecki. All rights reserved.

"""Cognitive store mapping for MemoryKind → harness retrieval scope (MEM-MAINT-01)."""

from __future__ import annotations

from intergrax.memory.org_memory_scope import OrgMemoryScope
from intergrax.memory.user_profile_memory import MemoryKind

_COGNITIVE_STORE_BY_KIND: dict[MemoryKind, str] = {
    MemoryKind.USER_FACT: "semantic_ltm",
    MemoryKind.PREFERENCE: "semantic_ltm",
    MemoryKind.SESSION_SUMMARY: "episodic_ltm",
    MemoryKind.EPISODIC_EVENT: "episodic_ltm",
    MemoryKind.SEMANTIC: "semantic_ltm",
    MemoryKind.PROCEDURAL: "procedural_ltm",
    MemoryKind.ORG_FACT: "org_semantic_ltm",
    MemoryKind.POLICY: "org_policy_ltm",
    MemoryKind.OTHER: "general_ltm",
}

_ORG_SCOPE_BY_KIND: dict[MemoryKind, OrgMemoryScope | None] = {
    MemoryKind.ORG_FACT: OrgMemoryScope.ORG_KNOWLEDGE,
    MemoryKind.POLICY: OrgMemoryScope.ORG_PROFILE,
    MemoryKind.PROCEDURAL: OrgMemoryScope.ORG_PROCEDURAL,
}


def cognitive_store_for_kind(kind: MemoryKind) -> str:
    """Return harness cognitive store id for a memory kind."""
    return _COGNITIVE_STORE_BY_KIND.get(kind, "general_ltm")


def org_scope_for_kind(kind: MemoryKind) -> OrgMemoryScope | None:
    """Return org LTM scope when kind is org-scoped; else None."""
    return _ORG_SCOPE_BY_KIND.get(kind)
