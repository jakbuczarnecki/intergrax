# © Artur Czarnecki. All rights reserved.

"""Org memory maturity checklist (MEM-MAINT-02)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.memory.org_memory_scope import ORG_MEMORY_SCOPES, OrgMemoryScope
from intergrax.memory.user_profile_memory import MemoryKind, UserProfile, UserProfileMemoryEntry


@dataclass(frozen=True)
class OrgMemoryMaturityResult:
    passed: bool
    violations: tuple[str, ...] = ()


def evaluate_org_memory_maturity(profile: UserProfile) -> OrgMemoryMaturityResult:
    """Check org-scoped entries declare scope metadata and known org kinds."""
    violations: list[str] = []
    org_kinds = {MemoryKind.ORG_FACT, MemoryKind.POLICY, MemoryKind.PROCEDURAL}

    for entry in profile.memory_entries:
        if entry.deleted or entry.kind not in org_kinds:
            continue
        scope = (entry.metadata or {}).get("org_scope")
        if not scope:
            violations.append(f"{entry.entry_id}: org kind {entry.kind.value} missing org_scope metadata")
            continue
        if scope not in {s.value for s in ORG_MEMORY_SCOPES}:
            violations.append(f"{entry.entry_id}: unknown org_scope {scope!r}")

    return OrgMemoryMaturityResult(passed=not violations, violations=tuple(violations))


def org_memory_entry(
    *,
    content: str,
    kind: MemoryKind,
    scope: OrgMemoryScope,
) -> UserProfileMemoryEntry:
    """Factory for org-scoped LTM entries with maturity metadata."""
    return UserProfileMemoryEntry(
        content=content,
        kind=kind,
        metadata={"org_scope": scope.value, "cognitive_store": kind.value},
    )
