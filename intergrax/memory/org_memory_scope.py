# © Artur Czarnecki. All rights reserved.

"""Organizational memory scope taxonomy (AUDIT-IDEAL-15.1 / org memory 2.5)."""

from __future__ import annotations

from enum import Enum


class OrgMemoryScope(str, Enum):
    """First-class org LTM scopes for harness memory governance."""

    ORG_PROFILE = "org_profile"
    ORG_KNOWLEDGE = "org_knowledge"
    ORG_PROCEDURAL = "org_procedural"


ORG_MEMORY_SCOPES: tuple[OrgMemoryScope, ...] = tuple(OrgMemoryScope)
