# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""V1 capability kind vocabulary (CAPABILITY-CATALOG-1 Stage 1)."""

from __future__ import annotations

from enum import StrEnum
from typing import Final


class CapabilityKind(StrEnum):
    """Frozen V1 capability types for federated discovery — Agent, Skill, Tool only."""

    AGENT = "agent"
    SKILL = "skill"
    TOOL = "tool"


V1_CAPABILITY_KINDS: Final[frozenset[CapabilityKind]] = frozenset(
    {
        CapabilityKind.AGENT,
        CapabilityKind.SKILL,
        CapabilityKind.TOOL,
    }
)
