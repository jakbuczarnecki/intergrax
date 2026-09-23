# © Artur Czarnecki. All rights reserved.

"""Authority discriminator for sequential HITL generations (UCA-6C-R6-R5)."""

from __future__ import annotations

from enum import StrEnum


class SuspendedOperationAuthorityScope(StrEnum):
    AGENT_RUNTIME_GOVERNANCE = "agent_runtime_governance"
    DECLARATIVE_GOVERNANCE = "declarative_governance"
    MEANINGFUL_SIDE_EFFECT = "meaningful_side_effect"


__all__ = ["SuspendedOperationAuthorityScope"]
