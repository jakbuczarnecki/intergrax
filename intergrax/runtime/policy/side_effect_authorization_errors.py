# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Fail-closed authorization boundary for meaningful tool side effects."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SideEffectAuthorizationFailureReason(StrEnum):
    """Typed reason for missing or non-enforcing declarative tool authorization."""

    NOT_CONFIGURED = "not_configured"
    NON_ENFORCING_MODE = "non_enforcing_mode"


@dataclass(frozen=True)
class MeaningfulSideEffectAuthorizationRequiredError(RuntimeError):
    """Raised when a side-effecting tool lacks recognized enforcing authorization."""

    run_id: str
    agent_id: str
    tool_id: str
    reason: SideEffectAuthorizationFailureReason

    def __str__(self) -> str:
        return (
            f"Meaningful side-effect authorization required for tool '{self.tool_id}' "
            f"(agent='{self.agent_id}', run_id={self.run_id}, reason={self.reason.value})."
        )
