# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Fail-closed isolation boundary errors (P0-SAFETY-7)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SandboxIsolationFailureReason(StrEnum):
    """Typed reason when required sandbox isolation is unavailable."""

    NOT_CONFIGURED = "not_configured"
    UNHEALTHY = "unhealthy"
    UNSUPPORTED_TIER = "unsupported_tier"


@dataclass(frozen=True)
class SandboxIsolationRequiredError(RuntimeError):
    """Raised when a sandbox-required tool cannot execute without isolation."""

    run_id: str
    agent_id: str
    tool_id: str
    reason: SandboxIsolationFailureReason

    def __str__(self) -> str:
        return (
            f"Sandbox isolation required for tool '{self.tool_id}' "
            f"(agent='{self.agent_id}', run_id={self.run_id}, reason={self.reason.value})."
        )
