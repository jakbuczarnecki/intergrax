# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Declarative policy enforcement errors at the tool invocation boundary."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeclarativePolicyViolationError(RuntimeError):
    """Raised when declarative policy denies a tool invocation in enforce mode."""

    run_id: str
    agent_id: str
    tool_id: str
    matched_rule_ids: tuple[str, ...]
    reasons: tuple[str, ...]

    def __str__(self) -> str:
        rules = ", ".join(self.matched_rule_ids) or "none"
        return (
            f"Declarative policy denied tool '{self.tool_id}' for agent "
            f"'{self.agent_id}' (run_id={self.run_id}, rules={rules})."
        )


@dataclass(frozen=True)
class DeclarativePolicyHitlRequiredError(RuntimeError):
    """Typed boundary signal: declarative policy requires HITL before tool execution."""

    run_id: str
    agent_id: str
    tool_id: str
    matched_rule_ids: tuple[str, ...]
    reasons: tuple[str, ...]

    def __str__(self) -> str:
        rules = ", ".join(self.matched_rule_ids) or "none"
        return (
            f"Declarative policy requires human approval for tool '{self.tool_id}' "
            f"(agent='{self.agent_id}', run_id={self.run_id}, rules={rules})."
        )
