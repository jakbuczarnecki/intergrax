# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Effective autonomy resolution (REL §35, REL-ADV.3)."""

from __future__ import annotations

from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.contracts.agent_contract_meta import AgentRiskLevel
from intergrax.contracts.autonomy_level import AutonomyLevel

_AUTONOMY_RANK = {
    AutonomyLevel.MANUAL: 0,
    AutonomyLevel.ASK: 1,
    AutonomyLevel.AUTONOMOUS: 2,
}

_READ_ONLY_TOOL_PREFIXES = (
    "memory.read",
    "memory.list",
    "rag.retrieve",
    "rag.list",
    "websearch.",
    "eval.",
    "harness.trace",
    "hitl.list",
    "message_bus.get_",
)

_SIDE_EFFECT_TOOL_MARKERS = (
    ".write",
    ".delete",
    ".send",
    ".enqueue",
    ".cancel",
    ".purge",
    "jira.add",
    "notify.send",
    "sandbox.exec",
)


def _rank(level: AutonomyLevel) -> int:
    return _AUTONOMY_RANK[level]


def execution_mode_ceiling(mode: ExecutionMode) -> AutonomyLevel:
    if mode is ExecutionMode.EXPLORATORY:
        return AutonomyLevel.AUTONOMOUS
    if mode is ExecutionMode.BALANCED:
        return AutonomyLevel.AUTONOMOUS
    return AutonomyLevel.ASK


def agent_risk_ceiling(risk_level: AgentRiskLevel | str | None) -> AutonomyLevel:
    if risk_level in {AgentRiskLevel.HIGH, "high", "critical"}:
        return AutonomyLevel.ASK
    return AutonomyLevel.AUTONOMOUS


def resolve_effective_autonomy(
    *,
    requested: AutonomyLevel | None,
    execution_mode: ExecutionMode,
    agent_risk: AgentRiskLevel | str | None = None,
    tenant_ceiling: AutonomyLevel | None = None,
) -> AutonomyLevel:
    effective = requested or AutonomyLevel.ASK
    ceilings = [
        execution_mode_ceiling(execution_mode),
        agent_risk_ceiling(agent_risk),
    ]
    if tenant_ceiling is not None:
        ceilings.append(tenant_ceiling)
    min_rank = min(_rank(level) for level in [effective, *ceilings])
    for level in (AutonomyLevel.MANUAL, AutonomyLevel.ASK, AutonomyLevel.AUTONOMOUS):
        if _rank(level) == min_rank:
            return level
    return AutonomyLevel.MANUAL


def is_read_only_tool(tool_id: str) -> bool:
    normalized = tool_id.strip().lower()
    if any(normalized.startswith(prefix) for prefix in _READ_ONLY_TOOL_PREFIXES):
        return True
    return not any(marker in normalized for marker in _SIDE_EFFECT_TOOL_MARKERS)


def tool_allowed_for_autonomy(tool_id: str, autonomy: AutonomyLevel) -> tuple[bool, str]:
    if autonomy is AutonomyLevel.AUTONOMOUS:
        return True, "autonomous_allow"
    if is_read_only_tool(tool_id):
        return True, "read_only_allow"
    if autonomy is AutonomyLevel.MANUAL:
        return False, "manual_requires_explicit_approval"
    return False, "ask_requires_human_approval"
