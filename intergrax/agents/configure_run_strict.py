# © Artur Czarnecki. All rights reserved.

"""STRICT-mode deny for configure_run / request widen attempts (ACP-CLOSE-ORG-1 · §30.6 · §39.4)."""

from __future__ import annotations

from typing import Any

from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.applications.contracts.org_policy import OrganizationalPolicyContext
from intergrax.contracts.acp_metadata_keys import AcpRunContextKey
from intergrax.contracts.agent_run import AgentEnvironmentOverrides

_STRICT_FORBIDDEN_OVERLAY_KEYS = frozenset(
    {
        "allowed_tools",
        "tool_allowlist_add",
        "tool_denylist_remove",
        "extra_tools",
        "skills",
        "skill_ids",
        "environment_overrides",
        "side_effect_mode",
        "organizational",
        AcpRunContextKey.ORGANIZATIONAL,
        "execution_mode",
        "policy_pre_deny",
    }
)


class ConfigureRunStrictViolation(Exception):
    """Raised when configure_run attempts to widen posture in STRICT mode."""

    def __init__(self, violations: list[str]) -> None:
        super().__init__("; ".join(violations))
        self.violations = list(violations)


def resolve_effective_execution_mode(
    *,
    app_execution_mode: ExecutionMode | None = None,
    organizational: OrganizationalPolicyContext | None = None,
) -> ExecutionMode:
    """STRICT wins when either host profile or org envelope requires it."""
    if app_execution_mode == ExecutionMode.STRICT:
        return ExecutionMode.STRICT
    if organizational is not None and organizational.execution_mode == ExecutionMode.STRICT:
        return ExecutionMode.STRICT
    if app_execution_mode is not None:
        return app_execution_mode
    if organizational is not None:
        return organizational.execution_mode
    return ExecutionMode.BALANCED


def _channel_widens(channel: str, organizational: OrganizationalPolicyContext) -> bool:
    if channel in organizational.channel_policy.denied_channels:
        return True
    allowed = organizational.channel_policy.allowed_channels
    return bool(allowed and channel not in allowed)


def validate_configure_run_overlay_strict(
    overlay: dict[str, Any],
    *,
    execution_mode: ExecutionMode,
    organizational: OrganizationalPolicyContext | None = None,
) -> list[str]:
    """Return violation codes when overlay widens tools/org/channel posture in STRICT."""
    if execution_mode != ExecutionMode.STRICT or not overlay:
        return []

    violations: list[str] = []
    for key in overlay:
        if key in _STRICT_FORBIDDEN_OVERLAY_KEYS:
            violations.append(f"configure_run.forbidden_key:{key}")

    channel = overlay.get("channel")
    if (
        isinstance(channel, str)
        and channel
        and organizational is not None
        and _channel_widens(channel, organizational)
    ):
        violations.append(f"configure_run.channel_widen:{channel}")

    tool_add = overlay.get("tool_allowlist_add")
    if isinstance(tool_add, list) and tool_add:
        violations.append("configure_run.tool_widen")

    return violations


def sanitize_configure_run_overlay_strict(
    overlay: dict[str, Any],
    *,
    execution_mode: ExecutionMode,
    organizational: OrganizationalPolicyContext | None = None,
) -> dict[str, Any]:
    """Drop forbidden overlay keys in STRICT; raise when widen is attempted."""
    violations = validate_configure_run_overlay_strict(
        overlay,
        execution_mode=execution_mode,
        organizational=organizational,
    )
    if violations:
        raise ConfigureRunStrictViolation(violations)
    return dict(overlay)


def clamp_environment_overrides_strict(
    overrides: AgentEnvironmentOverrides | None,
    *,
    execution_mode: ExecutionMode,
    ceiling_tools: set[str],
) -> AgentEnvironmentOverrides | None:
    """
    In STRICT, request overrides cannot widen tools beyond contract+binding ceiling.

    ``tool_allowlist_remove`` remains allowed (narrowing only).
    """
    if overrides is None or execution_mode != ExecutionMode.STRICT:
        return overrides

    if not overrides.tool_allowlist_add:
        return overrides

    widened = [tool_id for tool_id in overrides.tool_allowlist_add if tool_id not in ceiling_tools]
    if widened:
        raise ConfigureRunStrictViolation(
            [f"environment_overrides.tool_widen:{tool_id}" for tool_id in widened],
        )
    return overrides
