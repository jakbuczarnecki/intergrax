# © Artur Czarnecki. All rights reserved.

"""Shipped and registered security defense plugins (Phase SEC-BUNDLE-1)."""

from __future__ import annotations

from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.security.defense_plugin import (
    SecurityDefensePlugin,
    SecurityFailMode,
    SecurityInspectionResult,
)
from intergrax.runtime.hooks.hook_context import HookContext


class _StrictInjectionDefensePlugin:
    """Shipped bundle ``harness.strict_injection`` — extra blocked tool tokens."""

    plugin_id = "harness.strict_injection"
    version = "1.0.0"
    hook_points = frozenset({HookPoint.BEFORE_TOOL_CALL})
    priority = 56
    fail_mode = SecurityFailMode.FAIL_CLOSED

    _blocked_tokens = (
        "jailbreak",
        "bypass safety",
        "disable guardrails",
    )

    def inspect(self, point: HookPoint, ctx: HookContext) -> SecurityInspectionResult:
        if point != HookPoint.BEFORE_TOOL_CALL:
            return SecurityInspectionResult(allowed=True, plugin_id=self.plugin_id)
        arguments = ctx.runtime_state.get("arguments")
        if not isinstance(arguments, dict):
            return SecurityInspectionResult(allowed=True, plugin_id=self.plugin_id)
        blob = " ".join(str(value).lower() for value in arguments.values())
        for token in self._blocked_tokens:
            if token in blob:
                return SecurityInspectionResult(
                    allowed=False,
                    reasons=[f"blocked token: {token}"],
                    plugin_id=self.plugin_id,
                    hook_point=point.value,
                )
        return SecurityInspectionResult(allowed=True, plugin_id=self.plugin_id)


_SHIPPED: dict[str, SecurityDefensePlugin] = {
    _StrictInjectionDefensePlugin.plugin_id: _StrictInjectionDefensePlugin(),
}

_DYNAMIC: dict[str, SecurityDefensePlugin] = {}


def register_security_defense_plugin(
    plugin: SecurityDefensePlugin,
    *,
    override: bool = False,
) -> None:
    """Register a defense plugin instance (shipped or entry-point loaded)."""
    if plugin.plugin_id in _SHIPPED and not override:
        raise ValueError(f"cannot override shipped defense plugin: {plugin.plugin_id}")
    if plugin.plugin_id in _DYNAMIC and not override:
        raise ValueError(f"defense plugin already registered: {plugin.plugin_id}")
    _DYNAMIC[plugin.plugin_id] = plugin


def reset_security_defense_registry_for_tests() -> None:
    """Clear dynamically registered plugins between tests."""
    _DYNAMIC.clear()


def list_shipped_defense_bundle_ids() -> tuple[str, ...]:
    return tuple(_SHIPPED.keys())


def get_security_defense_plugin(plugin_id: str) -> SecurityDefensePlugin | None:
    if plugin_id in _SHIPPED:
        return _SHIPPED[plugin_id]
    return _DYNAMIC.get(plugin_id)


def resolve_security_defense_plugins(
    plugin_ids: tuple[str, ...],
    bundle_ids: tuple[str, ...],
) -> tuple[SecurityDefensePlugin, ...]:
    """Resolve explicit plugin ids and bundle ids to plugin instances."""
    resolved: list[SecurityDefensePlugin] = []
    seen: set[str] = set()
    for bundle_id in bundle_ids:
        plugin = get_security_defense_plugin(bundle_id)
        if plugin is None:
            continue
        if plugin.plugin_id not in seen:
            resolved.append(plugin)
            seen.add(plugin.plugin_id)
    for plugin_id in plugin_ids:
        if plugin_id in seen:
            continue
        plugin = get_security_defense_plugin(plugin_id)
        if plugin is not None:
            resolved.append(plugin)
            seen.add(plugin_id)
    return tuple(resolved)
