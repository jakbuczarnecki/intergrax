# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Runtime plugin bootstrap (§42.22)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from intergrax.runtime.hooks.hook_registry import HookRegistry
from intergrax.runtime.plugins.contract import (
    PolicyEngineLike,
    RuntimeEventBusLike,
    RuntimePlugin,
)


@dataclass
class PluginBootstrapResult:
    shutdown_callbacks: List[Callable[[], None]] = field(default_factory=list)


def bootstrap_runtime_plugins(
    plugins: List[RuntimePlugin],
    *,
    event_bus: RuntimeEventBusLike,
    hook_registry: HookRegistry,
    policy_engine: Optional[PolicyEngineLike] = None,
) -> PluginBootstrapResult:
    """
    Register Tier-3 plugins at application startup.

    Returns shutdown callbacks for FastAPI lifespan / on_event handlers.
    """
    policy = policy_engine or _NullPolicyEngine()
    shutdowns: List[Callable[[], None]] = []
    for plugin in plugins:
        if plugin.register is not None:
            plugin.register(event_bus, hook_registry, policy)
        if plugin.on_shutdown is not None:
            shutdowns.append(plugin.on_shutdown)
    return PluginBootstrapResult(shutdown_callbacks=shutdowns)


class _NullPolicyEngine:
    def register_rule(self, rule: object) -> None:
        _ = rule
