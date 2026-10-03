# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Runtime plugin bootstrap (§42.22)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List

from intergrax.contracts.host_orchestration_wiring_capabilities import (
    HostOrchestrationRuntimeEventPort,
)
from intergrax.runtime.plugins.compatibility import (
    RuntimePluginCompatibilityError,
    evaluate_runtime_plugin_compatibility,
)
from intergrax.runtime.plugins.contract import RuntimePlugin
from intergrax.runtime.schema.registry import current_runtime_version


class RuntimePluginPolicyRegistrationUnsupportedError(RuntimeError):
    """Governance policy rules are not mutable at runtime plugin bootstrap."""


@dataclass
class PluginBootstrapResult:
    shutdown_callbacks: List[Callable[[], None]] = field(default_factory=list)


def bootstrap_runtime_plugins(
    plugins: List[RuntimePlugin],
    *,
    event_bus: HostOrchestrationRuntimeEventPort,
) -> PluginBootstrapResult:
    """
    Register Tier-3 plugins at application startup.

    Returns shutdown callbacks for FastAPI lifespan handlers.
    """
    runtime = current_runtime_version()
    for plugin in plugins:
        result = evaluate_runtime_plugin_compatibility(plugin, runtime)
        if not result.compatible:
            raise RuntimePluginCompatibilityError(result)

    shutdowns: List[Callable[[], None]] = []
    for plugin in plugins:
        if plugin.register is not None:
            plugin.register(event_bus)
        if plugin.on_shutdown is not None:
            shutdowns.append(plugin.on_shutdown)
    return PluginBootstrapResult(shutdown_callbacks=shutdowns)
