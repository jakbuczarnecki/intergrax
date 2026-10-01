# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Runtime plugin contract (§42.22, Appendix B.07)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from intergrax.contracts.host_orchestration_wiring_capabilities import (
    HostOrchestrationRuntimeEventPort,
)
from intergrax.runtime.schema.registry import RuntimeVersionInfo, current_runtime_version

RuntimePluginRegisterCallback = Callable[[HostOrchestrationRuntimeEventPort], None]


@dataclass(frozen=True)
class RuntimePlugin:
    """
    Tier-3 bootstrap plugin (§42.22).

    Plugins subscribe to the runtime event spine at application startup.
    They MUST NOT import Tier-2 agent domain modules.
    """

    plugin_id: str
    version: str
    compatible_runtime: RuntimeVersionInfo = field(default_factory=current_runtime_version)
    register: Optional[RuntimePluginRegisterCallback] = None
    on_shutdown: Optional[Callable[[], None]] = None
