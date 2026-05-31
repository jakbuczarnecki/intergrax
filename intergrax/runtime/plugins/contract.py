# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Runtime plugin contract (§42.22, Appendix B.07)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, runtime_checkable

from intergrax.runtime.hooks.hook_registry import HookRegistry
from intergrax.runtime.schema.registry import RuntimeVersionInfo, current_runtime_version


@runtime_checkable
class PolicyEngineLike(Protocol):
    """Minimal surface plugins may extend (avoid importing heavy policy modules)."""

    def register_rule(self, rule: object) -> None: ...


@runtime_checkable
class RuntimeEventBusLike(Protocol):
    def subscribe(
        self,
        handler: Callable[..., object],
        *,
        event_types: object | None = None,
        priority: int = 100,
        subscription_id: str | None = None,
    ) -> str: ...


@dataclass(frozen=True)
class RuntimePlugin:
    """
    Tier-3 bootstrap plugin (§42.22).

    Plugins register hooks and optional policy rules at application startup.
    They MUST NOT import Tier-2 agent domain modules.
    """

    plugin_id: str
    version: str
    compatible_runtime: RuntimeVersionInfo = field(default_factory=current_runtime_version)
    register: Optional[
        Callable[[RuntimeEventBusLike, HookRegistry, PolicyEngineLike], None]
    ] = None
    on_shutdown: Optional[Callable[[], None]] = None
