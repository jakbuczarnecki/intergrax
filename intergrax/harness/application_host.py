# © Artur Czarnecki. All rights reserved.

"""Application host override protocol (Phase DX-5.1)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.runtime.hooks.hook_context import HookContext, HookResult
from intergrax.runtime.hooks.hook_point import HookPoint


@runtime_checkable
class ApplicationHost(Protocol):
    """Optional environment-level hooks mapped to Nexus ``HookPoint`` values."""

    def on_hook(self, point: HookPoint, context: HookContext) -> HookResult | None:
        """Return ``None`` to defer to default pipeline behavior."""
        ...
