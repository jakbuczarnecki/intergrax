# © Artur Czarnecki. All rights reserved.

"""Tier-2 protocols for tool enablement without importing ``intergrax.tools`` (Phase AA tier hygiene)."""

from __future__ import annotations

from typing import Protocol


class ToolEnablementProfile(Protocol):
    """Subset of Tier-0 ``ToolProfile`` used by harness reference agents."""

    def is_tool_enabled(self, tool_id: str) -> bool:
        """Return whether ``tool_id`` is enabled on the host tool profile."""


class ToolWiringContextLike(Protocol):
    """Opaque Tier-3 tool wiring context passed through ``RuntimeConfig``."""

    pass
