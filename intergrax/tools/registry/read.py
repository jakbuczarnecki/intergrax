# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Read-only Tool registry boundary for execution composition."""

from __future__ import annotations

from typing import Protocol

from intergrax.tools.registry.provenance import ToolRuntimeActivationMetadata
from intergrax.tools.registry.runtime import RegisteredTool


class ToolRegistryRead(Protocol):
    """Runtime availability/read surface — not lifecycle mutation authority."""

    def has(self, tool_id: str) -> bool: ...

    def get(self, tool_id: str) -> RegisteredTool: ...

    def activation_metadata(self, tool_id: str) -> ToolRuntimeActivationMetadata | None: ...


__all__ = ["ToolRegistryRead"]
