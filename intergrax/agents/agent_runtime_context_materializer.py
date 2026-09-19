# © Artur Czarnecki. All rights reserved.

"""Runtime/composition hook for legacy Nexus RuntimeContext materialization (EBH-2B)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest


@runtime_checkable
class AgentRuntimeContextMaterializer(Protocol):
    """Tier-2 implementation hook — not part of the public Agent contract surface."""

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        ...
