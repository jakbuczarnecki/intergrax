# © Artur Czarnecki. All rights reserved.

"""Execution-engine hook for legacy Nexus RuntimeContext materialization (EBH-2B / HARNESS-01-R4).

Owned by Execution Engine internal adapter layer — not a public Agent / plugin contract.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest


@runtime_checkable
class AgentRuntimeContextMaterializer(Protocol):
    """Execution-engine internal UAEP hook — not part of the public Agent contract surface."""

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        ...
