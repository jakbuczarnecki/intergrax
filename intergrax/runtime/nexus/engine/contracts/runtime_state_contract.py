# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations  # noqa: I001
from abc import ABC, abstractmethod
from typing import List, Optional, TYPE_CHECKING  # noqa: F401, UP035

from intergrax.runtime.nexus.tracing.trace_models import (
    DiagnosticPayload,
    TraceComponent,
    TraceLevel,
)

if TYPE_CHECKING:
    from intergrax.runtime.nexus.artifacts.models import TraceArtifactRef


class RuntimeStateContract(ABC):
    """
    Minimal contract required by runtime tool layer.

    This contract intentionally exposes only the surface
    needed by idempotency pre-effect coordination and related components.
    """

    # --- Tenant ---

    @property
    @abstractmethod
    def tenant_id(self) -> str:
        """
        Must return non-empty tenant identifier.
        """
        ...

    # --- Tracing ---

    @abstractmethod
    def trace_event(
        self,
        *,
        component: TraceComponent,
        step: str,
        message: str,
        level: TraceLevel = TraceLevel.INFO,
        payload: DiagnosticPayload | None = None,
        artifact_refs: list[TraceArtifactRef] | None = None,
    ) -> None: ...
