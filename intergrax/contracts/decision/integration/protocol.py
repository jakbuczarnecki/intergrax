# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Adapter and provider protocols for the decision integration boundary."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.decision.integration.references import (
    ReferenceDecisionLifecycleReference,
)
from intergrax.contracts.decision.integration.result import (
    DecisionIntegrationResult,
)


@runtime_checkable
class DecisionSystemIntegrationAdapter(Protocol):
    """Maps one reference source type to platform decision contracts."""

    @property
    def adapter_id(self) -> str: ...

    @property
    def adapter_version(self) -> str: ...

    @property
    def mapping_version(self) -> str: ...

    @property
    def source_type(self) -> str: ...


@runtime_checkable
class DecisionLifecycleIntegrationAdapter(DecisionSystemIntegrationAdapter, Protocol):
    def integrate_lifecycle(
        self,
        source: ReferenceDecisionLifecycleReference,
    ) -> DecisionIntegrationResult: ...


@runtime_checkable
class DecisionIntegrationAdapterProvider(Protocol):
    """Plugin registry — engine depends only on this port."""

    def provide_lifecycle_adapter(
        self,
        source_type: str,
    ) -> DecisionLifecycleIntegrationAdapter | None: ...


__all__ = [
    "DecisionIntegrationAdapterProvider",
    "DecisionLifecycleIntegrationAdapter",
    "DecisionSystemIntegrationAdapter",
]
