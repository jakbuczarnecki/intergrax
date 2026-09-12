# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Audit trail for decision system integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from intergrax.contracts.decision.integration.references import (
    PlatformDecisionLifecycleReference,
    ReferenceDecisionLifecycleReference,
)
from intergrax.contracts.decision.integration.result import (
    DecisionAdapterMetadata,
    DecisionIntegrationStatus,
)


@dataclass(frozen=True, slots=True)
class DecisionIntegrationAuditRecord:
    status: DecisionIntegrationStatus
    source: ReferenceDecisionLifecycleReference
    target: PlatformDecisionLifecycleReference | None
    adapter_metadata: DecisionAdapterMetadata
    mapping_detail: str

    def __post_init__(self) -> None:
        if type(self.status) is not DecisionIntegrationStatus:
            raise TypeError("status must be DecisionIntegrationStatus")
        if type(self.source) is not ReferenceDecisionLifecycleReference:
            raise TypeError("source must be ReferenceDecisionLifecycleReference")
        if (
            self.target is not None
            and type(self.target) is not PlatformDecisionLifecycleReference
        ):
            raise TypeError("target must be PlatformDecisionLifecycleReference or None")
        if type(self.adapter_metadata) is not DecisionAdapterMetadata:
            raise TypeError("adapter_metadata must be DecisionAdapterMetadata")
        if type(self.mapping_detail) is not str:
            raise TypeError("mapping_detail must be str")


@runtime_checkable
class DecisionIntegrationAuditProvider(Protocol):
    def record_integration(self, record: DecisionIntegrationAuditRecord) -> None: ...


__all__ = [
    "DecisionIntegrationAuditProvider",
    "DecisionIntegrationAuditRecord",
]
