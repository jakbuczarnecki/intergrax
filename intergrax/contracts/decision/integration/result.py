# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Integration outcome contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from intergrax.contracts.decision.integration.references import (
    PlatformDecisionLifecycleReference,
    ReferenceDecisionLifecycleReference,
)


class DecisionIntegrationStatus(StrEnum):
    SUCCESS = "success"
    WARNING = "warning"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class DecisionAdapterMetadata:
    source_type: str
    adapter_id: str
    adapter_version: str
    mapping_version: str
    integrated_at: datetime

    def __post_init__(self) -> None:
        if not self.source_type.strip():
            raise ValueError("source_type must be non-empty")
        if not self.adapter_id.strip():
            raise ValueError("adapter_id must be non-empty")
        if not self.adapter_version.strip():
            raise ValueError("adapter_version must be non-empty")
        if not self.mapping_version.strip():
            raise ValueError("mapping_version must be non-empty")
        if self.integrated_at.tzinfo is None:
            raise ValueError("integrated_at must be timezone-aware")


@dataclass(frozen=True, slots=True)
class DecisionIntegrationResult:
    status: DecisionIntegrationStatus
    source: ReferenceDecisionLifecycleReference
    target: PlatformDecisionLifecycleReference | None
    adapter_metadata: DecisionAdapterMetadata
    detail: str

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
        if type(self.detail) is not str:
            raise TypeError("detail must be str")


__all__ = [
    "DecisionAdapterMetadata",
    "DecisionIntegrationResult",
    "DecisionIntegrationStatus",
]
