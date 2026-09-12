# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Audit trail for decision system integration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from intergrax.contracts.decision.integration.audit_sink import (
    DecisionAuditSink,
    DecisionIntegrationAuditEnvelope,
)
from intergrax.contracts.decision.integration.metadata import (
    IntegrationAuditProviderMetadata,
)
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


@dataclass(frozen=True, slots=True)
class DefaultDecisionIntegrationAuditProvider:
    """Default audit plugin — no sink until platform composes a recording provider."""

    def record_integration(self, record: DecisionIntegrationAuditRecord) -> None:
        if type(record) is not DecisionIntegrationAuditRecord:
            raise TypeError("record must be DecisionIntegrationAuditRecord")


RECORDING_DECISION_INTEGRATION_AUDIT_PROVIDER_ID = (
    "decision.integration.audit.recording"
)
RECORDING_DECISION_INTEGRATION_AUDIT_PROVIDER_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class RecordingDecisionIntegrationAuditProvider:
    """Production audit plugin — forwards records to an injected sink without blocking."""

    sink: DecisionAuditSink
    provider_id: str = RECORDING_DECISION_INTEGRATION_AUDIT_PROVIDER_ID
    provider_version: str = RECORDING_DECISION_INTEGRATION_AUDIT_PROVIDER_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.sink, DecisionAuditSink):
            raise TypeError("sink must implement DecisionAuditSink")
        if not self.provider_id.strip():
            raise ValueError("provider_id must be non-empty")
        if not self.provider_version.strip():
            raise ValueError("provider_version must be non-empty")

    def record_integration(self, record: DecisionIntegrationAuditRecord) -> None:
        if type(record) is not DecisionIntegrationAuditRecord:
            raise TypeError("record must be DecisionIntegrationAuditRecord")
        envelope = DecisionIntegrationAuditEnvelope(
            record=record,
            provider_metadata=IntegrationAuditProviderMetadata(
                provider_id=self.provider_id,
                provider_version=self.provider_version,
                recorded_at=datetime.now(tz=UTC),
            ),
        )
        self.sink.append(envelope)


__all__ = [
    "DecisionIntegrationAuditProvider",
    "DecisionIntegrationAuditRecord",
    "DefaultDecisionIntegrationAuditProvider",
    "RECORDING_DECISION_INTEGRATION_AUDIT_PROVIDER_ID",
    "RECORDING_DECISION_INTEGRATION_AUDIT_PROVIDER_VERSION",
    "RecordingDecisionIntegrationAuditProvider",
]
