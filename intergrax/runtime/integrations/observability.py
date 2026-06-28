# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Observability vendor integration category contract (INTEGRATIONS-1B)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationConfig,
    PlatformIntegrationContract,
    PlatformIntegrationKind,
    derive_platform_integration_id,
)
from intergrax.runtime.observability.export_attributes import (
    SanitizedApplicationObservabilityAttributes,
)
from intergrax.runtime.observability.export_boundary import (
    ObservabilityExportEnvelope,
    envelope_is_content_safe,
)

OBSERVABILITY_VENDOR_INTEGRATION_CONTRACT_SCHEMA = "observability_vendor_integration_contract.v1"
OBSERVABILITY_VENDOR_PAYLOAD_SCHEMA = "observability_vendor_payload.v1"


class ObservabilityVendorSignal(StrEnum):
    """Observability signal families supported by vendor integrations."""

    EVENTS = "events"
    LOGS = "logs"
    TRACES = "traces"
    METRICS = "metrics"
    LLM_EVENTS = "llm_events"


class ObservabilityVendorKind(StrEnum):
    """Well-known observability vendor provider_id slugs — category-specific classes only."""

    LANGFUSE = "langfuse"
    ARIZE = "arize"
    PHOENIX = "phoenix"
    ELASTICSEARCH = "elasticsearch"
    CUSTOM = "custom"


_DEFAULT_OBSERVABILITY_VENDOR_SIGNALS: tuple[ObservabilityVendorSignal, ...] = (
    ObservabilityVendorSignal.EVENTS,
    ObservabilityVendorSignal.LOGS,
    ObservabilityVendorSignal.TRACES,
    ObservabilityVendorSignal.METRICS,
    ObservabilityVendorSignal.LLM_EVENTS,
)

_DEFAULT_OBSERVABILITY_VENDOR_CAPABILITIES: tuple[PlatformIntegrationCapability, ...] = (
    PlatformIntegrationCapability.EXPORT,
    PlatformIntegrationCapability.HEALTH_CHECK,
)


class ObservabilityVendorIntegrationConfig(PlatformIntegrationConfig):
    """Typed config for observability vendor integrations — secrets stay out of payloads."""

    pass


class ObservabilityVendorPayload(BaseModel):
    """Vendor-neutral, policy-safe payload mapped from ObservabilityExportEnvelope."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["observability_vendor_payload.v1"] = OBSERVABILITY_VENDOR_PAYLOAD_SCHEMA
    provider_id: str
    integration_id: str
    integration_kind: str
    record_type: str
    recorded_at: datetime

    run_id: str = ""
    task_id: str = ""
    agent_id: str = ""
    capability: str = ""
    event_type: str = ""
    status: str = ""
    tool_id: str = ""

    latency_ms: int | None = None
    counts: dict[str, int] = Field(default_factory=dict)

    artifact_ref: str = ""
    sha256: str = ""
    safe_relative_path: str = ""
    schema_id_source: str = ""

    tenant_id: str = ""
    workspace_id: str = ""
    source_schema_id: str = ""
    correlation_id: str = ""
    event_id: str = ""

    sanitized_application_attributes: SanitizedApplicationObservabilityAttributes | None = None


class ObservabilityVendorMappingResult(BaseModel):
    """Result of mapping a policy-sanitized envelope to a vendor-neutral payload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    payload: ObservabilityVendorPayload
    signal: ObservabilityVendorSignal


def require_policy_sanitized_envelope(envelope: ObservabilityExportEnvelope) -> ObservabilityExportEnvelope:
    """
    Reject envelopes that bypass export policy or still carry raw application attributes.

    Vendor integrations must receive envelopes from apply_observability_export_policy /
    try_export_observability_envelope — never directly from RuntimeEvent or raw builders
    with application_attributes populated.
    """
    if envelope.application_attributes is not None:
        msg = "raw application_attributes must not be consumed by observability vendor integrations"
        raise ValueError(msg)
    if not envelope_is_content_safe(envelope):
        msg = "envelope failed content safety check"
        raise ValueError(msg)
    return envelope


def map_envelope_to_vendor_payload(
    envelope: ObservabilityExportEnvelope,
    *,
    provider_id: str,
    integration_id: str,
    integration_kind: str = PlatformIntegrationKind.OBSERVABILITY_VENDOR.value,
) -> ObservabilityVendorMappingResult:
    """Map a policy-sanitized ObservabilityExportEnvelope to ObservabilityVendorPayload."""
    safe_envelope = require_policy_sanitized_envelope(envelope)
    record_type = safe_envelope.record_kind.value
    signal = _signal_for_record_type(record_type)

    payload = ObservabilityVendorPayload(
        provider_id=provider_id,
        integration_id=integration_id,
        integration_kind=integration_kind,
        record_type=record_type,
        recorded_at=safe_envelope.recorded_at,
        run_id=safe_envelope.run_id,
        task_id=safe_envelope.task_id,
        agent_id=safe_envelope.agent_id,
        capability=safe_envelope.capability,
        event_type=safe_envelope.event_type,
        status=safe_envelope.status.value,
        tool_id=safe_envelope.tool_id,
        latency_ms=safe_envelope.latency_ms,
        counts=dict(safe_envelope.counts),
        artifact_ref=safe_envelope.artifact_ref,
        sha256=safe_envelope.sha256,
        safe_relative_path=safe_envelope.safe_relative_path,
        schema_id_source=safe_envelope.schema_id,
        tenant_id=safe_envelope.tenant_id,
        workspace_id=safe_envelope.workspace_id,
        source_schema_id=safe_envelope.source_schema_id,
        correlation_id=safe_envelope.correlation_id,
        event_id=safe_envelope.event_id,
        sanitized_application_attributes=safe_envelope.sanitized_application_attributes,
    )
    return ObservabilityVendorMappingResult(payload=payload, signal=signal)


def _signal_for_record_type(record_type: str) -> ObservabilityVendorSignal:
    if record_type == "llm_call":
        return ObservabilityVendorSignal.LLM_EVENTS
    if record_type == "journal_ref":
        return ObservabilityVendorSignal.LOGS
    if record_type in {"tool_call", "rag_call", "runtime_event", "diagnostic"}:
        return ObservabilityVendorSignal.EVENTS
    return ObservabilityVendorSignal.EVENTS


class ObservabilityVendorIntegrationContract(PlatformIntegrationContract):
    """
    Category-specific contract for observability vendor integrations.

    Concrete vendors (Langfuse, Arize, Phoenix, Elasticsearch, custom backends)
    subclass this type — one integration class per category. The same provider_id
    may appear in other categories through separate integration classes.

    Implements the ObservabilityExporter protocol via export() — map_envelope() produces
    vendor-neutral payloads; deliver_payload() is overridden by concrete integrations.
    """

    schema_id: Literal["observability_vendor_integration_contract.v1"] = (
        OBSERVABILITY_VENDOR_INTEGRATION_CONTRACT_SCHEMA
    )
    integration_kind: str = PlatformIntegrationKind.OBSERVABILITY_VENDOR.value
    supported_signals: tuple[ObservabilityVendorSignal, ...] = Field(
        default_factory=lambda: _DEFAULT_OBSERVABILITY_VENDOR_SIGNALS
    )
    config: ObservabilityVendorIntegrationConfig = Field(
        default_factory=ObservabilityVendorIntegrationConfig
    )

    @classmethod
    def for_provider(
        cls,
        *,
        provider_id: str,
        supported_signals: tuple[ObservabilityVendorSignal, ...] = _DEFAULT_OBSERVABILITY_VENDOR_SIGNALS,
        capabilities: tuple[PlatformIntegrationCapability, ...] = _DEFAULT_OBSERVABILITY_VENDOR_CAPABILITIES,
        display_name: str | None = None,
        version: str | None = None,
        config: ObservabilityVendorIntegrationConfig | None = None,
    ) -> ObservabilityVendorIntegrationContract:
        return cls(
            integration_id=derive_platform_integration_id(
                provider_id,
                PlatformIntegrationKind.OBSERVABILITY_VENDOR.value,
            ),
            provider_id=provider_id,
            display_name=display_name,
            version=version,
            capabilities=capabilities,
            supported_signals=supported_signals,
            config=config or ObservabilityVendorIntegrationConfig(),
        )

    def map_envelope(self, envelope: ObservabilityExportEnvelope) -> ObservabilityVendorMappingResult:
        """Map a policy-sanitized envelope to a vendor-neutral payload."""
        return map_envelope_to_vendor_payload(
            envelope,
            provider_id=self.provider_id,
            integration_id=self.integration_id,
            integration_kind=self.integration_kind,
        )

    async def export(self, envelope: ObservabilityExportEnvelope) -> None:
        """
        Satisfy ObservabilityExporter — validate, map, and deliver vendor-safe payload.

        Subclasses override deliver_payload(); network/SDK I/O belongs there only.
        """
        if not self.config.enabled:
            return None
        mapping = self.map_envelope(envelope)
        if mapping.signal not in self.supported_signals:
            return None
        await self.deliver_payload(mapping.payload)

    async def deliver_payload(self, payload: ObservabilityVendorPayload) -> None:
        """Send a mapped payload to the vendor backend — override in concrete integrations."""
        raise NotImplementedError(
            f"{type(self).__name__} must override deliver_payload() for vendor I/O"
        )

    def public_view(self) -> Mapping[str, Any]:
        view = dict(super().public_view())
        view["supported_signals"] = [signal.value for signal in self.supported_signals]
        return view
