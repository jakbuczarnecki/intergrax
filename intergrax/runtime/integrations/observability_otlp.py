# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""OTLP observability vendor integration (INTEGRATIONS-1C)."""

from __future__ import annotations

from pydantic import PrivateAttr

from intergrax.runtime.integrations.observability import (
    ObservabilityVendorIntegrationConfig,
    ObservabilityVendorIntegrationContract,
    ObservabilityVendorPayload,
    ObservabilityVendorSignal,
    require_policy_sanitized_envelope,
)
from intergrax.runtime.observability.export_boundary import (
    ExportRecordKind,
    ExportStatus,
    ObservabilityExportEnvelope,
)
from intergrax.runtime.observability.otlp_exporter import OtlpObservabilityExporter

OTLP_OBSERVABILITY_PROVIDER_ID = "otlp"


class OtlpObservabilityIntegrationConfig(ObservabilityVendorIntegrationConfig):
    """Typed config for OTLP observability vendor integration."""

    pass


def vendor_payload_to_export_envelope(payload: ObservabilityVendorPayload) -> ObservabilityExportEnvelope:
    """Rebuild a policy-sanitized export envelope from a vendor-neutral payload."""
    return ObservabilityExportEnvelope(
        record_kind=ExportRecordKind(payload.record_type),
        recorded_at=payload.recorded_at,
        run_id=payload.run_id,
        task_id=payload.task_id,
        agent_id=payload.agent_id,
        capability=payload.capability,
        tool_id=payload.tool_id,
        event_type=payload.event_type,
        status=ExportStatus(payload.status),
        latency_ms=payload.latency_ms,
        counts=dict(payload.counts),
        artifact_ref=payload.artifact_ref,
        sha256=payload.sha256,
        safe_relative_path=payload.safe_relative_path,
        schema_id=payload.schema_id_source,
        tenant_id=payload.tenant_id,
        workspace_id=payload.workspace_id,
        source_schema_id=payload.source_schema_id,
        correlation_id=payload.correlation_id,
        event_id=payload.event_id,
        sanitized_application_attributes=payload.sanitized_application_attributes,
    )


class OtlpObservabilityIntegration(ObservabilityVendorIntegrationContract):
    """
    OTLP observability vendor integration.

    Wraps OtlpObservabilityExporter and OtlpTransport as the lower-level delivery path.
    Consumes only policy-sanitized ObservabilityExportEnvelope records.
    """

    config: OtlpObservabilityIntegrationConfig = OtlpObservabilityIntegrationConfig()
    _exporter: OtlpObservabilityExporter = PrivateAttr()

    @classmethod
    def from_exporter(
        cls,
        exporter: OtlpObservabilityExporter,
        *,
        enabled: bool = True,
        supported_signals: tuple[ObservabilityVendorSignal, ...] | None = None,
    ) -> OtlpObservabilityIntegration:
        integration = cls.for_provider(
            provider_id=OTLP_OBSERVABILITY_PROVIDER_ID,
            supported_signals=supported_signals
            or (
                ObservabilityVendorSignal.EVENTS,
                ObservabilityVendorSignal.LOGS,
                ObservabilityVendorSignal.LLM_EVENTS,
            ),
            display_name="OpenTelemetry Protocol (OTLP)",
            config=OtlpObservabilityIntegrationConfig(enabled=enabled),
        )
        integration._exporter = exporter
        return integration

    @property
    def exporter(self) -> OtlpObservabilityExporter:
        return self._exporter

    async def export(self, envelope: ObservabilityExportEnvelope) -> None:
        if not self.config.enabled:
            return None
        safe_envelope = require_policy_sanitized_envelope(envelope)
        mapping = self.map_envelope(safe_envelope)
        if mapping.signal not in self.supported_signals:
            return None
        await self._exporter.export(safe_envelope)

    async def deliver_payload(self, payload: ObservabilityVendorPayload) -> None:
        envelope = vendor_payload_to_export_envelope(payload)
        await self._exporter.export(envelope)
