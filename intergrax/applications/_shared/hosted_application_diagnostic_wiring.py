# © Artur Czarnecki. All rights reserved.

"""Product-owned HostedApplication diagnostic event composition (HOST-DIAG-3)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from intergrax.applications._shared.hosted_application_failure_projection import (
    hosted_application_failure_to_problem_signal,
)
from intergrax.hosting.contracts.context import HostedApplicationEventPublisher
from intergrax.hosting.contracts.events import (
    HostedApplicationEvent,
    HostedApplicationEventType,
)
from intergrax.hosting.eventing import ObservabilityHostedApplicationEventPublisher
from intergrax.runtime.diagnostics.deterministic_problem_grouping import STRATEGY_ID
from intergrax.runtime.diagnostics.diagnostic_orchestration_models import (
    DiagnosticOrchestrationRequest,
    DiagnosticSignalSubjectScope,
)
from intergrax.runtime.diagnostics.diagnostic_orchestrator import DiagnosticOrchestrator
from intergrax.runtime.observability.export_boundary import ObservabilityExporter
from intergrax.runtime.observability.export_policy import ObservabilityExportPolicy

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class HostedDiagnosticTenantBinding:
    """Explicit product-owned tenant scope for hosted application diagnostics."""

    tenant_id: str

    def __post_init__(self) -> None:
        normalized = self.tenant_id.strip()
        if not normalized:
            raise ValueError("tenant_id must be non-empty")
        if normalized != self.tenant_id:
            object.__setattr__(self, "tenant_id", normalized)


class HostedApplicationDiagnosticEventPublisher:
    """
    Compose observability export with central non-execution diagnostics.

    Canonical hosting events are published to observability first; APPLICATION_FAILED
    may additionally project into DiagnosticOrchestrator when bounded failure facts exist.
    """

    def __init__(
        self,
        observability_publisher: HostedApplicationEventPublisher,
        tenant_binding: HostedDiagnosticTenantBinding,
        orchestrator: DiagnosticOrchestrator,
    ) -> None:
        self._observability_publisher = observability_publisher
        self._tenant_binding = tenant_binding
        self._orchestrator = orchestrator

    async def publish(self, event: HostedApplicationEvent) -> None:
        await self._observability_publisher.publish(event)
        if event.event_type is not HostedApplicationEventType.APPLICATION_FAILED:
            return
        try:
            signal = hosted_application_failure_to_problem_signal(event)
            if signal is None:
                return
            scope = DiagnosticSignalSubjectScope(
                tenant_id=self._tenant_binding.tenant_id,
                application_id=event.application_id,
                instance_id=event.instance_id,
                problem_signals=(signal,),
            )
            request = DiagnosticOrchestrationRequest(
                tenant_id=self._tenant_binding.tenant_id,
                grouping_strategy_id=STRATEGY_ID,
                observed_at=event.occurred_at,
                signal_subjects=(scope,),
            )
            self._orchestrator.run(request)
        except Exception:
            _LOGGER.exception(
                "hosted application diagnostic projection failed",
                extra={
                    "hosted_fields": {
                        "application_id": event.application_id,
                        "instance_id": event.instance_id,
                        "event_id": event.event_id,
                    },
                },
            )


def build_hosted_application_diagnostic_event_publisher(
    *,
    tenant_binding: HostedDiagnosticTenantBinding,
    orchestrator: DiagnosticOrchestrator,
    observability_exporter: ObservabilityExporter | None = None,
    observability_policy: ObservabilityExportPolicy | None = None,
) -> HostedApplicationEventPublisher:
    """Build composed publisher: observability export then bounded diagnostic projection."""
    observability = ObservabilityHostedApplicationEventPublisher(
        observability_exporter,
        policy=observability_policy or ObservabilityExportPolicy(enabled=True),
    )
    return HostedApplicationDiagnosticEventPublisher(
        observability_publisher=observability,
        tenant_binding=tenant_binding,
        orchestrator=orchestrator,
    )


__all__ = [
    "HostedApplicationDiagnosticEventPublisher",
    "HostedDiagnosticTenantBinding",
    "build_hosted_application_diagnostic_event_publisher",
]
