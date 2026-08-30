# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Vendor-neutral observability export routing/fanout (OBS-ROUTING-0).

``FanoutObservabilityExporter`` operates on envelopes that have already passed
``ObservabilityExportPolicy`` sanitization (typically via
``try_export_observability_envelope``). It does **not** call
``apply_observability_export_policy`` itself.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from intergrax.runtime.observability.export_boundary import (
    ExportRecordKind,
    ObservabilityExportEnvelope,
    ObservabilityExporter,
)
from intergrax.runtime.observability.export_health import (
    ObservabilityExporterHealthRegistry,
    normalize_export_failure_reason,
)

logger = logging.getLogger(__name__)


def _record_route_health_success(
    *,
    health_registry: ObservabilityExporterHealthRegistry | None,
    route_id: str,
    observed_at: datetime,
) -> None:
    if health_registry is None:
        return
    try:
        health_registry.record_success(route_id, observed_at)
    except Exception:
        logger.warning(
            "observability export route health success recording failed route_id=%s",
            route_id,
            exc_info=True,
        )


def _record_route_health_failure(
    *,
    health_registry: ObservabilityExporterHealthRegistry | None,
    route_id: str,
    reason: str,
    observed_at: datetime,
) -> None:
    if health_registry is None:
        return
    try:
        health_registry.record_failure(
            route_id,
            normalize_export_failure_reason(reason),
            observed_at,
        )
    except Exception:
        logger.warning(
            "observability export route health failure recording failed route_id=%s",
            route_id,
            exc_info=True,
        )


@dataclass(frozen=True, slots=True)
class ObservabilityExportRoute:
    """Logical export destination selected by operator/platform wiring."""

    route_id: str
    exporter: ObservabilityExporter
    enabled: bool = True
    record_kinds: tuple[ExportRecordKind, ...] = ()
    problem_kinds: tuple[str, ...] = ()
    problem_severities: tuple[str, ...] = ()
    problem_error_codes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ObservabilityRouteDeliveryResult:
    route_id: str
    selected: bool
    exported: bool
    reason: str


@dataclass(frozen=True, slots=True)
class ObservabilityFanoutResult:
    selected_count: int
    exported_count: int
    failed_count: int
    skipped_count: int
    deliveries: tuple[ObservabilityRouteDeliveryResult, ...]


def route_matches_envelope(
    route: ObservabilityExportRoute,
    envelope: ObservabilityExportEnvelope,
) -> tuple[bool, str]:
    """Return whether a route matches an envelope and a skip reason when it does not."""
    if not route.enabled:
        return False, "route_disabled"
    if route.record_kinds and envelope.record_kind not in route.record_kinds:
        return False, "record_kind_not_matched"
    if route.problem_kinds and envelope.problem_kind not in route.problem_kinds:
        return False, "problem_kind_not_matched"
    if route.problem_severities and envelope.problem_severity not in route.problem_severities:
        return False, "problem_severity_not_matched"
    if route.problem_error_codes and envelope.problem_error_code not in route.problem_error_codes:
        return False, "problem_error_code_not_matched"
    return True, ""


class FanoutObservabilityExporter:
    """Fan out a policy-safe envelope to zero/one/many configured exporters.

    Expects envelopes that have already been sanitized by export policy.
    Per-route exporter failures are isolated and never propagate to callers.
    """

    def __init__(
        self,
        routes: tuple[ObservabilityExportRoute, ...] | list[ObservabilityExportRoute],
        *,
        health_registry: ObservabilityExporterHealthRegistry | None = None,
    ) -> None:
        self._routes: tuple[ObservabilityExportRoute, ...] = tuple(routes)
        self._health_registry = health_registry
        self.last_result: ObservabilityFanoutResult | None = None

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    async def export(self, envelope: ObservabilityExportEnvelope) -> None:
        await self.export_with_result(envelope)

    async def export_with_result(
        self,
        envelope: ObservabilityExportEnvelope,
    ) -> ObservabilityFanoutResult:
        deliveries: list[ObservabilityRouteDeliveryResult] = []
        selected_count = 0
        exported_count = 0
        failed_count = 0
        skipped_count = 0

        for route in self._routes:
            matched, skip_reason = route_matches_envelope(route, envelope)
            if not matched:
                skipped_count += 1
                deliveries.append(
                    ObservabilityRouteDeliveryResult(
                        route_id=route.route_id,
                        selected=False,
                        exported=False,
                        reason=skip_reason,
                    )
                )
                continue

            selected_count += 1
            attempt_at = self._utc_now()
            try:
                await route.exporter.export(envelope)
            except Exception:
                failed_count += 1
                _record_route_health_failure(
                    health_registry=self._health_registry,
                    route_id=route.route_id,
                    reason="exporter_failed",
                    observed_at=attempt_at,
                )
                deliveries.append(
                    ObservabilityRouteDeliveryResult(
                        route_id=route.route_id,
                        selected=True,
                        exported=False,
                        reason="exporter_failed",
                    )
                )
                continue

            _record_route_health_success(
                health_registry=self._health_registry,
                route_id=route.route_id,
                observed_at=attempt_at,
            )

            exported_count += 1
            deliveries.append(
                ObservabilityRouteDeliveryResult(
                    route_id=route.route_id,
                    selected=True,
                    exported=True,
                    reason="exported",
                )
            )

        result = ObservabilityFanoutResult(
            selected_count=selected_count,
            exported_count=exported_count,
            failed_count=failed_count,
            skipped_count=skipped_count,
            deliveries=tuple(deliveries),
        )
        self.last_result = result
        return result
