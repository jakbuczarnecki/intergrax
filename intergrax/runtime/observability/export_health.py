# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Process-local observability exporter health/degradation contract (OBS-HEALTH-lite)."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

_SAFE_EXPORT_FAILURE_REASONS: frozenset[str] = frozenset(
    {
        "exporter_failed",
        "timeout",
        "transport_error",
    }
)


class ObservabilityExporterHealthStatus(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"


@dataclass(frozen=True, slots=True)
class ObservabilityExporterHealthSnapshot:
    """Immutable operator-visible exporter health snapshot (process-local scope)."""

    exporter_id: str
    status: ObservabilityExporterHealthStatus
    consecutive_failures: int
    last_attempt_at: datetime
    last_success_at: datetime | None
    last_failure_at: datetime | None
    last_failure_reason: str | None
    recovery_count: int = 0


def normalize_export_failure_reason(reason: str) -> str:
    """Return a safe, vendor-neutral failure reason for operator health snapshots."""
    normalized = reason.strip()
    if normalized in _SAFE_EXPORT_FAILURE_REASONS:
        return normalized
    return "exporter_failed"


class ObservabilityExporterHealthRegistry:
    """In-process exporter health tracker — operator state, not runtime truth."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshots: dict[str, ObservabilityExporterHealthSnapshot] = {}

    def record_success(self, exporter_id: str, observed_at: datetime) -> None:
        with self._lock:
            current = self._snapshots.get(exporter_id)
            if current is None:
                self._snapshots[exporter_id] = ObservabilityExporterHealthSnapshot(
                    exporter_id=exporter_id,
                    status=ObservabilityExporterHealthStatus.HEALTHY,
                    consecutive_failures=0,
                    last_attempt_at=observed_at,
                    last_success_at=observed_at,
                    last_failure_at=None,
                    last_failure_reason=None,
                    recovery_count=0,
                )
                return

            recovery_count = current.recovery_count
            if current.status is ObservabilityExporterHealthStatus.DEGRADED:
                recovery_count += 1

            self._snapshots[exporter_id] = ObservabilityExporterHealthSnapshot(
                exporter_id=exporter_id,
                status=ObservabilityExporterHealthStatus.HEALTHY,
                consecutive_failures=0,
                last_attempt_at=observed_at,
                last_success_at=observed_at,
                last_failure_at=current.last_failure_at,
                last_failure_reason=current.last_failure_reason,
                recovery_count=recovery_count,
            )

    def record_failure(
        self,
        exporter_id: str,
        reason: str,
        observed_at: datetime,
    ) -> None:
        safe_reason = normalize_export_failure_reason(reason)
        with self._lock:
            current = self._snapshots.get(exporter_id)
            if current is None:
                self._snapshots[exporter_id] = ObservabilityExporterHealthSnapshot(
                    exporter_id=exporter_id,
                    status=ObservabilityExporterHealthStatus.DEGRADED,
                    consecutive_failures=1,
                    last_attempt_at=observed_at,
                    last_success_at=None,
                    last_failure_at=observed_at,
                    last_failure_reason=safe_reason,
                    recovery_count=0,
                )
                return

            self._snapshots[exporter_id] = ObservabilityExporterHealthSnapshot(
                exporter_id=exporter_id,
                status=ObservabilityExporterHealthStatus.DEGRADED,
                consecutive_failures=current.consecutive_failures + 1,
                last_attempt_at=observed_at,
                last_success_at=current.last_success_at,
                last_failure_at=observed_at,
                last_failure_reason=safe_reason,
                recovery_count=current.recovery_count,
            )

    def get(self, exporter_id: str) -> ObservabilityExporterHealthSnapshot | None:
        with self._lock:
            snapshot = self._snapshots.get(exporter_id)
            return snapshot

    def list(self) -> tuple[ObservabilityExporterHealthSnapshot, ...]:
        with self._lock:
            return tuple(self._snapshots.values())
