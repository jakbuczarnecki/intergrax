# © Artur Czarnecki. All rights reserved.

"""Auditability / diagnostic readiness health contract (DIAG-FOUNDATION-2)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class AuditabilityDiagnosticReadiness(str, Enum):
    """Typed diagnostic attachment posture for health consumers."""

    ATTACHED = "attached"
    NOT_REQUIRED_UNAVAILABLE = "not_required_unavailable"


class AuditabilityHealthSnapshot(BaseModel):
    """Canonical auditability health slice derived from runtime diagnostic facts."""

    schema_version: str = "1.0.0"
    diagnostics_required: bool
    diagnostics_attached: bool
    diagnostic_readiness: AuditabilityDiagnosticReadiness
    runtime_event_persistence_available: bool
    auditability_ready: bool


def resolve_auditability_ready(
    *,
    diagnostics_required: bool,
    diagnostics_attached: bool,
    runtime_event_persistence_available: bool,
    diagnostic_read_side_required: bool = False,
    diagnostic_read_side_ready: bool = True,
) -> bool:
    """
    Canonical auditability-ready rule.

    When diagnostics are not required, optional unavailability does not block readiness.
    When required, RuntimeEvent persistence and attached write-side wiring must be present.
    When the diagnostics read pane is required, read-side availability must also hold.
    """
    if not diagnostics_required:
        return True
    if not runtime_event_persistence_available or not diagnostics_attached:
        return False
    if diagnostic_read_side_required and not diagnostic_read_side_ready:
        return False
    return True


def build_auditability_health_snapshot(
    *,
    diagnostics_required: bool,
    diagnostics_attached: bool,
    runtime_event_persistence_available: bool,
    diagnostic_read_side_required: bool = False,
    diagnostic_read_side_ready: bool = True,
) -> AuditabilityHealthSnapshot:
    """Build typed auditability health from runtime-projected facts."""
    readiness = (
        AuditabilityDiagnosticReadiness.ATTACHED
        if diagnostics_attached
        else AuditabilityDiagnosticReadiness.NOT_REQUIRED_UNAVAILABLE
    )
    return AuditabilityHealthSnapshot(
        diagnostics_required=diagnostics_required,
        diagnostics_attached=diagnostics_attached,
        diagnostic_readiness=readiness,
        runtime_event_persistence_available=runtime_event_persistence_available,
        auditability_ready=resolve_auditability_ready(
            diagnostics_required=diagnostics_required,
            diagnostics_attached=diagnostics_attached,
            runtime_event_persistence_available=runtime_event_persistence_available,
            diagnostic_read_side_required=diagnostic_read_side_required,
            diagnostic_read_side_ready=diagnostic_read_side_ready,
        ),
    )


__all__ = [
    "AuditabilityDiagnosticReadiness",
    "AuditabilityHealthSnapshot",
    "build_auditability_health_snapshot",
    "resolve_auditability_ready",
]
