# © Artur Czarnecki. All rights reserved.

"""Diagnostic assembly validation and readiness contract (DIAG-FOUNDATION-1)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Sequence

from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    DiagnosticPosture,
)

if TYPE_CHECKING:
    from intergrax.applications._shared.scenario_runtime_profiles import ScenarioRuntimeMode


class DiagnosticReadiness(str, Enum):
    """Typed runtime fact for central diagnostic engine composition."""

    ATTACHED = "attached"
    NOT_REQUIRED_UNAVAILABLE = "not_required_unavailable"


@dataclass(frozen=True, slots=True)
class DiagnosticWiring:
    """Resolved diagnostic composition posture and attachment outcome."""

    required: bool
    attached: bool

    @property
    def readiness(self) -> DiagnosticReadiness:
        if self.attached:
            return DiagnosticReadiness.ATTACHED
        return DiagnosticReadiness.NOT_REQUIRED_UNAVAILABLE


class DiagnosticAssemblyError(ValueError):
    """Raised when required central diagnostics cannot be composed."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors: tuple[str, ...] = tuple(errors)
        message = "; ".join(self.errors)
        super().__init__(message)


def resolve_central_diagnostics_required(
    env: ApplicationEnvironmentProfile,
    *,
    scenario_runtime_mode: ScenarioRuntimeMode | None = None,
    explicit_posture: DiagnosticPosture | None = None,
) -> bool:
    """
    Canonical decision: does this runtime require central diagnostics?

    Single authoritative path for harness hosts and scenario runtimes.
    """
    if explicit_posture is not None:
        return explicit_posture is DiagnosticPosture.REQUIRED

    from intergrax.applications._shared.scenario_runtime_profiles import ScenarioRuntimeMode

    if scenario_runtime_mode is ScenarioRuntimeMode.PRODUCTION_ATTACHED:
        return True
    if env.application_profile is ApplicationProfile.PRODUCT:
        return True
    return env.diagnostic_profile.posture is DiagnosticPosture.REQUIRED


def diagnostic_assembly_errors(
    *,
    required: bool,
    attached: bool,
    missing_document_store: bool,
    missing_runtime_events: bool,
) -> tuple[str, ...]:
    """Build typed assembly errors when required diagnostics cannot attach."""
    if not required or attached:
        return ()
    errors: list[str] = []
    if missing_document_store:
        errors.append(
            "central diagnostics are required but no document store is available",
        )
    if missing_runtime_events:
        errors.append(
            "central diagnostics are required but RuntimeEvent persistence is unavailable",
        )
    if not errors:
        errors.append(
            "central diagnostics are required but terminal diagnostic trigger could not be attached",
        )
    return tuple(errors)


def assert_diagnostic_assembly_valid(
    *,
    required: bool,
    attached: bool,
    missing_document_store: bool,
    missing_runtime_events: bool,
) -> None:
    """Raise :class:`DiagnosticAssemblyError` when required diagnostics are unavailable."""
    errors = diagnostic_assembly_errors(
        required=required,
        attached=attached,
        missing_document_store=missing_document_store,
        missing_runtime_events=missing_runtime_events,
    )
    if errors:
        raise DiagnosticAssemblyError(errors)


__all__ = [
    "DiagnosticAssemblyError",
    "DiagnosticReadiness",
    "DiagnosticWiring",
    "assert_diagnostic_assembly_valid",
    "diagnostic_assembly_errors",
    "resolve_central_diagnostics_required",
]
