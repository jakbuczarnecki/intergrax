# © Artur Czarnecki. All rights reserved.

"""Runtime-aware auditability health projection for Tier-3 hosts (DIAG-FOUNDATION-2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from intergrax.applications._shared.diagnostic_assembly_resolver import (
    DiagnosticWiring,
    resolve_central_diagnostics_required,
)
from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.observability.auditability_health import (
    AuditabilityHealthSnapshot,
    build_auditability_health_snapshot,
)

if TYPE_CHECKING:
    from intergrax.applications._shared.harness_host_runtime import HarnessHostRuntime


@dataclass(frozen=True, slots=True)
class HostAuditabilityHealthFacts:
    """Thin runtime projection consumed by health and observability assembly."""

    diagnostics_required: bool
    diagnostics_attached: bool
    runtime_event_persistence_available: bool
    diagnostic_read_side_required: bool
    diagnostic_read_side_ready: bool


def project_host_auditability_health_facts(
    *,
    env: ApplicationEnvironmentProfile,
    diagnostic_wiring: DiagnosticWiring,
    runtime_event_persistence_available: bool,
    diagnostic_read_side_ready: bool,
) -> HostAuditabilityHealthFacts:
    """Project canonical runtime facts without inferring attachment from configuration."""
    diagnostics_pane_enabled = env.observability_profile.diagnostics_pane_enabled
    read_side_required = (
        env.application_profile is ApplicationProfile.PRODUCT and diagnostics_pane_enabled
    )
    return HostAuditabilityHealthFacts(
        diagnostics_required=diagnostic_wiring.required,
        diagnostics_attached=diagnostic_wiring.attached,
        runtime_event_persistence_available=runtime_event_persistence_available,
        diagnostic_read_side_required=read_side_required,
        diagnostic_read_side_ready=diagnostic_read_side_ready,
    )


def project_host_auditability_health_facts_from_runtime(
    runtime: HarnessHostRuntime,
    *,
    diagnostic_read_side_ready: bool,
) -> HostAuditabilityHealthFacts:
    """Resolve auditability facts from ``HarnessHostRuntime.diagnostic_wiring``."""
    runtime_events = runtime.observability.runtime_event_store
    if runtime_events is None:
        runtime_events = runtime.nexus_loop.runtime_event_store
    return project_host_auditability_health_facts(
        env=runtime.environment,
        diagnostic_wiring=runtime.diagnostic_wiring,
        runtime_event_persistence_available=runtime_events is not None,
        diagnostic_read_side_ready=diagnostic_read_side_ready,
    )


def project_conservative_auditability_health_facts(
    env: ApplicationEnvironmentProfile,
) -> HostAuditabilityHealthFacts:
    """
    Environment-only conservative projection when runtime facts are unavailable.

    Never claims diagnostic attachment or persistence without runtime wiring.
    """
    diagnostics_required = resolve_central_diagnostics_required(env)
    diagnostics_pane_enabled = env.observability_profile.diagnostics_pane_enabled
    read_side_required = (
        env.application_profile is ApplicationProfile.PRODUCT and diagnostics_pane_enabled
    )
    return HostAuditabilityHealthFacts(
        diagnostics_required=diagnostics_required,
        diagnostics_attached=False,
        runtime_event_persistence_available=False,
        diagnostic_read_side_required=read_side_required,
        diagnostic_read_side_ready=False,
    )


def project_auditability_health_snapshot(
    facts: HostAuditabilityHealthFacts,
) -> AuditabilityHealthSnapshot:
    """Build typed auditability health snapshot from runtime-projected facts."""
    return build_auditability_health_snapshot(
        diagnostics_required=facts.diagnostics_required,
        diagnostics_attached=facts.diagnostics_attached,
        runtime_event_persistence_available=facts.runtime_event_persistence_available,
        diagnostic_read_side_required=facts.diagnostic_read_side_required,
        diagnostic_read_side_ready=facts.diagnostic_read_side_ready,
    )


def assert_host_auditability_health_valid(
    facts: HostAuditabilityHealthFacts,
    env: ApplicationEnvironmentProfile,
) -> None:
    """Fail closed when a PRODUCT host reports required-but-unready auditability."""
    from intergrax.applications._shared.observability_assembly_resolver import (
        ObservabilityAssemblyError,
    )

    snapshot = project_auditability_health_snapshot(facts)
    if env.application_profile is not ApplicationProfile.PRODUCT:
        return
    if snapshot.auditability_ready:
        return
    if not snapshot.diagnostics_required:
        return
    raise ObservabilityAssemblyError(
        (
            "product host auditability is not ready: central diagnostics are required but "
            "runtime diagnostic wiring or RuntimeEvent persistence is unavailable",
        ),
    )


__all__ = [
    "HostAuditabilityHealthFacts",
    "assert_host_auditability_health_valid",
    "project_host_auditability_health_facts",
    "project_auditability_health_snapshot",
    "project_conservative_auditability_health_facts",
    "project_host_auditability_health_facts",
    "project_host_auditability_health_facts_from_runtime",
]
