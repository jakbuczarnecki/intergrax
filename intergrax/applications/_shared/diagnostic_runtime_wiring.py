# © Artur Czarnecki. All rights reserved.

"""Shared platform diagnostic runtime composition for Tier-3 harness hosts (ONE-SPINE-3)."""

from __future__ import annotations

from intergrax.applications._shared.diagnostic_composition import (
    DiagnosticCompositionOverrides,
    DiagnosticPersistenceComposition,
    build_diagnostic_orchestrator_from_composition,
    resolve_diagnostic_composition,
    resolve_diagnostic_persistence_composition,
)
from intergrax.applications._shared.diagnostic_read_wiring import (
    HostDiagnosticReadDependencies,
    resolve_host_diagnostic_read_dependencies,
)
from intergrax.applications._shared.diagnostic_assembly_resolver import (
    DiagnosticWiring,
    assert_diagnostic_assembly_valid,
    resolve_central_diagnostics_required,
)
from intergrax.applications._shared.environment_wiring import ApplicationEnvironmentWiring
from intergrax.applications._shared.harness_host_runtime import HarnessHostRuntime
from intergrax.runtime.diagnostics.diagnostic_orchestrator import DiagnosticOrchestrator
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
)
from intergrax.contracts.diagnostics.terminal_execution_diagnostic_port import (
    TerminalExecutionDiagnosticPort,
)
from intergrax.runtime.diagnostics.central_terminal_execution_diagnostic_port import (
    wrap_terminal_execution_diagnostic_trigger,
)
from intergrax.runtime.diagnostics.terminal_execution_diagnostic_trigger import (
    TerminalExecutionDiagnosticTrigger,
)
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.observability_wiring import NexusObservabilityStores


def _resolve_overrides(
    env_wiring: ApplicationEnvironmentWiring,
    overrides: DiagnosticCompositionOverrides | None,
) -> DiagnosticCompositionOverrides | None:
    if overrides is not None:
        return overrides
    return env_wiring.composition.diagnostic_composition_overrides


def resolve_host_diagnostic_runtime_dependencies(
    *,
    env_wiring: ApplicationEnvironmentWiring,
    observability: NexusObservabilityStores,
    overrides: DiagnosticCompositionOverrides | None = None,
) -> HostDiagnosticReadDependencies | None:
    """
    Resolve shared diagnostic persistence for runtime write orchestration.

    Returns ``None`` when the host lacks required platform document-store capabilities
    and no durable overrides were supplied.
    """
    resolved_overrides = _resolve_overrides(env_wiring, overrides)
    wiring_context = env_wiring.composition.tool_wiring_context
    document_store = None
    if wiring_context is not None:
        document_store = wiring_context.document_store

    persistence = resolve_diagnostic_persistence_composition(
        document_store=document_store,
        runtime_event_persistence=observability.runtime_event_store,
        overrides=resolved_overrides,
        require_durable=False,
    )
    if persistence is None:
        return None
    return HostDiagnosticReadDependencies(persistence=persistence)


def build_diagnostic_orchestrator(
    dependencies: HostDiagnosticReadDependencies,
    *,
    overrides: DiagnosticCompositionOverrides | None = None,
) -> DiagnosticOrchestrator:
    """Construct canonical ``DiagnosticOrchestrator`` over shared platform persistence."""
    composition = resolve_diagnostic_composition(
        dependencies.persistence,
        overrides=overrides,
        execution_lineage_reader=dependencies.execution_lineage_reader,
    )
    return build_diagnostic_orchestrator_from_composition(composition)


def build_terminal_execution_diagnostic_trigger(
    dependencies: HostDiagnosticReadDependencies,
    *,
    overrides: DiagnosticCompositionOverrides | None = None,
) -> TerminalExecutionDiagnosticTrigger:
    """Construct production terminal diagnostic trigger over shared orchestrator."""
    return TerminalExecutionDiagnosticTrigger(
        build_diagnostic_orchestrator(dependencies, overrides=overrides),
    )


def try_build_terminal_execution_diagnostic_trigger(
    *,
    env_wiring: ApplicationEnvironmentWiring,
    observability: NexusObservabilityStores,
    overrides: DiagnosticCompositionOverrides | None = None,
) -> TerminalExecutionDiagnosticTrigger | None:
    """Best-effort runtime trigger when required platform storage is available."""
    resolved_overrides = _resolve_overrides(env_wiring, overrides)
    dependencies = resolve_host_diagnostic_runtime_dependencies(
        env_wiring=env_wiring,
        observability=observability,
        overrides=resolved_overrides,
    )
    if dependencies is None:
        return None
    return build_terminal_execution_diagnostic_trigger(
        dependencies,
        overrides=resolved_overrides,
    )


def build_terminal_execution_diagnostic_port(
    dependencies: HostDiagnosticReadDependencies,
    *,
    event_bus: RuntimeEventBus | None = None,
    overrides: DiagnosticCompositionOverrides | None = None,
) -> TerminalExecutionDiagnosticPort:
    """Construct production terminal diagnostic port over shared orchestrator."""
    return wrap_terminal_execution_diagnostic_trigger(
        build_terminal_execution_diagnostic_trigger(
            dependencies,
            overrides=overrides,
        ),
        event_bus=event_bus,
    )


def try_build_terminal_execution_diagnostic_port(
    *,
    env_wiring: ApplicationEnvironmentWiring,
    observability: NexusObservabilityStores,
    event_bus: RuntimeEventBus | None = None,
    overrides: DiagnosticCompositionOverrides | None = None,
) -> TerminalExecutionDiagnosticPort | None:
    """Best-effort neutral port when required platform storage is available."""
    resolved_overrides = _resolve_overrides(env_wiring, overrides)
    dependencies = resolve_host_diagnostic_runtime_dependencies(
        env_wiring=env_wiring,
        observability=observability,
        overrides=resolved_overrides,
    )
    if dependencies is None:
        return None
    return build_terminal_execution_diagnostic_port(
        dependencies,
        event_bus=event_bus,
        overrides=resolved_overrides,
    )


def _diagnostic_prerequisite_gaps(
    *,
    env_wiring: ApplicationEnvironmentWiring,
    observability: NexusObservabilityStores,
    overrides: DiagnosticCompositionOverrides | None = None,
) -> tuple[bool, bool]:
    resolved_overrides = _resolve_overrides(env_wiring, overrides)
    wiring_context = env_wiring.composition.tool_wiring_context
    has_document_store = (
        wiring_context is not None and wiring_context.document_store is not None
    )
    has_persistence_override = False
    if resolved_overrides is not None:
        has_persistence_override = (
            resolved_overrides.problem_persistence is not None
            and resolved_overrides.occurrence_persistence is not None
            and resolved_overrides.causal_evidence_persistence is not None
        )
    missing_document_store = not has_document_store and not has_persistence_override
    missing_runtime_events = observability.runtime_event_store is None
    return missing_document_store, missing_runtime_events


def wire_terminal_execution_diagnostics(
    *,
    env: ApplicationEnvironmentProfile,
    env_wiring: ApplicationEnvironmentWiring,
    observability: NexusObservabilityStores,
    nexus_loop: NexusLoop,
    scenario_runtime_mode: object | None = None,
    overrides: DiagnosticCompositionOverrides | None = None,
) -> DiagnosticWiring:
    """
    Policy-aware terminal diagnostic composition over the canonical orchestrator spine.

    When diagnostics are required, missing prerequisites fail closed.
    """
    resolved_overrides = _resolve_overrides(env_wiring, overrides)
    required = resolve_central_diagnostics_required(
        env,
        scenario_runtime_mode=scenario_runtime_mode,  # type: ignore[arg-type]
    )
    missing_document_store, missing_runtime_events = _diagnostic_prerequisite_gaps(
        env_wiring=env_wiring,
        observability=observability,
        overrides=resolved_overrides,
    )
    terminal_diagnostic_port = try_build_terminal_execution_diagnostic_port(
        env_wiring=env_wiring,
        observability=observability,
        event_bus=nexus_loop.event_bus,
        overrides=resolved_overrides,
    )
    attached = terminal_diagnostic_port is not None
    assert_diagnostic_assembly_valid(
        required=required,
        attached=attached,
        missing_document_store=missing_document_store,
        missing_runtime_events=missing_runtime_events,
    )
    if attached:
        nexus_loop.attach_terminal_diagnostic_trigger(terminal_diagnostic_port)
    return DiagnosticWiring(required=required, attached=attached)


def resolve_host_terminal_execution_diagnostic_trigger(
    runtime: HarnessHostRuntime,
    *,
    overrides: DiagnosticCompositionOverrides | None = None,
) -> TerminalExecutionDiagnosticTrigger:
    """Resolve production terminal diagnostic trigger from harness host runtime wiring."""
    resolved_overrides = _resolve_overrides(runtime.env_wiring, overrides)
    return build_terminal_execution_diagnostic_trigger(
        resolve_host_diagnostic_read_dependencies(
            runtime,
            overrides=resolved_overrides,
        ),
        overrides=resolved_overrides,
    )


__all__ = [
    "DiagnosticCompositionOverrides",
    "DiagnosticPersistenceComposition",
    "build_diagnostic_orchestrator",
    "build_terminal_execution_diagnostic_port",
    "build_terminal_execution_diagnostic_trigger",
    "resolve_host_diagnostic_runtime_dependencies",
    "resolve_host_terminal_execution_diagnostic_trigger",
    "try_build_terminal_execution_diagnostic_port",
    "try_build_terminal_execution_diagnostic_trigger",
    "wire_terminal_execution_diagnostics",
]
