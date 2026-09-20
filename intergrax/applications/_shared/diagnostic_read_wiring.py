# © Artur Czarnecki. All rights reserved.

"""Shared platform diagnostic read composition for Tier-3 product hosts (ONE-SPINE-2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from intergrax.applications._shared.diagnostic_composition import (
    DiagnosticCompositionError,
    DiagnosticCompositionOverrides,
    DiagnosticPersistenceComposition,
    build_default_execution_reconstruction_reader,
    resolve_diagnostic_persistence_composition,
)
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
)
from intergrax.applications._shared.environment_wiring import ApplicationEnvironmentWiring
from intergrax.runtime.nexus.observability_wiring import NexusObservabilityStores

if TYPE_CHECKING:
    from intergrax.applications._shared.harness_host_runtime import HarnessHostRuntime
from intergrax.contracts.execution_lineage import ExecutionLineageReader
from intergrax.contracts.execution_reconstruction import ExecutionReconstructionReader
from intergrax.runtime.diagnostics.diagnostic_read_service import DiagnosticReadService
from intergrax.runtime.diagnostics.diagnostic_scope_discovery_service import (
    DiagnosticScopeDiscoveryService,
)
from intergrax.runtime.diagnostics.problem_occurrence_persistence import (
    ProblemOccurrencePersistence,
)
from intergrax.runtime.diagnostics.problem_persistence import ProblemPersistence
from intergrax.runtime.diagnostics.providers.causal_transport_scope_provider import (
    CausalTransportScopeProvider,
)
from intergrax.runtime.diagnostics.providers.problem_scope_provider import (
    ProblemScopeProvider,
)
from intergrax.runtime.diagnostics.providers.runtime_event_scope_provider import (
    RuntimeEventScopeProvider,
)
from intergrax.runtime.events.persistence_contract import RuntimeEventPersistence
from intergrax.runtime.execution.lineage.wiring import (
    resolve_execution_lineage_persistence,
)
from intergrax.runtime.observability.causal_evidence_persistence import (
    CausalEvidencePersistence,
)
from intergrax.tools.registry.wiring import ToolWiringContext


@dataclass(frozen=True, slots=True)
class HostDiagnosticReadDependencies:
    """Shared platform diagnostic persistence resolved from one harness host runtime."""

    persistence: DiagnosticPersistenceComposition
    execution_lineage_reader: ExecutionLineageReader | None = None

    @property
    def problem_persistence(self) -> ProblemPersistence:
        return self.persistence.problem_persistence

    @property
    def occurrence_persistence(self) -> ProblemOccurrencePersistence:
        return self.persistence.occurrence_persistence

    @property
    def runtime_event_persistence(self) -> RuntimeEventPersistence:
        return self.persistence.runtime_event_persistence

    @property
    def causal_evidence_persistence(self) -> CausalEvidencePersistence:
        return self.persistence.causal_evidence_persistence


def _document_store_from_wiring(
    tool_wiring_context: ToolWiringContext | None,
) -> object | None:
    if tool_wiring_context is None:
        return None
    return tool_wiring_context.document_store


_EMPTY_DIAGNOSTIC_COMPOSITION_OVERRIDES = DiagnosticCompositionOverrides()


def _configured_host_composition_overrides(
    runtime: HarnessHostRuntime,
) -> DiagnosticCompositionOverrides:
    configured = runtime.env_wiring.composition.diagnostic_composition_overrides
    if configured is None:
        return _EMPTY_DIAGNOSTIC_COMPOSITION_OVERRIDES
    return configured


def assert_host_diagnostic_composition_frozen(
    runtime: HarnessHostRuntime,
    *,
    overrides: DiagnosticCompositionOverrides | None,
) -> None:
    """
    Fail closed when a host-bound caller attempts post-build diagnostic reconfiguration.

    Canonical harness hosts freeze ``DiagnosticCompositionOverrides`` at construction.
    """
    if overrides is None:
        return
    if overrides != _configured_host_composition_overrides(runtime):
        raise DiagnosticCompositionError(
            "diagnostic composition is frozen at host construction; "
            "conflicting post-build overrides are unsupported",
        )


def materialize_host_diagnostic_read_dependencies(
    *,
    env_wiring: ApplicationEnvironmentWiring,
    observability: NexusObservabilityStores,
    environment: ApplicationEnvironmentProfile,
    overrides: DiagnosticCompositionOverrides | None = None,
    require_durable: bool = False,
) -> HostDiagnosticReadDependencies | None:
    """
    Resolve canonical host diagnostic persistence once for write/read/scope discovery.

    Returns ``None`` when durable persistence is unavailable and ``require_durable`` is false.
    """
    resolved_overrides = (
        overrides
        if overrides is not None
        else env_wiring.composition.diagnostic_composition_overrides
    )
    wiring_context = env_wiring.composition.tool_wiring_context
    runtime_events = observability.runtime_event_store

    persistence = resolve_diagnostic_persistence_composition(
        document_store=_document_store_from_wiring(wiring_context),
        runtime_event_persistence=runtime_events,
        overrides=resolved_overrides,
        require_durable=require_durable,
    )
    if persistence is None:
        return None

    document_store = _document_store_from_wiring(wiring_context)
    execution_lineage_reader = None
    if document_store is not None:
        from intergrax.integrations._shared.conformance import (
            assert_conditional_document_store,
        )

        store = assert_conditional_document_store(document_store)
        execution_lineage_reader = resolve_execution_lineage_persistence(
            document_store=store,
            provider=environment.reliability_profile.execution_lineage_persistence_provider,
        )

    return HostDiagnosticReadDependencies(
        persistence=persistence,
        execution_lineage_reader=execution_lineage_reader,
    )


def resolve_host_diagnostic_read_dependencies(
    runtime: HarnessHostRuntime,
    *,
    overrides: DiagnosticCompositionOverrides | None = None,
) -> HostDiagnosticReadDependencies:
    """
    Resolve canonical diagnostic persistence from harness host runtime wiring.

    Uses host-materialized dependencies when present (canonical factory path).
    """
    from intergrax.applications._shared.harness_host_runtime import HarnessHostRuntime as _HarnessHostRuntime

    if isinstance(runtime, _HarnessHostRuntime):
        stored = runtime.host_diagnostic_dependencies
        if stored is not None:
            assert_host_diagnostic_composition_frozen(runtime, overrides=overrides)
            return stored

    resolved_overrides = (
        overrides
        if overrides is not None
        else runtime.env_wiring.composition.diagnostic_composition_overrides
    )
    dependencies = materialize_host_diagnostic_read_dependencies(
        env_wiring=runtime.env_wiring,
        observability=runtime.observability,
        environment=runtime.environment,
        overrides=resolved_overrides,
        require_durable=True,
    )
    if dependencies is None:
        raise ValueError(
            "diagnostics-enabled product host requires platform document_store for shared "
            "ProblemPersistence and CausalEvidencePersistence",
        )
    return dependencies


def build_diagnostic_read_service(
    dependencies: HostDiagnosticReadDependencies,
    *,
    overrides: DiagnosticCompositionOverrides | None = None,
    execution_reconstruction_reader: ExecutionReconstructionReader | None = None,
) -> DiagnosticReadService:
    """Construct canonical DiagnosticReadService over shared platform persistence."""
    if execution_reconstruction_reader is not None:
        reader = execution_reconstruction_reader
    elif overrides is not None and overrides.execution_reconstruction_reader is not None:
        reader = overrides.execution_reconstruction_reader
    else:
        reader = build_default_execution_reconstruction_reader(
            dependencies.persistence,
            execution_lineage_reader=dependencies.execution_lineage_reader,
        )
    return DiagnosticReadService(
        problem_persistence=dependencies.problem_persistence,
        occurrence_persistence=dependencies.occurrence_persistence,
        execution_reconstructor=reader,
    )


def build_diagnostic_scope_discovery_service(
    dependencies: HostDiagnosticReadDependencies,
) -> DiagnosticScopeDiscoveryService:
    """Construct canonical scope discovery over shared platform diagnostic persistence."""
    return DiagnosticScopeDiscoveryService(
        providers=(
            ProblemScopeProvider(
                problem_persistence=dependencies.problem_persistence,
                occurrence_persistence=dependencies.occurrence_persistence,
            ),
            CausalTransportScopeProvider(
                causal_evidence_persistence=dependencies.causal_evidence_persistence,
            ),
            RuntimeEventScopeProvider(
                runtime_event_persistence=dependencies.runtime_event_persistence,
            ),
        ),
    )


def resolve_host_diagnostic_read_service(
    runtime: HarnessHostRuntime,
    *,
    overrides: DiagnosticCompositionOverrides | None = None,
) -> DiagnosticReadService:
    """Resolve shared DiagnosticReadService for product host observability surfaces."""
    resolved_overrides = _configured_host_composition_overrides(runtime)
    if overrides is not None:
        assert_host_diagnostic_composition_frozen(runtime, overrides=overrides)
        resolved_overrides = overrides
    dependencies = resolve_host_diagnostic_read_dependencies(runtime, overrides=overrides)
    return build_diagnostic_read_service(
        dependencies,
        overrides=resolved_overrides,
    )


__all__ = [
    "HostDiagnosticReadDependencies",
    "assert_host_diagnostic_composition_frozen",
    "build_diagnostic_read_service",
    "build_diagnostic_scope_discovery_service",
    "materialize_host_diagnostic_read_dependencies",
    "resolve_host_diagnostic_read_dependencies",
    "resolve_host_diagnostic_read_service",
]
