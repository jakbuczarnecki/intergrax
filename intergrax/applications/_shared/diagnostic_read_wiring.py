# © Artur Czarnecki. All rights reserved.

"""Shared platform diagnostic read composition for Tier-3 product hosts (ONE-SPINE-2)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications._shared.diagnostic_composition import (
    DiagnosticCompositionOverrides,
    DiagnosticPersistenceComposition,
    build_default_execution_reconstruction_reader,
    resolve_diagnostic_persistence_composition,
)
from intergrax.applications._shared.harness_host_runtime import HarnessHostRuntime
from intergrax.applications._shared.harness_host_composition import (
    resolve_harness_host_runtime_event_persistence,
)
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


def resolve_host_diagnostic_read_dependencies(
    runtime: HarnessHostRuntime,
    *,
    overrides: DiagnosticCompositionOverrides | None = None,
) -> HostDiagnosticReadDependencies:
    """
    Resolve canonical diagnostic persistence from harness host runtime wiring.

    Uses the same document_store, runtime event store, and causal evidence adapters
    as platform queue-worker and diagnostic lifecycle composition — no dashboard-local stores.
    """
    resolved_overrides = (
        overrides
        if overrides is not None
        else runtime.env_wiring.composition.diagnostic_composition_overrides
    )
    wiring_context = runtime.env_wiring.composition.tool_wiring_context
    runtime_events = resolve_harness_host_runtime_event_persistence(runtime)

    persistence = resolve_diagnostic_persistence_composition(
        document_store=_document_store_from_wiring(wiring_context),
        runtime_event_persistence=runtime_events,
        overrides=resolved_overrides,
        require_durable=True,
    )
    if persistence is None:
        raise ValueError(
            "diagnostics-enabled product host requires platform document_store for shared "
            "ProblemPersistence and CausalEvidencePersistence",
        )

    document_store = _document_store_from_wiring(wiring_context)
    execution_lineage_reader = None
    if document_store is not None:
        from intergrax.integrations._shared.conformance import (
            assert_conditional_document_store,
        )

        store = assert_conditional_document_store(document_store)
        execution_lineage_reader = resolve_execution_lineage_persistence(
            document_store=store,
            provider=runtime.environment.reliability_profile.execution_lineage_persistence_provider,
        )

    return HostDiagnosticReadDependencies(
        persistence=persistence,
        execution_lineage_reader=execution_lineage_reader,
    )


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
    resolved_overrides = (
        overrides
        if overrides is not None
        else runtime.env_wiring.composition.diagnostic_composition_overrides
    )
    return build_diagnostic_read_service(
        resolve_host_diagnostic_read_dependencies(runtime, overrides=resolved_overrides),
        overrides=resolved_overrides,
    )


__all__ = [
    "HostDiagnosticReadDependencies",
    "build_diagnostic_read_service",
    "build_diagnostic_scope_discovery_service",
    "resolve_host_diagnostic_read_dependencies",
    "resolve_host_diagnostic_read_service",
]
