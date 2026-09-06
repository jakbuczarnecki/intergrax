# © Artur Czarnecki. All rights reserved.

"""Shared platform diagnostic read composition for Tier-3 product hosts (ONE-SPINE-2)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications._shared.diagnostic_cursor_secret import (
    resolve_problem_list_cursor_secret,
)
from intergrax.applications._shared.harness_host_runtime import HarnessHostRuntime
from intergrax.applications._shared.harness_host_runtime_compat import (
    resolve_harness_host_nexus_loop_legacy,
)
from intergrax.integrations._shared.conformance import assert_conditional_document_store
from intergrax.runtime.diagnostics.diagnostic_read_service import DiagnosticReadService
from intergrax.runtime.diagnostics.diagnostic_scope_discovery_service import (
    DiagnosticScopeDiscoveryService,
)
from intergrax.runtime.diagnostics.document_store_problem_occurrence_persistence import (
    wire_problem_occurrence_persistence,
)
from intergrax.runtime.diagnostics.document_store_problem_persistence import wire_problem_persistence
from intergrax.runtime.diagnostics.execution_reconstruction import ExecutionReconstructor
from intergrax.runtime.diagnostics.problem_occurrence_persistence import ProblemOccurrencePersistence
from intergrax.runtime.diagnostics.problem_persistence import ProblemPersistence
from intergrax.runtime.diagnostics.providers.causal_transport_scope_provider import (
    CAUSAL_TRANSPORT_SCOPE_PROVIDER_ID,
    CausalTransportScopeProvider,
)
from intergrax.runtime.diagnostics.providers.problem_scope_provider import ProblemScopeProvider
from intergrax.runtime.events.persistence_contract import RuntimeEventPersistence
from intergrax.runtime.observability.causal_evidence_persistence import CausalEvidencePersistence
from intergrax.runtime.observability.document_store_causal_evidence_persistence import (
    wire_causal_evidence_persistence,
)


@dataclass(frozen=True, slots=True)
class HostDiagnosticReadDependencies:
    """Shared platform diagnostic persistence resolved from one harness host runtime."""

    problem_persistence: ProblemPersistence
    occurrence_persistence: ProblemOccurrencePersistence
    runtime_event_persistence: RuntimeEventPersistence
    causal_evidence_persistence: CausalEvidencePersistence


def resolve_host_diagnostic_read_dependencies(
    runtime: HarnessHostRuntime,
) -> HostDiagnosticReadDependencies:
    """
    Resolve canonical diagnostic persistence from harness host runtime wiring.

    Uses the same document_store, runtime event store, and causal evidence adapters
    as platform queue-worker and diagnostic lifecycle composition — no dashboard-local stores.
    """
    build_context = runtime.env_wiring.build_context
    wiring_context = build_context.tool_wiring_context
    if wiring_context is None or wiring_context.document_store is None:
        raise ValueError(
            "diagnostics-enabled product host requires platform document_store for shared "
            "ProblemPersistence and CausalEvidencePersistence",
        )
    document_store = assert_conditional_document_store(wiring_context.document_store)

    runtime_events = runtime.observability.runtime_event_store
    if runtime_events is None:
        runtime_events = resolve_harness_host_nexus_loop_legacy(runtime).runtime_event_store
    if runtime_events is None:
        raise ValueError(
            "diagnostics-enabled product host requires RuntimeEventPersistence from "
            "harness observability wiring",
        )

    return HostDiagnosticReadDependencies(
        problem_persistence=wire_problem_persistence(
            document_store=document_store,
            list_cursor_secret=resolve_problem_list_cursor_secret(),
        ),
        occurrence_persistence=wire_problem_occurrence_persistence(
            document_store=document_store,
            occurrence_cursor_secret=resolve_problem_list_cursor_secret(),
        ),
        runtime_event_persistence=runtime_events,
        causal_evidence_persistence=wire_causal_evidence_persistence(
            document_store=document_store,
        ),
    )


def build_diagnostic_read_service(
    dependencies: HostDiagnosticReadDependencies,
) -> DiagnosticReadService:
    """Construct canonical DiagnosticReadService over shared platform persistence."""
    return DiagnosticReadService(
        problem_persistence=dependencies.problem_persistence,
        occurrence_persistence=dependencies.occurrence_persistence,
        execution_reconstructor=ExecutionReconstructor(
            runtime_events=dependencies.runtime_event_persistence,
            causal_evidence=dependencies.causal_evidence_persistence,
        ),
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
        ),
    )


def resolve_host_diagnostic_read_service(
    runtime: HarnessHostRuntime,
) -> DiagnosticReadService:
    """Resolve shared DiagnosticReadService for product host observability surfaces."""
    return build_diagnostic_read_service(resolve_host_diagnostic_read_dependencies(runtime))


__all__ = [
    "HostDiagnosticReadDependencies",
    "build_diagnostic_read_service",
    "build_diagnostic_scope_discovery_service",
    "resolve_host_diagnostic_read_dependencies",
    "resolve_host_diagnostic_read_service",
]
