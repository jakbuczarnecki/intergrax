# © Artur Czarnecki. All rights reserved.

"""Contract-driven diagnostic host composition (OBS-DIAG-X2).

Resolves pluggable persistence / reconstruction / grouping providers into typed
dependencies for the single canonical ``DiagnosticOrchestrator`` spine.

Does not introduce a second diagnostic authority, service locator, or vendor
branching. Hard invariants (orchestrator, lifecycle engine, assessment schema,
lifecycle anomaly analysis) remain platform-owned.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from intergrax.applications._shared.diagnostic_cursor_secret import (
    resolve_problem_list_cursor_secret,
)
from intergrax.contracts.diagnostics.problem_persistence import ProblemPersistence
from intergrax.contracts.execution_lineage import ExecutionLineageReader
from intergrax.contracts.execution_reconstruction import ExecutionReconstructionReader
from intergrax.integrations._shared.conformance import assert_conditional_document_store
from intergrax.runtime.diagnostics.deterministic_problem_grouping import (
    DeterministicProblemGroupingStrategy,
)
from intergrax.runtime.diagnostics.diagnostic_assessment import DiagnosticAssessmentBuilder
from intergrax.runtime.diagnostics.diagnostic_orchestrator import DiagnosticOrchestrator
from intergrax.runtime.diagnostics.diagnostic_problem_grouping_feature_projector import (
    DiagnosticProblemGroupingFeatureProjector,
)
from intergrax.runtime.diagnostics.document_store_problem_occurrence_persistence import (
    wire_problem_occurrence_persistence,
)
from intergrax.runtime.diagnostics.document_store_problem_persistence import (
    wire_problem_persistence,
)
from intergrax.runtime.diagnostics.lifecycle_analysis import LifecycleAnomalyAnalyzer
from intergrax.runtime.diagnostics.problem_grouping import (
    ProblemGroupingEngine,
    ProblemGroupingStrategy,
    ProblemGroupingStrategyRegistry,
)
from intergrax.runtime.diagnostics.problem_lifecycle import ProblemLifecycleEngine
from intergrax.runtime.diagnostics.problem_occurrence_persistence import (
    ProblemOccurrencePersistence,
)
from intergrax.runtime.events.persistence_contract import RuntimeEventPersistence
from intergrax.runtime.observability.causal_evidence_persistence import (
    CausalEvidencePersistence,
)
from intergrax.runtime.observability.document_store_causal_evidence_persistence import (
    wire_causal_evidence_persistence,
)
from intergrax.runtime.observability.reconstruction import ExecutionReconstructor
from intergrax.tools.registry.wiring import ToolWiringContext


class DiagnosticCompositionError(ValueError):
    """Raised when diagnostic composition cannot be resolved fail-closed."""


class DiagnosticProviderResolutionError(DiagnosticCompositionError):
    """Raised when a required diagnostic provider cannot be resolved."""


class DiagnosticComponentOwnership(StrEnum):
    """Lifecycle ownership for injected diagnostic composition resources."""

    HOST_CREATED = "host_created"
    BORROWED = "borrowed"


@runtime_checkable
class SupportsClose(Protocol):
    """Optional close capability for host-created composition resources."""

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class DiagnosticCompositionOverrides:
    """
    Typed host-supplied replacements for pluggable diagnostic mechanisms.

    All fields are optional. Absent fields resolve to platform defaults.
    Never carries ``DiagnosticOrchestrator`` or Problem lifecycle authority.
    """

    problem_persistence: ProblemPersistence | None = None
    occurrence_persistence: ProblemOccurrencePersistence | None = None
    causal_evidence_persistence: CausalEvidencePersistence | None = None
    execution_reconstruction_reader: ExecutionReconstructionReader | None = None
    additional_grouping_strategies: tuple[ProblemGroupingStrategy, ...] = ()


@dataclass(frozen=True, slots=True)
class DiagnosticPersistenceComposition:
    """One resolved persistence bundle shared by write and read paths."""

    problem_persistence: ProblemPersistence
    occurrence_persistence: ProblemOccurrencePersistence
    causal_evidence_persistence: CausalEvidencePersistence
    runtime_event_persistence: RuntimeEventPersistence
    problem_persistence_ownership: DiagnosticComponentOwnership
    occurrence_persistence_ownership: DiagnosticComponentOwnership
    causal_evidence_persistence_ownership: DiagnosticComponentOwnership


@dataclass(frozen=True, slots=True)
class ResolvedDiagnosticComposition:
    """Typed resolved dependencies for canonical diagnostic orchestration."""

    persistence: DiagnosticPersistenceComposition
    execution_reconstruction_reader: ExecutionReconstructionReader
    reconstruction_ownership: DiagnosticComponentOwnership
    grouping_registry: ProblemGroupingStrategyRegistry


def resolve_diagnostic_persistence_composition(
    *,
    document_store: object | None,
    runtime_event_persistence: RuntimeEventPersistence | None,
    overrides: DiagnosticCompositionOverrides | None = None,
    list_cursor_secret: bytes | None = None,
    require_durable: bool = False,
) -> DiagnosticPersistenceComposition | None:
    """
    Resolve one persistence composition for both write and read wiring.

    Platform defaults use ``DocumentStore`` contracts only — no vendor imports.
    Custom overrides are borrowed and never replaced by in-memory fallbacks.
    """
    overrides = overrides or DiagnosticCompositionOverrides()

    if runtime_event_persistence is None:
        if require_durable:
            raise DiagnosticProviderResolutionError(
                "central diagnostics require RuntimeEventPersistence",
            )
        return None

    problem_persistence: ProblemPersistence | None = overrides.problem_persistence
    occurrence_persistence: ProblemOccurrencePersistence | None = (
        overrides.occurrence_persistence
    )
    causal_evidence_persistence: CausalEvidencePersistence | None = (
        overrides.causal_evidence_persistence
    )
    problem_ownership = DiagnosticComponentOwnership.BORROWED
    occurrence_ownership = DiagnosticComponentOwnership.BORROWED
    causal_ownership = DiagnosticComponentOwnership.BORROWED

    needs_document_store = (
        problem_persistence is None
        or occurrence_persistence is None
        or causal_evidence_persistence is None
    )
    if needs_document_store:
        if document_store is None:
            if require_durable:
                raise DiagnosticProviderResolutionError(
                    "central diagnostics require durable document_store persistence; "
                    "in-memory ProblemPersistence fallback is forbidden",
                )
            return None
        store = assert_conditional_document_store(document_store)
        secret = (
            list_cursor_secret
            if list_cursor_secret is not None
            else resolve_problem_list_cursor_secret()
        )
        if problem_persistence is None:
            problem_persistence = wire_problem_persistence(
                document_store=store,
                list_cursor_secret=secret,
            )
            problem_ownership = DiagnosticComponentOwnership.HOST_CREATED
        if occurrence_persistence is None:
            occurrence_persistence = wire_problem_occurrence_persistence(
                document_store=store,
                occurrence_cursor_secret=secret,
            )
            occurrence_ownership = DiagnosticComponentOwnership.HOST_CREATED
        if causal_evidence_persistence is None:
            causal_evidence_persistence = wire_causal_evidence_persistence(
                document_store=store,
            )
            causal_ownership = DiagnosticComponentOwnership.HOST_CREATED

    assert problem_persistence is not None
    assert occurrence_persistence is not None
    assert causal_evidence_persistence is not None

    return DiagnosticPersistenceComposition(
        problem_persistence=problem_persistence,
        occurrence_persistence=occurrence_persistence,
        causal_evidence_persistence=causal_evidence_persistence,
        runtime_event_persistence=runtime_event_persistence,
        problem_persistence_ownership=problem_ownership,
        occurrence_persistence_ownership=occurrence_ownership,
        causal_evidence_persistence_ownership=causal_ownership,
    )


def resolve_persistence_from_tool_wiring(
    *,
    tool_wiring_context: ToolWiringContext | None,
    runtime_event_persistence: RuntimeEventPersistence | None,
    overrides: DiagnosticCompositionOverrides | None = None,
    require_durable: bool = False,
) -> DiagnosticPersistenceComposition | None:
    """Resolve persistence from platform tool wiring + optional overrides."""
    document_store = None
    if tool_wiring_context is not None:
        document_store = tool_wiring_context.document_store
    return resolve_diagnostic_persistence_composition(
        document_store=document_store,
        runtime_event_persistence=runtime_event_persistence,
        overrides=overrides,
        require_durable=require_durable,
    )


def build_default_execution_reconstruction_reader(
    persistence: DiagnosticPersistenceComposition,
    *,
    execution_lineage_reader: ExecutionLineageReader | None = None,
) -> ExecutionReconstructionReader:
    """Canonical Evidence-owned default for ``ExecutionReconstructionReader``."""
    return ExecutionReconstructor(
        runtime_events=persistence.runtime_event_persistence,
        causal_evidence=persistence.causal_evidence_persistence,
        execution_lineage=execution_lineage_reader,
    )


def build_grouping_strategy_registry(
    overrides: DiagnosticCompositionOverrides | None = None,
) -> ProblemGroupingStrategyRegistry:
    """
    Build registry with canonical default + optional additional strategies.

    Duplicate strategy IDs fail deterministically via the registry.
    """
    registry = ProblemGroupingStrategyRegistry()
    registry.register(DeterministicProblemGroupingStrategy())
    if overrides is not None:
        for strategy in overrides.additional_grouping_strategies:
            registry.register(strategy)
    return registry


def resolve_diagnostic_composition(
    persistence: DiagnosticPersistenceComposition,
    *,
    overrides: DiagnosticCompositionOverrides | None = None,
    execution_lineage_reader: ExecutionLineageReader | None = None,
) -> ResolvedDiagnosticComposition:
    """Resolve full typed composition over one persistence bundle."""
    overrides = overrides or DiagnosticCompositionOverrides()
    if overrides.execution_reconstruction_reader is not None:
        reader = overrides.execution_reconstruction_reader
        reconstruction_ownership = DiagnosticComponentOwnership.BORROWED
    else:
        reader = build_default_execution_reconstruction_reader(
            persistence,
            execution_lineage_reader=execution_lineage_reader,
        )
        reconstruction_ownership = DiagnosticComponentOwnership.HOST_CREATED
    return ResolvedDiagnosticComposition(
        persistence=persistence,
        execution_reconstruction_reader=reader,
        reconstruction_ownership=reconstruction_ownership,
        grouping_registry=build_grouping_strategy_registry(overrides),
    )


def build_diagnostic_orchestrator_from_composition(
    composition: ResolvedDiagnosticComposition,
) -> DiagnosticOrchestrator:
    """
    Construct the single canonical ``DiagnosticOrchestrator``.

    ``LifecycleAnomalyAnalyzer`` and ``DiagnosticAssessmentBuilder`` are hard
    platform invariants — not host-replaceable strategies.
    """
    persistence = composition.persistence
    return DiagnosticOrchestrator(
        execution_reconstructor=composition.execution_reconstruction_reader,
        lifecycle_analyzer=LifecycleAnomalyAnalyzer(),
        assessment_builder=DiagnosticAssessmentBuilder(),
        grouping_engine=ProblemGroupingEngine(
            composition.grouping_registry,
            feature_projector=DiagnosticProblemGroupingFeatureProjector(),
        ),
        problem_lifecycle_engine=ProblemLifecycleEngine(
            persistence.problem_persistence,
            persistence.occurrence_persistence,
        ),
    )


def _close_if_host_owned(
    resource: object,
    ownership: DiagnosticComponentOwnership,
) -> None:
    if ownership is not DiagnosticComponentOwnership.HOST_CREATED:
        return
    if isinstance(resource, SupportsClose):
        resource.close()


def close_host_owned_diagnostic_persistence(
    persistence: DiagnosticPersistenceComposition,
) -> None:
    """Close host-created persistence resources once; never close borrowed ones."""
    _close_if_host_owned(
        persistence.problem_persistence,
        persistence.problem_persistence_ownership,
    )
    _close_if_host_owned(
        persistence.occurrence_persistence,
        persistence.occurrence_persistence_ownership,
    )
    _close_if_host_owned(
        persistence.causal_evidence_persistence,
        persistence.causal_evidence_persistence_ownership,
    )


__all__ = [
    "DiagnosticComponentOwnership",
    "DiagnosticCompositionError",
    "DiagnosticCompositionOverrides",
    "DiagnosticPersistenceComposition",
    "DiagnosticProviderResolutionError",
    "ResolvedDiagnosticComposition",
    "SupportsClose",
    "build_default_execution_reconstruction_reader",
    "build_diagnostic_orchestrator_from_composition",
    "build_grouping_strategy_registry",
    "close_host_owned_diagnostic_persistence",
    "resolve_diagnostic_composition",
    "resolve_diagnostic_persistence_composition",
    "resolve_persistence_from_tool_wiring",
]
