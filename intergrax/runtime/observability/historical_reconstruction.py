# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical read-only historical reconstruction composition (NPSC-5F/R4)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from intergrax.contracts.bitemporal_knowledge import (
    BitemporalKnowledgeBasis,
    KnowledgeRevisionId,
    RevisionOrderingAuthority,
)
from intergrax.contracts.execution_lineage import ExecutionLineageReader
from intergrax.contracts.execution_identity import validate_task_id
from intergrax.contracts.historical_reconstruction import (
    ExecutionHistoricalReconstructionRequest,
    HistoricalEvidenceIntegrityError,
    HistoricalReconstructionBasis,
    KnowledgeBoundaryNotFinalizedError,
    revision_admissible_at_bitemporal_query,
)
from intergrax.runtime.diagnostics.execution_reconstruction import (
    ExecutionAttemptDiscoveryCompleteness,
    ExecutionAttemptDiscoveryReadStatus,
    ExecutionReconstruction,
    ExecutionReconstructionIntegrityError,
    ExecutionReconstructor,
    RuntimeHistoryCompleteness,
)
from intergrax.runtime.events.asof_projection import (
    InvalidRunExecutionHistoryError,
    RunExecutionBoundaryNotFoundError,
    RunExecutionHistoryNotFoundError,
    RunExecutionHistoryTruncatedError,
    RunExecutionAsOfProjection,
    reconstruct_run_execution_as_of,
)
from intergrax.runtime.events.persistence_contract import RuntimeEventPersistence
from intergrax.runtime.observability.causal_evidence_persistence import CausalEvidencePersistence
from intergrax.runtime.observability.knowledge_reconstruction import (
    HistoricalKnowledgeProjection,
    HistoricalKnowledgeRevisionReference,
    KnowledgeReconstructionError,
    KnowledgeRevisionReader,
    reconstruct_knowledge_at_watermark,
)

RevisionT_co = TypeVar("RevisionT_co", covariant=True)
StateT = TypeVar("StateT")


class BitemporalKnowledgeRevision(Protocol[RevisionT_co]):
    """Revision payloads that carry canonical bitemporal axes for filtering."""

    @property
    def revision_id(self) -> KnowledgeRevisionId: ...

    @property
    def bitemporal_basis(self) -> BitemporalKnowledgeBasis: ...


@dataclass(frozen=True, slots=True)
class ExecutionHistoricalReconstructionLimitations:
    runtime_history_completeness: RuntimeHistoryCompleteness
    attempt_discovery_read_status: ExecutionAttemptDiscoveryReadStatus | None
    attempt_discovery_completeness: ExecutionAttemptDiscoveryCompleteness | None
    lineage_partial: bool
    knowledge_bitemporal_filtered: bool


@dataclass(frozen=True, slots=True)
class ExecutionHistoricalReconstruction(Generic[StateT]):
    """Immutable historical view at explicit E/K/VT — read-only, never authoritative for execution."""

    basis: HistoricalReconstructionBasis
    execution_projection: RunExecutionAsOfProjection
    execution_reconstruction: ExecutionReconstruction
    knowledge_view: HistoricalKnowledgeProjection[StateT]
    limitations: ExecutionHistoricalReconstructionLimitations


class HistoricalReconstructionService:
    """Compose execution as-of, execution reconstruction, and bitemporal knowledge at K."""

    def __init__(
        self,
        runtime_events: RuntimeEventPersistence,
        causal_evidence: CausalEvidencePersistence,
        revision_ordering: RevisionOrderingAuthority,
        *,
        execution_lineage: ExecutionLineageReader | None = None,
        execution_reconstructor: ExecutionReconstructor | None = None,
    ) -> None:
        self._runtime_events = runtime_events
        self._causal_evidence = causal_evidence
        self._revision_ordering = revision_ordering
        self._execution_reconstructor = execution_reconstructor or ExecutionReconstructor(
            runtime_events,
            causal_evidence,
            execution_lineage,
        )

    def reconstruct(
        self,
        request: ExecutionHistoricalReconstructionRequest,
        *,
        revision_reader: KnowledgeRevisionReader[RevisionT_co],
        revision_bitemporal_basis: Callable[[RevisionT_co], BitemporalKnowledgeBasis],
        reducer: Callable[[StateT, RevisionT_co], StateT],
        initial_state: StateT,
        initial_limit: int = 1000,
        max_limit: int = 1_000_000,
    ) -> ExecutionHistoricalReconstruction[StateT]:
        basis = request.to_basis()
        self._assert_knowledge_watermark_finalized(request)

        try:
            execution_projection = reconstruct_run_execution_as_of(
                persistence=self._runtime_events,
                tenant_id=request.tenant_id,
                boundary=request.execution_as_of,
                initial_limit=initial_limit,
                max_limit=max_limit,
            )
        except (
            RunExecutionHistoryTruncatedError,
            InvalidRunExecutionHistoryError,
        ) as exc:
            raise HistoricalEvidenceIntegrityError(str(exc)) from exc
        except RunExecutionBoundaryNotFoundError as exc:
            raise HistoricalEvidenceIntegrityError(str(exc)) from exc
        except RunExecutionHistoryNotFoundError as exc:
            raise HistoricalEvidenceIntegrityError(str(exc)) from exc

        task_id = validate_task_id(execution_projection.task_id)
        try:
            execution_reconstruction = self._execution_reconstructor.reconstruct_execution(
                request.tenant_id,
                task_id,
                request.run_id,
                execution_as_of=request.execution_as_of,
                initial_limit=initial_limit,
                max_limit=max_limit,
            )
        except ExecutionReconstructionIntegrityError as exc:
            raise HistoricalEvidenceIntegrityError(str(exc)) from exc

        knowledge_view, bitemporal_filtered = _reconstruct_bitemporal_knowledge(
            self._revision_ordering,
            request,
            revision_reader=revision_reader,
            revision_bitemporal_basis=revision_bitemporal_basis,
            reducer=reducer,
            initial_state=initial_state,
        )

        limitations = ExecutionHistoricalReconstructionLimitations(
            runtime_history_completeness=execution_reconstruction.runtime_history_completeness,
            attempt_discovery_read_status=execution_reconstruction.attempt_discovery_read_status,
            attempt_discovery_completeness=execution_reconstruction.attempt_discovery_completeness,
            lineage_partial=execution_reconstruction.has_partial_lineage,
            knowledge_bitemporal_filtered=bitemporal_filtered,
        )
        return ExecutionHistoricalReconstruction(
            basis=basis,
            execution_projection=execution_projection,
            execution_reconstruction=execution_reconstruction,
            knowledge_view=knowledge_view,
            limitations=limitations,
        )

    def _assert_knowledge_watermark_finalized(
        self,
        request: ExecutionHistoricalReconstructionRequest,
    ) -> None:
        finalized = self._revision_ordering.watermark(request.knowledge_watermark.scope)
        if (
            request.knowledge_watermark.finalized_through_value
            > finalized.finalized_through_value
        ):
            raise KnowledgeBoundaryNotFinalizedError(
                "requested knowledge watermark exceeds durable finalized prefix"
            )


def _reconstruct_bitemporal_knowledge(
    authority: RevisionOrderingAuthority,
    request: ExecutionHistoricalReconstructionRequest,
    *,
    revision_reader: KnowledgeRevisionReader[RevisionT_co],
    revision_bitemporal_basis: Callable[[RevisionT_co], BitemporalKnowledgeBasis],
    reducer: Callable[[StateT, RevisionT_co], StateT],
    initial_state: StateT,
) -> tuple[HistoricalKnowledgeProjection[StateT], bool]:
    query = request.bitemporal_query
    try:
        prefix_projection = reconstruct_knowledge_at_watermark(
            authority,
            request.knowledge_watermark,
            revision_reader=revision_reader,
            reducer=lambda state, _revision: state,
            initial_state=initial_state,
        )
    except KnowledgeReconstructionError as exc:
        raise HistoricalEvidenceIntegrityError(str(exc)) from exc

    accepted: list[HistoricalKnowledgeRevisionReference] = []
    state: StateT = initial_state
    filtered_any = False
    for reference in prefix_projection.accepted_revisions:
        revision = revision_reader.load_revision(reference.revision_id)
        revision_basis = revision_bitemporal_basis(revision)
        if not revision_admissible_at_bitemporal_query(query=query, revision=revision_basis):
            filtered_any = True
            continue
        accepted.append(reference)
        state = reducer(state, revision)

    projection = HistoricalKnowledgeProjection(
        watermark=prefix_projection.watermark,
        state=state,
        accepted_revisions=tuple(accepted),
    )
    return projection, filtered_any
