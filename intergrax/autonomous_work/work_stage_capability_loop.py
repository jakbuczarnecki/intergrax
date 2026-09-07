# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Bounded reference work-stage capability discovery loop (CAPABILITY-CATALOG-1 Stage 14).

Thin composition over Stage-8 discovery and domain execution authorities.
Does not own worker lifecycle, scheduling, or registry mutation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from intergrax.capability_catalog.errors import (
    CapabilityCatalogDiscoveryError,
    CapabilityCatalogIdentityConflict,
    CapabilityCatalogSourceFailure,
)
from intergrax.capability_catalog.federation import FederatedCapabilityCatalog
from intergrax.capability_catalog.governed_candidate import GovernedCapabilityCandidate
from intergrax.capability_catalog.work_stage_discovery import (
    WorkStageCapabilityDiscoveryService,
)
from intergrax.capability_catalog.work_stage_effective import (
    SCHEMA_WORK_STAGE_CAPABILITY_DISCOVERY_EVIDENCE_V1,
    WorkStageCapabilityDiscoveryEvidence,
)
from intergrax.contracts.capability_catalog.evidence import (
    CapabilityDiscoveryAvailabilityEvidence,
)
from intergrax.contracts.capability_catalog.governance import CapabilityGovernanceContext
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.capability_catalog.kind import CapabilityKind
from intergrax.contracts.capability_catalog.work_stage import WorkStageCapabilityNeed
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    reset_active_execution_identity,
    validate_run_id,
)
from intergrax.contracts.capability_catalog.work_stage_loop import (
    WorkStageCapabilityExecutionEvidenceRef,
    WorkStageCapabilityLoopDisposition,
    WorkStageCapabilityLoopIterationEvidence,
    WorkStageCapabilityLoopResult,
    WorkStageCapabilityObservation,
    WorkStageDomainAuthorityKind,
    derive_work_stage_capability_execution_evidence_ref,
)


@dataclass(frozen=True, slots=True)
class WorkStageDiscoveryIterationContext:
    """Per-iteration availability and governance inputs — snapshot is caller-owned."""

    availability_evidence: CapabilityDiscoveryAvailabilityEvidence
    governance_context: CapabilityGovernanceContext


@runtime_checkable
class WorkStageDiscoveryContextProvider(Protocol):
    """Supply iteration-scoped discovery context without hiding catalog snapshot reads."""

    def iteration_context(
        self,
        need: WorkStageCapabilityNeed,
        iteration_index: int,
    ) -> WorkStageDiscoveryIterationContext:
        """Return current availability and governance evidence for one loop iteration."""


@dataclass(frozen=True, slots=True)
class WorkStageToolExecutionRequest:
    """Governed Tool selection routed to canonical Tool execution authority."""

    candidate: GovernedCapabilityCandidate
    run_id: str
    step_id: str


@dataclass(frozen=True, slots=True)
class WorkStageToolExecutionResult:
    """Typed Tool execution outcome for loop observation."""

    tool_id: str
    success: bool
    output_summary: str


@runtime_checkable
class WorkStageToolExecutionPort(Protocol):
    """Domain Tool execution port — Capability Catalog never implements this."""

    def execute(self, request: WorkStageToolExecutionRequest) -> WorkStageToolExecutionResult:
        """Execute the selected governed Tool candidate."""


@runtime_checkable
class WorkStageObservationProvider(Protocol):
    """Deterministic observation boundary — maps execution to typed next need."""

    def observe(
        self,
        *,
        iteration_index: int,
        need: WorkStageCapabilityNeed,
        selected: GovernedCapabilityCandidate | None,
        execution: WorkStageToolExecutionResult | None,
        discovery: WorkStageCapabilityDiscoveryEvidence,
    ) -> WorkStageCapabilityObservation:
        """Produce typed observation; may include next WorkStageCapabilityNeed."""


def _domain_authority_kind(candidate: GovernedCapabilityCandidate) -> WorkStageDomainAuthorityKind:
    kind = candidate.identity.kind
    if kind is CapabilityKind.TOOL:
        return WorkStageDomainAuthorityKind.TOOL
    if kind is CapabilityKind.AGENT:
        return WorkStageDomainAuthorityKind.AGENT
    return WorkStageDomainAuthorityKind.SKILL


def _iteration_evidence(
    *,
    iteration_index: int,
    need: WorkStageCapabilityNeed,
    discovery: WorkStageCapabilityDiscoveryEvidence,
    selected: GovernedCapabilityCandidate | None,
    execution_ref: str | None,
    observation: WorkStageCapabilityObservation | None,
) -> WorkStageCapabilityLoopIterationEvidence:
    selected_key: CapabilityIdentityKey | None = None
    authority: WorkStageDomainAuthorityKind | None = None
    evidence_ref: WorkStageCapabilityExecutionEvidenceRef | None = None
    if selected is not None:
        selected_key = CapabilityIdentityKey.from_discovery_identity(selected.identity)
        authority = _domain_authority_kind(selected)
        if execution_ref is not None:
            evidence_ref = WorkStageCapabilityExecutionEvidenceRef(reference=execution_ref)
    return WorkStageCapabilityLoopIterationEvidence(
        iteration_index=iteration_index,
        need=need,
        discovery_evidence_schema=SCHEMA_WORK_STAGE_CAPABILITY_DISCOVERY_EVIDENCE_V1,
        selected_identity_key=selected_key,
        domain_authority_kind=authority,
        execution_evidence_ref=evidence_ref,
        observation=observation,
    )


class WorkStageCapabilityDiscoveryLoopCoordinator:
    """Bounded reference loop: typed need → fresh discovery → domain authority → observe."""

    def __init__(
        self,
        *,
        discovery_service: WorkStageCapabilityDiscoveryService,
        federated_catalog: FederatedCapabilityCatalog,
        context_provider: WorkStageDiscoveryContextProvider,
        tool_execution: WorkStageToolExecutionPort,
        observation_provider: WorkStageObservationProvider,
        run_id: str,
        max_iterations: int = 10,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        self._discovery_service = discovery_service
        self._federated_catalog = federated_catalog
        self._context_provider = context_provider
        self._tool_execution = tool_execution
        self._observation_provider = observation_provider
        self._run_id = run_id
        self._max_iterations = max_iterations

    def run(self, initial_need: WorkStageCapabilityNeed) -> WorkStageCapabilityLoopRunOutcome:
        canonical_run_id = validate_run_id(self._run_id)
        identity_token = bind_active_execution_identity(
            run_id=canonical_run_id,
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
        )
        try:
            return self._run_bounded(initial_need)
        finally:
            reset_active_execution_identity(identity_token)

    def _run_bounded(self, initial_need: WorkStageCapabilityNeed) -> WorkStageCapabilityLoopRunOutcome:
        iterations: list[WorkStageCapabilityLoopIterationEvidence] = []
        discovery_records: list[WorkStageCapabilityDiscoveryEvidence] = []
        current_need = initial_need

        for iteration_index in range(self._max_iterations):
            try:
                snapshot = self._federated_catalog.snapshot()
            except CapabilityCatalogSourceFailure:
                return WorkStageCapabilityLoopRunOutcome(
                    result=WorkStageCapabilityLoopResult(
                        disposition=WorkStageCapabilityLoopDisposition.UNAVAILABLE,
                        iterations=tuple(iterations),
                    ),
                    discovery_records=tuple(discovery_records),
                )
            except CapabilityCatalogIdentityConflict:
                return WorkStageCapabilityLoopRunOutcome(
                    result=WorkStageCapabilityLoopResult(
                        disposition=WorkStageCapabilityLoopDisposition.CONFLICT,
                        iterations=tuple(iterations),
                    ),
                    discovery_records=tuple(discovery_records),
                )

            iteration_context = self._context_provider.iteration_context(
                current_need,
                iteration_index,
            )
            try:
                discovery = self._discovery_service.resolve(
                    current_need,
                    snapshot=snapshot,
                    availability_evidence=iteration_context.availability_evidence,
                    governance_context=iteration_context.governance_context,
                )
            except CapabilityCatalogDiscoveryError:
                return WorkStageCapabilityLoopRunOutcome(
                    result=WorkStageCapabilityLoopResult(
                        disposition=WorkStageCapabilityLoopDisposition.BLOCKED,
                        iterations=tuple(iterations),
                    ),
                    discovery_records=tuple(discovery_records),
                )

            discovery_records.append(discovery)
            effective = discovery.effective_set.effective_candidates
            if not effective:
                blocked_iteration = _iteration_evidence(
                    iteration_index=iteration_index,
                    need=current_need,
                    discovery=discovery,
                    selected=None,
                    execution_ref=None,
                    observation=self._observation_provider.observe(
                        iteration_index=iteration_index,
                        need=current_need,
                        selected=None,
                        execution=None,
                        discovery=discovery,
                    ),
                )
                iterations.append(blocked_iteration)
                return WorkStageCapabilityLoopRunOutcome(
                    result=WorkStageCapabilityLoopResult(
                        disposition=WorkStageCapabilityLoopDisposition.BLOCKED,
                        iterations=tuple(iterations),
                    ),
                    discovery_records=tuple(discovery_records),
                )

            selected = effective[0]
            if selected.identity.kind is not CapabilityKind.TOOL:
                blocked_iteration = _iteration_evidence(
                    iteration_index=iteration_index,
                    need=current_need,
                    discovery=discovery,
                    selected=selected,
                    execution_ref=None,
                    observation=self._observation_provider.observe(
                        iteration_index=iteration_index,
                        need=current_need,
                        selected=selected,
                        execution=None,
                        discovery=discovery,
                    ),
                )
                iterations.append(blocked_iteration)
                return WorkStageCapabilityLoopRunOutcome(
                    result=WorkStageCapabilityLoopResult(
                        disposition=WorkStageCapabilityLoopDisposition.BLOCKED,
                        iterations=tuple(iterations),
                    ),
                    discovery_records=tuple(discovery_records),
                )

            execution = self._tool_execution.execute(
                WorkStageToolExecutionRequest(
                    candidate=selected,
                    run_id=self._run_id,
                    step_id=str(iteration_index),
                ),
            )
            evidence_ref = derive_work_stage_capability_execution_evidence_ref(
                work_reference=current_need.work_reference,
                stage_reference=current_need.stage_reference,
                iteration_index=iteration_index,
                logical_id=selected.identity.logical.logical_id,
            ).reference
            observation = self._observation_provider.observe(
                iteration_index=iteration_index,
                need=current_need,
                selected=selected,
                execution=execution,
                discovery=discovery,
            )
            iteration = _iteration_evidence(
                iteration_index=iteration_index,
                need=current_need,
                discovery=discovery,
                selected=selected,
                execution_ref=evidence_ref,
                observation=observation,
            )
            iterations.append(iteration)

            if observation.next_need is None:
                return WorkStageCapabilityLoopRunOutcome(
                    result=WorkStageCapabilityLoopResult(
                        disposition=WorkStageCapabilityLoopDisposition.COMPLETED,
                        iterations=tuple(iterations),
                    ),
                    discovery_records=tuple(discovery_records),
                )
            if (
                observation.next_need.work_reference == current_need.work_reference
                and observation.next_need.stage_reference == current_need.stage_reference
                and observation.next_need.discovery_query == current_need.discovery_query
                and not observation.execution_succeeded
            ):
                return WorkStageCapabilityLoopRunOutcome(
                    result=WorkStageCapabilityLoopResult(
                        disposition=WorkStageCapabilityLoopDisposition.ESCALATED,
                        iterations=tuple(iterations),
                    ),
                    discovery_records=tuple(discovery_records),
                )
            current_need = observation.next_need

        return WorkStageCapabilityLoopRunOutcome(
            result=WorkStageCapabilityLoopResult(
                disposition=WorkStageCapabilityLoopDisposition.ESCALATED,
                iterations=tuple(iterations),
            ),
            discovery_records=tuple(discovery_records),
        )


@dataclass(frozen=True, slots=True)
class WorkStageCapabilityLoopRunOutcome:
    """Coordinator output with immutable contract result and Stage-8 discovery records."""

    result: WorkStageCapabilityLoopResult
    discovery_records: tuple[WorkStageCapabilityDiscoveryEvidence, ...]
