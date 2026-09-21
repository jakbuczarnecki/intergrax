# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Worker capability recovery outcomes after canonical discovery/UCA (UCA-6B)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from intergrax.contracts.autonomous_work._validation import (
    freeze_tuple,
    require_aware_utc,
    require_non_empty_text,
)
from intergrax.contracts.autonomous_work.references import (
    ProblemReference,
    validate_problem_reference,
)
from intergrax.contracts.capability_acquisition.acquisition_result import (
    CapabilityAcquisitionResult,
)
from intergrax.contracts.capability_catalog.discovery_completion import (
    DiscoveryCompletion,
)
from intergrax.contracts.capability_qualification.qualification_result import (
    CapabilityQualificationResult,
)


class WorkerCapabilityRecoveryPhase(StrEnum):
    """High-level worker recovery phase after canonical coordination."""

    DIRECT_REUSE = "DIRECT_REUSE"
    REALIZATION_REQUIRED = "REALIZATION_REQUIRED"
    PENDING_QUALIFICATION = "PENDING_QUALIFICATION"
    QUALIFICATION_COMPLETE = "QUALIFICATION_COMPLETE"
    PAUSE_HITL = "PAUSE_HITL"
    FAIL_CLOSED = "FAIL_CLOSED"


@dataclass(frozen=True, slots=True)
class WorkerCapabilityRecoveryProvenance:
    """Immutable provenance chain for worker recovery — preserves exact canonical IDs."""

    worker_need_id: str
    canonical_need_id: str
    discovery_correlation_id: str
    discovery_completion_outcome: str
    gap_id: str | None = None
    acquisition_request_id: str | None = None
    acquisition_strategy_id: str | None = None
    qualification_request_id: str | None = None
    qualified_subject_reference: str | None = None
    binding_operation_id: str | None = None
    execution_request_id: str | None = None
    evidence_refs: tuple[ProblemReference, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "worker_need_id",
            require_non_empty_text(self.worker_need_id, label="worker_need_id"),
        )
        object.__setattr__(
            self,
            "canonical_need_id",
            require_non_empty_text(self.canonical_need_id, label="canonical_need_id"),
        )
        object.__setattr__(
            self,
            "discovery_correlation_id",
            require_non_empty_text(
                self.discovery_correlation_id,
                label="discovery_correlation_id",
            ),
        )
        object.__setattr__(
            self,
            "discovery_completion_outcome",
            require_non_empty_text(
                self.discovery_completion_outcome,
                label="discovery_completion_outcome",
            ),
        )
        for label, value in (
            ("gap_id", self.gap_id),
            ("acquisition_request_id", self.acquisition_request_id),
            ("acquisition_strategy_id", self.acquisition_strategy_id),
            ("qualification_request_id", self.qualification_request_id),
            ("qualified_subject_reference", self.qualified_subject_reference),
            ("binding_operation_id", self.binding_operation_id),
            ("execution_request_id", self.execution_request_id),
        ):
            if value is not None:
                object.__setattr__(
                    self,
                    label,
                    require_non_empty_text(value, label=label),
                )
        object.__setattr__(
            self,
            "evidence_refs",
            freeze_tuple(self.evidence_refs, label="evidence_refs"),
        )
        for ref in self.evidence_refs:
            validate_problem_reference(ref)


@dataclass(frozen=True, slots=True)
class WorkerCapabilityRecoveryOutcome:
    """Worker-specific recovery outcome — strategy-opaque canonical consumption."""

    phase: WorkerCapabilityRecoveryPhase
    provenance: WorkerCapabilityRecoveryProvenance
    discovery_completion: DiscoveryCompletion | None = None
    acquisition_result: CapabilityAcquisitionResult | None = None
    qualification_result: CapabilityQualificationResult | None = None
    decided_at: datetime | None = None

    def __post_init__(self) -> None:
        if type(self.phase) is not WorkerCapabilityRecoveryPhase:
            raise TypeError("phase must be WorkerCapabilityRecoveryPhase")
        if type(self.provenance) is not WorkerCapabilityRecoveryProvenance:
            raise TypeError("provenance must be WorkerCapabilityRecoveryProvenance")
        if self.decided_at is not None:
            object.__setattr__(
                self,
                "decided_at",
                require_aware_utc(self.decided_at, label="decided_at"),
            )


__all__ = [
    "WorkerCapabilityRecoveryOutcome",
    "WorkerCapabilityRecoveryPhase",
    "WorkerCapabilityRecoveryProvenance",
]
