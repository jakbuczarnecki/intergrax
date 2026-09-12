# © Artur Czarnecki. All rights reserved.

"""Observability boundary for Execution Runtime persistence reliability decisions."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.contracts.execution_evidence.persistence_boundary_errors import (
    EvidencePersistenceBoundaryError,
    MandatoryEvidencePersistenceError,
)
from intergrax.contracts.execution_evidence.persistence_failure_contract import (
    EvidencePersistenceFailureCategory,
)
from intergrax.contracts.execution_evidence.persistence_reliability_diagnostics_contract import (
    NullPersistenceReliabilityDecisionObserver,
    PersistenceReliabilityDiagnostic,
)
from intergrax.contracts.execution_evidence.persistence_reliability_policy_contract import (
    PersistenceReliabilityDisposition,
    PersistenceReliabilityPolicyDecision,
    PersistenceReliabilityPolicyRequest,
    RuntimeEvidenceDurabilityRequirement,
)
from intergrax.runtime.events.default_persistence_reliability_policy import (
    DEFAULT_PERSISTENCE_RELIABILITY_POLICY_ID,
)
from intergrax.runtime.events.evidence_durability import EvidencePersistenceRequirement
from intergrax.runtime.events.runtime_persistence_resilience import (
    classify_controlled_persistence_failure,
    resolve_runtime_persistence_failure,
)
from intergrax.runtime.observability.persistence_conformance import sample_runtime_event

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_RECONSTRUCTION_PATH = _REPO_ROOT / "intergrax/runtime/diagnostics/execution_reconstruction.py"
_UNIFIED_RUN_JOURNAL_PATH = _REPO_ROOT / "intergrax/runtime/events/unified_run_journal.py"


class _CapturingObserver:
    def __init__(self) -> None:
        self.diagnostics: list[PersistenceReliabilityDiagnostic] = []

    def observe_persistence_reliability_decision(
        self,
        diagnostic: PersistenceReliabilityDiagnostic,
    ) -> None:
        self.diagnostics.append(diagnostic)


class _AlternateObserver:
    def __init__(self) -> None:
        self.seen = False

    def observe_persistence_reliability_decision(
        self,
        diagnostic: PersistenceReliabilityDiagnostic,
    ) -> None:
        self.seen = True


class _AlwaysContinuePolicy:
    def decide(
        self,
        request: PersistenceReliabilityPolicyRequest,
    ) -> PersistenceReliabilityPolicyDecision:
        return PersistenceReliabilityPolicyDecision(
            disposition=PersistenceReliabilityDisposition.ALLOW_CONTINUE,
        )


def test_reliability_decision_emits_structured_diagnostic() -> None:
    observer = _CapturingObserver()
    event = sample_runtime_event(tenant_id="tenant-diag")
    with pytest.raises(MandatoryEvidencePersistenceError):
        resolve_runtime_persistence_failure(
            requirement=EvidencePersistenceRequirement.MANDATORY,
            failure=EvidencePersistenceBoundaryError("backend down"),
            event_type=event.event_type,
            observer=observer,
        )
    assert len(observer.diagnostics) == 1
    diagnostic = observer.diagnostics[0]
    assert diagnostic.failure_category is EvidencePersistenceFailureCategory.INFRASTRUCTURE
    assert diagnostic.durability is RuntimeEvidenceDurabilityRequirement.MANDATORY
    assert diagnostic.disposition is PersistenceReliabilityDisposition.FAIL_CLOSED
    assert diagnostic.policy_id == DEFAULT_PERSISTENCE_RELIABILITY_POLICY_ID
    assert diagnostic.runtime_event_type == event.event_type.value


def test_observer_boundary_is_swappable() -> None:
    primary = _CapturingObserver()
    alternate = _AlternateObserver()
    event = sample_runtime_event(tenant_id="tenant-observer-swap")
    failure = EvidencePersistenceBoundaryError("backend down")
    resolve_runtime_persistence_failure(
        requirement=EvidencePersistenceRequirement.BEST_EFFORT,
        failure=failure,
        event_type=event.event_type,
        observer=primary,
    )
    assert len(primary.diagnostics) == 1
    assert alternate.seen is False

    resolve_runtime_persistence_failure(
        requirement=EvidencePersistenceRequirement.BEST_EFFORT,
        failure=failure,
        event_type=event.event_type,
        observer=alternate,
    )
    assert alternate.seen is True


def test_null_observer_preserves_resilience_behavior() -> None:
    event = sample_runtime_event(tenant_id="tenant-null-observer")
    with pytest.raises(MandatoryEvidencePersistenceError):
        resolve_runtime_persistence_failure(
            requirement=EvidencePersistenceRequirement.MANDATORY,
            failure=EvidencePersistenceBoundaryError("backend down"),
            event_type=event.event_type,
            observer=NullPersistenceReliabilityDecisionObserver(),
        )

    resolve_runtime_persistence_failure(
        requirement=EvidencePersistenceRequirement.BEST_EFFORT,
        failure=EvidencePersistenceBoundaryError("backend down"),
        event_type=event.event_type,
        observer=NullPersistenceReliabilityDecisionObserver(),
    )


def test_diagnostic_observer_does_not_change_execution_outcome() -> None:
    event = sample_runtime_event(tenant_id="tenant-outcome-parity")
    failure = EvidencePersistenceBoundaryError("backend down")

    with pytest.raises(MandatoryEvidencePersistenceError):
        resolve_runtime_persistence_failure(
            requirement=EvidencePersistenceRequirement.MANDATORY,
            failure=failure,
            event_type=event.event_type,
        )

    with pytest.raises(MandatoryEvidencePersistenceError):
        resolve_runtime_persistence_failure(
            requirement=EvidencePersistenceRequirement.MANDATORY,
            failure=failure,
            event_type=event.event_type,
            observer=_CapturingObserver(),
        )

    resolve_runtime_persistence_failure(
        requirement=EvidencePersistenceRequirement.BEST_EFFORT,
        failure=failure,
        event_type=event.event_type,
    )
    resolve_runtime_persistence_failure(
        requirement=EvidencePersistenceRequirement.BEST_EFFORT,
        failure=failure,
        event_type=event.event_type,
        observer=_CapturingObserver(),
    )


def test_classify_without_observer_matches_controlled_failure_semantics() -> None:
    without_observer = classify_controlled_persistence_failure(
        EvidencePersistenceBoundaryError("storage unavailable"),
        requirement=EvidencePersistenceRequirement.MANDATORY,
    )
    with_observer = classify_controlled_persistence_failure(
        EvidencePersistenceBoundaryError("storage unavailable"),
        requirement=EvidencePersistenceRequirement.MANDATORY,
        observer=_CapturingObserver(),
        policy=_AlwaysContinuePolicy(),
    )
    assert without_observer.runtime_may_continue is False
    assert with_observer.runtime_may_continue is True


def test_evidence_plane_modules_untouched_by_reliability_diagnostics() -> None:
    for path in (_RECONSTRUCTION_PATH, _UNIFIED_RUN_JOURNAL_PATH):
        source = path.read_text(encoding="utf-8")
        assert "persistence_reliability_diagnostics" not in source
        assert "PersistenceReliabilityDiagnostic" not in source
