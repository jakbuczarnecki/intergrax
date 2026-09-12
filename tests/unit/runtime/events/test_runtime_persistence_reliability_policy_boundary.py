# © Artur Czarnecki. All rights reserved.

"""Reliability policy boundary at the Execution Runtime persistence port."""

from __future__ import annotations

import pytest

from intergrax.contracts.execution_evidence.persistence_boundary_errors import (
    EvidencePersistenceBoundaryError,
    MandatoryEvidencePersistenceError,
)
from intergrax.contracts.execution_evidence.persistence_reliability_policy_contract import (
    KnownPersistencePortFailure,
    PersistenceReliabilityDisposition,
    PersistenceReliabilityPolicyDecision,
    PersistenceReliabilityPolicyRequest,
    RuntimeEvidenceDurabilityRequirement,
)
from intergrax.runtime.events.evidence_durability import EvidencePersistenceRequirement
from intergrax.runtime.events.runtime_persistence_resilience import (
    resolve_runtime_persistence_failure,
)
from intergrax.runtime.observability.persistence_conformance import sample_runtime_event

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _AlwaysFailClosedPolicy:
    def decide(
        self,
        request: PersistenceReliabilityPolicyRequest,
    ) -> PersistenceReliabilityPolicyDecision:
        return PersistenceReliabilityPolicyDecision(
            disposition=PersistenceReliabilityDisposition.FAIL_CLOSED,
        )


class _AlwaysContinuePolicy:
    def decide(
        self,
        request: PersistenceReliabilityPolicyRequest,
    ) -> PersistenceReliabilityPolicyDecision:
        return PersistenceReliabilityPolicyDecision(
            disposition=PersistenceReliabilityDisposition.ALLOW_CONTINUE,
        )


def test_resolve_delegates_to_injected_policy() -> None:
    event = sample_runtime_event(tenant_id="tenant-policy-inject")
    with pytest.raises(MandatoryEvidencePersistenceError):
        resolve_runtime_persistence_failure(
            requirement=EvidencePersistenceRequirement.BEST_EFFORT,
            failure=EvidencePersistenceBoundaryError("backend down"),
            event_type=event.event_type,
            policy=_AlwaysFailClosedPolicy(),
        )


def test_swap_policy_changes_reaction_without_runtime_changes() -> None:
    event = sample_runtime_event(tenant_id="tenant-policy-swap")
    resolve_runtime_persistence_failure(
        requirement=EvidencePersistenceRequirement.MANDATORY,
        failure=EvidencePersistenceBoundaryError("backend down"),
        event_type=event.event_type,
        policy=_AlwaysContinuePolicy(),
    )


def test_policy_request_carries_normalized_problem_only() -> None:
    captured: list[PersistenceReliabilityPolicyRequest] = []

    class _CapturingPolicy:
        def decide(
            self,
            request: PersistenceReliabilityPolicyRequest,
        ) -> PersistenceReliabilityPolicyDecision:
            captured.append(request)
            return PersistenceReliabilityPolicyDecision(
                disposition=PersistenceReliabilityDisposition.ALLOW_CONTINUE,
            )

    event = sample_runtime_event(tenant_id="tenant-capture")
    resolve_runtime_persistence_failure(
        requirement=EvidencePersistenceRequirement.MANDATORY,
        failure=EvidencePersistenceBoundaryError("opaque provider fault"),
        event_type=event.event_type,
        policy=_CapturingPolicy(),
    )
    assert len(captured) == 1
    assert captured[0].durability is RuntimeEvidenceDurabilityRequirement.MANDATORY
    assert isinstance(captured[0].failure, KnownPersistencePortFailure)
