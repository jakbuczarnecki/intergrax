# © Artur Czarnecki. All rights reserved.

"""EE-B1.1 — persistence failure semantics (mandatory evidence, checkpoint integrity)."""

from __future__ import annotations

import pytest

from intergrax.contracts.execution_evidence.persistence_failure_contract import (
    EvidencePersistenceFailureCategory,
)
from intergrax.contracts.execution_evidence.persistence_reliability_policy_contract import (
    PersistenceReliabilityDisposition,
    RuntimeEvidenceDurabilityRequirement,
)
from intergrax.contracts.execution_reliability import (
    PersistenceFailurePolicy,
    resolve_persistence_failure_policy,
)
from intergrax.runtime.events.runtime_persistence_resilience import (
    resolve_runtime_persistence_failure,
)
from intergrax.contracts.execution_evidence.persistence_boundary_errors import (
    EvidencePersistenceBoundaryError,
    MandatoryEvidencePersistenceError,
)
from intergrax.runtime.events.evidence_durability import EvidencePersistenceRequirement
from intergrax.runtime.events.runtime_event import RuntimeEventType

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_ee_b1_1_mandatory_append_fail_closed() -> None:
    disposition = resolve_persistence_failure_policy(
        policy=PersistenceFailurePolicy.DEGRADE,
        durability=RuntimeEvidenceDurabilityRequirement.MANDATORY,
        category=EvidencePersistenceFailureCategory.INFRASTRUCTURE,
    )
    assert disposition is PersistenceReliabilityDisposition.FAIL_CLOSED


def test_ee_b1_1_checkpoint_integrity_never_allows_continue() -> None:
    disposition = resolve_persistence_failure_policy(
        policy=PersistenceFailurePolicy.DEGRADE,
        durability=RuntimeEvidenceDurabilityRequirement.BEST_EFFORT,
        category=EvidencePersistenceFailureCategory.INTEGRITY,
    )
    assert disposition is PersistenceReliabilityDisposition.FAIL_CLOSED


def test_ee_b1_1_runtime_resolve_mandatory_infrastructure_raises() -> None:
    with pytest.raises(MandatoryEvidencePersistenceError):
        resolve_runtime_persistence_failure(
            requirement=EvidencePersistenceRequirement.MANDATORY,
            failure=EvidencePersistenceBoundaryError("append_failed"),
            event_type=RuntimeEventType.EXECUTION_FAILED,
        )
