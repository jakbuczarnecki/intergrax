# © Artur Czarnecki. All rights reserved.

"""Unit tests for canonical orchestration topology contracts."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from intergrax.contracts.orchestration_topology import (
    OrchestrationResult,
    OrchestrationSchedulingPolicy,
    OrchestrationSchedulingPolicyValidationError,
    OrchestrationSlot,
    OrchestrationSlotFailure,
    OrchestrationSlotId,
    OrchestrationSlotOutcome,
    OrchestrationSlotStatus,
    OrchestrationTopology,
    OrchestrationTopologyValidationError,
    build_orchestration_result,
    orchestration_slot_order,
    validate_orchestration_scheduling_policy,
    validate_orchestration_topology,
)

pytestmark = pytest.mark.unit


@dataclass(frozen=True, slots=True)
class _Payload:
    value: int


def _slot(
    slot_id: str,
    value: int,
    *,
    depends_on: tuple[str, ...] = (),
) -> OrchestrationSlot[_Payload]:
    return OrchestrationSlot(
        slot_id=OrchestrationSlotId(slot_id),
        payload=_Payload(value),
        depends_on=tuple(OrchestrationSlotId(dep) for dep in depends_on),
    )


def test_validate_orchestration_topology_rejects_empty() -> None:
    with pytest.raises(OrchestrationTopologyValidationError, match="must contain slots"):
        validate_orchestration_topology(OrchestrationTopology(slots=()))


def test_validate_orchestration_topology_rejects_duplicate_slot_ids() -> None:
    topology = OrchestrationTopology(
        slots=(
            _slot("a", 1),
            _slot("a", 2),
        )
    )
    with pytest.raises(OrchestrationTopologyValidationError, match="duplicate"):
        validate_orchestration_topology(topology)


def test_validate_orchestration_topology_rejects_missing_dependency() -> None:
    topology = OrchestrationTopology(
        slots=(
            _slot("a", 1, depends_on=("missing",)),
        )
    )
    with pytest.raises(OrchestrationTopologyValidationError, match="missing orchestration slot dependency"):
        validate_orchestration_topology(topology)


def test_validate_orchestration_topology_rejects_cycle() -> None:
    topology = OrchestrationTopology(
        slots=(
            _slot("a", 1, depends_on=("b",)),
            _slot("b", 2, depends_on=("a",)),
        )
    )
    with pytest.raises(OrchestrationTopologyValidationError, match="cycle"):
        validate_orchestration_topology(topology)


def test_validate_orchestration_scheduling_policy_rejects_invalid_concurrency() -> None:
    with pytest.raises(OrchestrationSchedulingPolicyValidationError):
        validate_orchestration_scheduling_policy(
            OrchestrationSchedulingPolicy(max_concurrency=0)
        )


def test_orchestration_slot_order_follows_submission_order() -> None:
    topology = OrchestrationTopology(
        slots=(
            _slot("a", 1),
            _slot("b", 2),
            _slot("c", 3),
        )
    )
    assert orchestration_slot_order(topology) == (
        OrchestrationSlotId("a"),
        OrchestrationSlotId("b"),
        OrchestrationSlotId("c"),
    )


def test_build_orchestration_result_preserves_submission_order() -> None:
    topology = OrchestrationTopology(
        slots=(
            _slot("a", 1),
            _slot("b", 2),
            _slot("c", 3),
        )
    )
    outcomes_by_slot = {
        OrchestrationSlotId("c"): OrchestrationSlotOutcome(
            slot_id=OrchestrationSlotId("c"),
            status=OrchestrationSlotStatus.SUCCESS,
            result=9,
        ),
        OrchestrationSlotId("a"): OrchestrationSlotOutcome(
            slot_id=OrchestrationSlotId("a"),
            status=OrchestrationSlotStatus.SUCCESS,
            result=1,
        ),
        OrchestrationSlotId("b"): OrchestrationSlotOutcome(
            slot_id=OrchestrationSlotId("b"),
            status=OrchestrationSlotStatus.FAILURE,
            failure=OrchestrationSlotFailure(code="x", message="y"),
        ),
    }
    result = build_orchestration_result(topology, outcomes_by_slot=outcomes_by_slot)
    assert [outcome.slot_id for outcome in result.outcomes] == [
        OrchestrationSlotId("a"),
        OrchestrationSlotId("b"),
        OrchestrationSlotId("c"),
    ]
    assert result.outcomes[1].status is OrchestrationSlotStatus.FAILURE


def test_build_orchestration_result_rejects_unknown_outcome_ids() -> None:
    topology = OrchestrationTopology(slots=(_slot("a", 1),))
    with pytest.raises(OrchestrationTopologyValidationError, match="missing orchestration outcome"):
        build_orchestration_result(topology, outcomes_by_slot={})


def test_build_orchestration_result_rejects_mismatched_outcome_slot_id() -> None:
    topology = OrchestrationTopology(slots=(_slot("a", 1),))
    with pytest.raises(OrchestrationTopologyValidationError, match="does not match topology slot"):
        build_orchestration_result(
            topology,
            outcomes_by_slot={
                OrchestrationSlotId("a"): OrchestrationSlotOutcome(
                    slot_id=OrchestrationSlotId("b"),
                    status=OrchestrationSlotStatus.SUCCESS,
                    result=1,
                )
            },
        )


def test_orchestration_result_cardinality_matches_topology() -> None:
    topology = OrchestrationTopology(
        slots=(
            _slot("a", 1),
            _slot("b", 2),
        )
    )
    result: OrchestrationResult[int] = build_orchestration_result(
        topology,
        outcomes_by_slot={
            OrchestrationSlotId("a"): OrchestrationSlotOutcome(
                slot_id=OrchestrationSlotId("a"),
                status=OrchestrationSlotStatus.SUCCESS,
                result=1,
            ),
            OrchestrationSlotId("b"): OrchestrationSlotOutcome(
                slot_id=OrchestrationSlotId("b"),
                status=OrchestrationSlotStatus.SUCCESS,
                result=2,
            ),
        },
    )
    assert len(result.outcomes) == len(topology.slots)
