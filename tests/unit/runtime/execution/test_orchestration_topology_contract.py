# © Artur Czarnecki. All rights reserved.

"""Unit tests for canonical orchestration topology contracts."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from intergrax.contracts.orchestration_topology import (
    OrchestrationOutcomeValidationError,
    OrchestrationResult,
    OrchestrationSchedulingPolicy,
    OrchestrationSchedulingPolicyValidationError,
    OrchestrationSlot,
    OrchestrationSlotExecutionError,
    OrchestrationSlotFailure,
    OrchestrationSlotId,
    OrchestrationSlotOutcome,
    OrchestrationSlotStatus,
    OrchestrationTopology,
    OrchestrationTopologyValidationError,
    build_orchestration_result,
    orchestration_slot_order,
    resolve_effective_orchestration_concurrency,
    validate_orchestration_scheduling_policy,
    validate_orchestration_topology,
)
from intergrax.runtime.nexus.execution.orchestration_node_execution import (
    bind_orchestration_node_execution,
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


def test_build_orchestration_result_rejects_missing_outcome_ids() -> None:
    topology = OrchestrationTopology(slots=(_slot("a", 1), _slot("b", 2)))
    with pytest.raises(
        OrchestrationTopologyValidationError,
        match=r"missing orchestration outcomes: \['a', 'b'\]",
    ):
        build_orchestration_result(topology, outcomes_by_slot={})


def test_build_orchestration_result_rejects_extra_outcome_ids() -> None:
    topology = OrchestrationTopology(slots=(_slot("a", 1),))
    with pytest.raises(
        OrchestrationTopologyValidationError,
        match=r"extra orchestration outcomes: \['x'\]",
    ):
        build_orchestration_result(
            topology,
            outcomes_by_slot={
                OrchestrationSlotId("a"): OrchestrationSlotOutcome(
                    slot_id=OrchestrationSlotId("a"),
                    status=OrchestrationSlotStatus.SUCCESS,
                    result=1,
                ),
                OrchestrationSlotId("x"): OrchestrationSlotOutcome(
                    slot_id=OrchestrationSlotId("x"),
                    status=OrchestrationSlotStatus.SUCCESS,
                    result=9,
                ),
            },
        )


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


def test_resolve_effective_orchestration_concurrency_unbounded_when_both_none() -> None:
    assert resolve_effective_orchestration_concurrency(None, None) is None


def test_resolve_effective_orchestration_concurrency_uses_single_limit() -> None:
    assert resolve_effective_orchestration_concurrency(3, None) == 3
    assert resolve_effective_orchestration_concurrency(None, 4) == 4


def test_resolve_effective_orchestration_concurrency_uses_minimum() -> None:
    assert resolve_effective_orchestration_concurrency(10, 3) == 3
    assert resolve_effective_orchestration_concurrency(3, 10) == 3


def test_orchestration_slot_outcome_rejects_success_with_failure() -> None:
    with pytest.raises(OrchestrationOutcomeValidationError, match="SUCCESS"):
        OrchestrationSlotOutcome(
            slot_id=OrchestrationSlotId("a"),
            status=OrchestrationSlotStatus.SUCCESS,
            result=1,
            failure=OrchestrationSlotFailure(code="x", message="y"),
        )


def test_orchestration_slot_outcome_rejects_failure_without_failure() -> None:
    with pytest.raises(OrchestrationOutcomeValidationError, match="FAILURE"):
        OrchestrationSlotOutcome(
            slot_id=OrchestrationSlotId("a"),
            status=OrchestrationSlotStatus.FAILURE,
        )


def test_orchestration_slot_outcome_rejects_failure_with_result() -> None:
    with pytest.raises(OrchestrationOutcomeValidationError, match="FAILURE"):
        OrchestrationSlotOutcome(
            slot_id=OrchestrationSlotId("a"),
            status=OrchestrationSlotStatus.FAILURE,
            result=1,
            failure=OrchestrationSlotFailure(code="x", message="y"),
        )


def test_orchestration_slot_outcome_rejects_skipped_with_result() -> None:
    with pytest.raises(OrchestrationOutcomeValidationError, match="SKIPPED"):
        OrchestrationSlotOutcome(
            slot_id=OrchestrationSlotId("a"),
            status=OrchestrationSlotStatus.SKIPPED,
            result=1,
        )


def test_orchestration_slot_outcome_allows_success_with_none_result() -> None:
    outcome = OrchestrationSlotOutcome(
        slot_id=OrchestrationSlotId("a"),
        status=OrchestrationSlotStatus.SUCCESS,
        result=None,
    )
    assert outcome.result is None
    assert outcome.failure is None


def test_orchestration_slot_failure_rejects_empty_code() -> None:
    with pytest.raises(OrchestrationOutcomeValidationError, match="code"):
        OrchestrationSlotFailure(code="", message="x")


def test_orchestration_slot_failure_rejects_empty_message() -> None:
    with pytest.raises(OrchestrationOutcomeValidationError, match="message"):
        OrchestrationSlotFailure(code="x", message="")


@pytest.mark.asyncio
async def test_bound_node_execution_projects_typed_slot_failure() -> None:
    class _FailingExecutor:
        async def execute_slot(
            self,
            *,
            slot_id: OrchestrationSlotId,
            payload: _Payload,
        ) -> int:
            del slot_id, payload
            raise OrchestrationSlotExecutionError(
                code="expected_failure",
                message="slot failed",
            )

    node_execution = bind_orchestration_node_execution(
        payloads={OrchestrationSlotId("a"): _Payload(1)},
        slot_executor=_FailingExecutor(),
    )
    outcome = await node_execution.execute_node(slot_id=OrchestrationSlotId("a"))
    assert outcome.status is OrchestrationSlotStatus.FAILURE
    assert outcome.failure is not None
    assert outcome.failure.code == "expected_failure"
    assert outcome.failure.message == "slot failed"


@pytest.mark.asyncio
async def test_bound_node_execution_propagates_programming_errors() -> None:
    class _BuggyExecutor:
        async def execute_slot(
            self,
            *,
            slot_id: OrchestrationSlotId,
            payload: _Payload,
        ) -> int:
            del slot_id, payload
            raise TypeError("bug")

    node_execution = bind_orchestration_node_execution(
        payloads={OrchestrationSlotId("a"): _Payload(1)},
        slot_executor=_BuggyExecutor(),
    )
    with pytest.raises(TypeError, match="bug"):
        await node_execution.execute_node(slot_id=OrchestrationSlotId("a"))
