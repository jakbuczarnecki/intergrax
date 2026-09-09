# © Artur Czarnecki. All rights reserved.

"""NPSC-5C — typed coordination intent contract tests."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from intergrax.agent_distribution.bounded_multi_agent_fanout import (
    MAX_FAN_OUT_CONCURRENCY,
    MAX_FAN_OUT_ITEMS,
)
from intergrax.agent_distribution.coordination_intent import (
    CoordinationContribution,
    CoordinationContributionId,
    CoordinationExecutionMode,
    CoordinationIntent,
    CoordinationIntentContractError,
    CoordinationIntentId,
    CoordinationIntentPlanner,
    MIN_FAN_OUT_CONTRIBUTIONS,
    effective_fan_out_max_concurrency,
    validate_coordination_contribution_id,
    validate_coordination_intent,
    validate_coordination_intent_id,
)
from intergrax.agent_distribution.capability_matching import (
    CapabilityId,
    build_agent_capability_requirement,
)
from intergrax.agent_distribution.task_capability_resolution import (
    AgentDistributionCapabilityNeed,
    AgentDistributionCapabilityNeedKind,
    TaskCapabilityResolutionContractError,
    build_task_capability_resolution_request,
    resolved_agent_distribution_capability_need,
    unresolved_agent_distribution_capability_need,
)
from tests.unit.agent_distribution.test_delegated_subtasks import OcrRequest

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _capability_need(task_kind: str = "document.ocr"):
    return unresolved_agent_distribution_capability_need(
        build_task_capability_resolution_request(task_kind=task_kind),
    )


def _contribution(
    contribution_id: str,
    *,
    document_ref: str = "doc-1",
) -> CoordinationContribution[OcrRequest]:
    return CoordinationContribution(
        contribution_id=CoordinationContributionId(contribution_id),
        payload=OcrRequest(document_ref=document_ref),
        capability_need=_capability_need(),
    )


def _single_intent(
    contribution_id: str = "contrib-a",
) -> CoordinationIntent[OcrRequest]:
    return CoordinationIntent(
        intent_id=CoordinationIntentId("intent-single"),
        mode=CoordinationExecutionMode.SINGLE,
        contributions=(_contribution(contribution_id),),
    )


def _fan_out_intent(
    contribution_ids: tuple[str, ...],
    *,
    requested_max_concurrency: int | None = None,
) -> CoordinationIntent[OcrRequest]:
    return CoordinationIntent(
        intent_id=CoordinationIntentId("intent-fan-out"),
        mode=CoordinationExecutionMode.FAN_OUT,
        contributions=tuple(_contribution(cid, document_ref=cid) for cid in contribution_ids),
        requested_max_concurrency=requested_max_concurrency,
    )


def test_validate_coordination_intent_id_rejects_non_string() -> None:
    with pytest.raises(TypeError, match="intent_id must be str"):
        validate_coordination_intent_id(42)


def test_validate_coordination_contribution_id_rejects_empty() -> None:
    with pytest.raises(ValueError, match="must be non-empty"):
        validate_coordination_contribution_id("   ")


def test_empty_intent_rejected() -> None:
    intent = CoordinationIntent(
        intent_id=CoordinationIntentId("intent-empty"),
        mode=CoordinationExecutionMode.SINGLE,
        contributions=(),
    )
    with pytest.raises(
        CoordinationIntentContractError,
        match="coordination intent must be non-empty",
    ):
        validate_coordination_intent(intent)


def test_duplicate_contribution_id_rejected() -> None:
    intent = CoordinationIntent(
        intent_id=CoordinationIntentId("intent-dup"),
        mode=CoordinationExecutionMode.FAN_OUT,
        contributions=(
            _contribution("contrib-a"),
            _contribution("contrib-a"),
        ),
    )
    with pytest.raises(
        CoordinationIntentContractError,
        match="duplicate contribution_id",
    ):
        validate_coordination_intent(intent)


def test_single_requires_exactly_one_contribution() -> None:
    intent = CoordinationIntent(
        intent_id=CoordinationIntentId("intent-single"),
        mode=CoordinationExecutionMode.SINGLE,
        contributions=(
            _contribution("contrib-a"),
            _contribution("contrib-b"),
        ),
    )
    with pytest.raises(
        CoordinationIntentContractError,
        match="exactly one contribution",
    ):
        validate_coordination_intent(intent)


def test_single_rejects_requested_max_concurrency() -> None:
    intent = CoordinationIntent(
        intent_id=CoordinationIntentId("intent-single"),
        mode=CoordinationExecutionMode.SINGLE,
        contributions=(_contribution("contrib-a"),),
        requested_max_concurrency=2,
    )
    with pytest.raises(
        CoordinationIntentContractError,
        match="must not set requested_max_concurrency",
    ):
        validate_coordination_intent(intent)


def test_fan_out_below_required_cardinality_rejected() -> None:
    intent = CoordinationIntent(
        intent_id=CoordinationIntentId("intent-fan-out"),
        mode=CoordinationExecutionMode.FAN_OUT,
        contributions=(_contribution("contrib-a"),),
    )
    with pytest.raises(
        CoordinationIntentContractError,
        match=f"at least {MIN_FAN_OUT_CONTRIBUTIONS}",
    ):
        validate_coordination_intent(intent)


def test_fan_out_above_max_items_rejected() -> None:
    contribution_ids = tuple(f"contrib-{index}" for index in range(MAX_FAN_OUT_ITEMS + 1))
    intent = _fan_out_intent(contribution_ids)
    with pytest.raises(
        CoordinationIntentContractError,
        match="exceeds platform limit",
    ):
        validate_coordination_intent(intent)


def test_invalid_max_concurrency_rejected() -> None:
    intent = _fan_out_intent(
        ("contrib-a", "contrib-b"),
        requested_max_concurrency=MAX_FAN_OUT_CONCURRENCY + 1,
    )
    with pytest.raises(
        CoordinationIntentContractError,
        match="requested_max_concurrency exceeds platform limit",
    ):
        validate_coordination_intent(intent)


def test_invalid_zero_max_concurrency_rejected() -> None:
    intent = _fan_out_intent(
        ("contrib-a", "contrib-b"),
        requested_max_concurrency=0,
    )
    with pytest.raises(
        CoordinationIntentContractError,
        match="must be positive",
    ):
        validate_coordination_intent(intent)


def test_typed_ids_preserved() -> None:
    intent = _single_intent("contrib-a")
    validate_coordination_intent(intent)
    assert intent.intent_id == CoordinationIntentId("intent-single")
    assert intent.contributions[0].contribution_id == CoordinationContributionId(
        "contrib-a",
    )
    assert validate_coordination_intent_id(intent.intent_id) == intent.intent_id


def test_intent_is_immutable() -> None:
    intent = _single_intent()
    with pytest.raises(AttributeError):
        intent.intent_id = CoordinationIntentId("other")  # type: ignore[misc]


def test_effective_fan_out_max_concurrency_defaults_to_item_count() -> None:
    intent = _fan_out_intent(("contrib-a", "contrib-b", "contrib-c"))
    assert effective_fan_out_max_concurrency(intent) == 3


def test_effective_fan_out_max_concurrency_honors_request_cap() -> None:
    intent = _fan_out_intent(
        ("contrib-a", "contrib-b", "contrib-c"),
        requested_max_concurrency=2,
    )
    assert effective_fan_out_max_concurrency(intent) == 2


@dataclass(frozen=True, slots=True)
class _StaticPlannerInput:
    contribution_id: str


class _StaticCoordinationIntentPlanner(
    CoordinationIntentPlanner[_StaticPlannerInput, OcrRequest],
):
    async def plan(self, input: _StaticPlannerInput) -> CoordinationIntent[OcrRequest]:
        return _single_intent(input.contribution_id)


@pytest.mark.asyncio
async def test_static_planner_produces_valid_intent() -> None:
    planner = _StaticCoordinationIntentPlanner()
    intent = await planner.plan(_StaticPlannerInput(contribution_id="contrib-static"))
    validate_coordination_intent(intent)
    assert intent.contributions[0].contribution_id == CoordinationContributionId(
        "contrib-static",
    )


def test_coordination_contribution_has_no_physical_agent_fields() -> None:
    fields = {field.name for field in CoordinationContribution.__dataclass_fields__.values()}
    forbidden = {
        "agent_id",
        "agent_instance_id",
        "lease_id",
    }
    assert forbidden.isdisjoint(fields)


def test_resolved_capability_need_accepts_multiple_capabilities() -> None:
    requirement = build_agent_capability_requirement(
        required=("invoice_ocr", "document.read"),
        optional=("citation.generate",),
    )
    contribution = CoordinationContribution(
        contribution_id=CoordinationContributionId("contrib-resolved"),
        payload=OcrRequest(document_ref="doc-resolved"),
        capability_need=resolved_agent_distribution_capability_need(requirement),
    )
    assert contribution.capability_need.kind is (
        AgentDistributionCapabilityNeedKind.RESOLVED_REQUIREMENT
    )
    assert contribution.capability_need.resolved_requirement == requirement


def test_decision_compatible_capability_id_projects_to_canonical_requirement() -> None:
    decision_capability_id = "invoice_ocr"
    requirement = build_agent_capability_requirement(required=(decision_capability_id,))
    need = resolved_agent_distribution_capability_need(requirement)
    required_ids = {
        item.value for item in need.resolved_requirement.required_capability_ids
    }
    assert required_ids == {decision_capability_id}
    assert CapabilityId(value=decision_capability_id) in (
        need.resolved_requirement.required_capability_ids
    )


def test_capability_need_rejects_mixed_variant_payload() -> None:
    with pytest.raises(
        TaskCapabilityResolutionContractError,
        match="unresolved_task capability need must not set resolved_requirement",
    ):
        AgentDistributionCapabilityNeed(
            kind=AgentDistributionCapabilityNeedKind.UNRESOLVED_TASK,
            unresolved_task=build_task_capability_resolution_request(
                task_kind="document.ocr",
            ),
            resolved_requirement=build_agent_capability_requirement(
                required=("document.ocr",),
            ),
        )
