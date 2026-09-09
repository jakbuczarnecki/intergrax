# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed multi-agent coordination intent contracts (NPSC-5C).

Semantic description of *what* coordinated specialist work is requested, without
runtime scheduling topology, physical agent identity, or execution lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Generic, NewType, Protocol, TypeVar, runtime_checkable

from intergrax.agent_distribution.bounded_multi_agent_fanout import (
    MAX_FAN_OUT_CONCURRENCY,
    MAX_FAN_OUT_ITEMS,
)
from intergrax.agent_distribution.multi_agent_coordination import CoordinationPolicy
from intergrax.agent_distribution.task_capability_resolution import (
    TaskCapabilityResolutionRequest,
)

MIN_FAN_OUT_CONTRIBUTIONS: Final = 2

CoordinationIntentId = NewType("CoordinationIntentId", str)
CoordinationContributionId = NewType("CoordinationContributionId", str)

InputT = TypeVar("InputT")
RequestT = TypeVar("RequestT")


def _strip_required(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("must be non-empty")
    return normalized


def validate_coordination_intent_id(value: object) -> CoordinationIntentId:
    if type(value) is not str:
        raise TypeError("intent_id must be str")
    return CoordinationIntentId(_strip_required(value))


def validate_coordination_contribution_id(
    value: object,
) -> CoordinationContributionId:
    if type(value) is not str:
        raise TypeError("contribution_id must be str")
    return CoordinationContributionId(_strip_required(value))


class CoordinationIntentContractError(ValueError):
    """Malformed coordination intent or contribution contract."""


class CoordinationExecutionMode(StrEnum):
    """Semantic execution shape for one coordination intent."""

    SINGLE = "single"
    FAN_OUT = "fan_out"


@dataclass(frozen=True, slots=True)
class CoordinationContribution(Generic[RequestT]):
    """One bounded specialist contribution within a coordination intent."""

    contribution_id: CoordinationContributionId
    payload: RequestT
    capability_requirement: TaskCapabilityResolutionRequest
    policy: CoordinationPolicy = CoordinationPolicy()


@dataclass(frozen=True, slots=True)
class CoordinationIntent(Generic[RequestT]):
    """Semantic multi-agent work request independent of runtime topology."""

    intent_id: CoordinationIntentId
    mode: CoordinationExecutionMode
    contributions: tuple[CoordinationContribution[RequestT], ...]
    requested_max_concurrency: int | None = None


@runtime_checkable
class CoordinationIntentPlanner(Protocol[InputT, RequestT]):
    """Neutral producer boundary: context input → typed coordination intent."""

    async def plan(self, input: InputT) -> CoordinationIntent[RequestT]:
        ...


def validate_coordination_intent(intent: CoordinationIntent[object]) -> None:
    """Fail-closed validation for coordination intent invariants."""
    try:
        validate_coordination_intent_id(intent.intent_id)
    except (TypeError, ValueError) as exc:
        raise CoordinationIntentContractError(str(exc)) from exc

    if not intent.contributions:
        raise CoordinationIntentContractError("coordination intent must be non-empty")

    if intent.mode is CoordinationExecutionMode.SINGLE:
        if len(intent.contributions) != 1:
            raise CoordinationIntentContractError(
                "SINGLE coordination intent requires exactly one contribution",
            )
        if intent.requested_max_concurrency is not None:
            raise CoordinationIntentContractError(
                "SINGLE coordination intent must not set requested_max_concurrency",
            )
    elif intent.mode is CoordinationExecutionMode.FAN_OUT:
        if len(intent.contributions) < MIN_FAN_OUT_CONTRIBUTIONS:
            raise CoordinationIntentContractError(
                f"FAN_OUT coordination intent requires at least "
                f"{MIN_FAN_OUT_CONTRIBUTIONS} contributions",
            )
        if len(intent.contributions) > MAX_FAN_OUT_ITEMS:
            raise CoordinationIntentContractError(
                f"FAN_OUT contribution count exceeds platform limit {MAX_FAN_OUT_ITEMS}",
            )
        _validate_requested_max_concurrency(intent.requested_max_concurrency)
    else:
        raise CoordinationIntentContractError(
            f"unsupported coordination execution mode: {intent.mode}",
        )

    seen_contribution_ids: set[CoordinationContributionId] = set()
    for contribution in intent.contributions:
        try:
            validate_coordination_contribution_id(contribution.contribution_id)
        except (TypeError, ValueError) as exc:
            raise CoordinationIntentContractError(str(exc)) from exc
        if contribution.contribution_id in seen_contribution_ids:
            raise CoordinationIntentContractError(
                f"duplicate contribution_id: {contribution.contribution_id}",
            )
        seen_contribution_ids.add(contribution.contribution_id)


def _validate_requested_max_concurrency(value: int | None) -> None:
    if value is None:
        return
    if value <= 0:
        raise CoordinationIntentContractError(
            "requested_max_concurrency must be positive when set",
        )
    if value > MAX_FAN_OUT_CONCURRENCY:
        raise CoordinationIntentContractError(
            f"requested_max_concurrency exceeds platform limit {MAX_FAN_OUT_CONCURRENCY}",
        )


def effective_fan_out_max_concurrency(
    intent: CoordinationIntent[object],
) -> int:
    """Resolve bounded fan-out concurrency from intent semantics."""
    validate_coordination_intent(intent)
    if intent.mode is not CoordinationExecutionMode.FAN_OUT:
        raise CoordinationIntentContractError(
            "effective_fan_out_max_concurrency requires FAN_OUT intent",
        )
    requested = intent.requested_max_concurrency
    item_count = len(intent.contributions)
    if requested is None:
        return min(item_count, MAX_FAN_OUT_CONCURRENCY)
    return min(requested, item_count, MAX_FAN_OUT_CONCURRENCY)


__all__ = [
    "CoordinationContribution",
    "CoordinationContributionId",
    "CoordinationExecutionMode",
    "CoordinationIntent",
    "CoordinationIntentContractError",
    "CoordinationIntentId",
    "CoordinationIntentPlanner",
    "MIN_FAN_OUT_CONTRIBUTIONS",
    "effective_fan_out_max_concurrency",
    "validate_coordination_contribution_id",
    "validate_coordination_intent",
    "validate_coordination_intent_id",
]
