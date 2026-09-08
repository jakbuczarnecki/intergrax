# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Bounded multi-agent fan-out / fan-in coordination (NPSC-5B)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Generic, NewType, Protocol, TypeVar

from intergrax.agent_distribution.errors import AgentDistributionError
from intergrax.agent_distribution.multi_agent_coordination import (
    CoordinationCleanupError,
    CoordinationDelegation,
    CoordinationError,
    CoordinationFailureCode,
    CoordinationRequest,
    CoordinationResult,
    MultiAgentCoordinationService,
)
from intergrax.contracts.agent_run import RequestIdentity

MAX_FAN_OUT_CONCURRENCY: Final = 64
MAX_FAN_OUT_ITEMS: Final = 256

FanOutId = NewType("FanOutId", str)
FanOutItemId = NewType("FanOutItemId", str)

RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")


def _strip_required(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("must be non-empty")
    return normalized


def validate_fan_out_id(value: object) -> FanOutId:
    if type(value) is not str:
        raise TypeError("fan_out_id must be str")
    return FanOutId(_strip_required(value))


def validate_fan_out_item_id(value: object) -> FanOutItemId:
    if type(value) is not str:
        raise TypeError("item_id must be str")
    return FanOutItemId(_strip_required(value))


class FanOutFailureCode(StrEnum):
    """Bounded semantic categories for fan-out boundary violations."""

    INVALID_FAN_OUT = "invalid_fan_out"
    EXECUTOR_CONTRACT_VIOLATION = "executor_contract_violation"


class FanOutError(AgentDistributionError):
    """Base error for bounded multi-agent fan-out violations."""

    failure_code: FanOutFailureCode

    def __init__(
        self,
        message: str,
        *,
        failure_code: FanOutFailureCode,
    ) -> None:
        super().__init__(message)
        self.failure_code = failure_code


class InvalidFanOutError(FanOutError):
    """Malformed fan-out request or projection."""

    def __init__(self, message: str) -> None:
        super().__init__(
            message,
            failure_code=FanOutFailureCode.INVALID_FAN_OUT,
        )


class FanOutExecutorContractError(FanOutError):
    """Executor returned outcomes that violate the fan-out contract."""

    def __init__(self, message: str) -> None:
        super().__init__(
            message,
            failure_code=FanOutFailureCode.EXECUTOR_CONTRACT_VIOLATION,
        )


class FanOutItemStatus(StrEnum):
    """Per-item fan-out execution disposition."""

    SUCCESS = "success"
    FAILURE = "failure"


@dataclass(frozen=True, slots=True)
class FanOutItemFailure(Generic[ResultT]):
    """Typed per-item failure projected from coordination errors."""

    failure_code: CoordinationFailureCode
    message: str
    partial_result: ResultT | None = None


@dataclass(frozen=True, slots=True)
class FanOutItem(Generic[RequestT]):
    """One bounded specialist contribution within a fan-out operation."""

    item_id: FanOutItemId
    request: CoordinationRequest
    delegation: CoordinationDelegation[RequestT]


@dataclass(frozen=True, slots=True)
class FanOutRequest(Generic[RequestT]):
    """Parent intent for bounded parallel specialist coordination."""

    fan_out_id: FanOutId
    items: tuple[FanOutItem[RequestT], ...]
    max_concurrency: int


@dataclass(frozen=True, slots=True)
class FanOutItemOutcome(Generic[ResultT]):
    """Deterministic per-item fan-out result aligned to request identity."""

    item_id: FanOutItemId
    status: FanOutItemStatus
    result: CoordinationResult[ResultT] | None = None
    failure: FanOutItemFailure[ResultT] | None = None

    def __post_init__(self) -> None:
        if self.status is FanOutItemStatus.SUCCESS:
            if self.result is None or self.failure is not None:
                raise ValueError(
                    "SUCCESS outcome requires result and forbids failure",
                )
            return
        if self.status is FanOutItemStatus.FAILURE:
            if self.result is not None or self.failure is None:
                raise ValueError(
                    "FAILURE outcome requires failure and forbids result",
                )
            return
        raise ValueError(f"unsupported fan-out item status: {self.status}")


@dataclass(frozen=True, slots=True)
class FanOutResult(Generic[ResultT]):
    """Aggregate fan-out outcome in stable request order."""

    fan_out_id: FanOutId
    items: tuple[FanOutItemOutcome[ResultT], ...]

    @property
    def all_succeeded(self) -> bool:
        return all(item.status is FanOutItemStatus.SUCCESS for item in self.items)

    @property
    def any_failed(self) -> bool:
        return any(item.status is FanOutItemStatus.FAILURE for item in self.items)


def validate_fan_out_request(request: FanOutRequest[object]) -> None:
    """Fail-closed validation for fan-out request contracts."""
    try:
        validate_fan_out_id(request.fan_out_id)
    except (TypeError, ValueError) as exc:
        raise InvalidFanOutError(str(exc)) from exc

    if not request.items:
        raise InvalidFanOutError("fan-out items must be non-empty")

    if request.max_concurrency <= 0:
        raise InvalidFanOutError("max_concurrency must be positive")

    if request.max_concurrency > MAX_FAN_OUT_CONCURRENCY:
        raise InvalidFanOutError(
            f"max_concurrency exceeds platform limit {MAX_FAN_OUT_CONCURRENCY}",
        )

    if len(request.items) > MAX_FAN_OUT_ITEMS:
        raise InvalidFanOutError(
            f"fan-out item count exceeds platform limit {MAX_FAN_OUT_ITEMS}",
        )

    seen_item_ids: set[FanOutItemId] = set()
    for item in request.items:
        try:
            validate_fan_out_item_id(item.item_id)
        except (TypeError, ValueError) as exc:
            raise InvalidFanOutError(str(exc)) from exc
        if item.item_id in seen_item_ids:
            raise InvalidFanOutError(
                f"duplicate fan-out item_id: {item.item_id}",
            )
        seen_item_ids.add(item.item_id)


async def _coordinate_item(
    coordination: MultiAgentCoordinationService[RequestT, ResultT],
    item: FanOutItem[RequestT],
    *,
    principal: RequestIdentity,
) -> FanOutItemOutcome[ResultT]:
    try:
        coordination_result = await coordination.coordinate(
            item.request,
            delegation=item.delegation,
            principal=principal,
        )
    except CoordinationCleanupError as exc:
        return FanOutItemOutcome(
            item_id=item.item_id,
            status=FanOutItemStatus.FAILURE,
            failure=FanOutItemFailure(
                failure_code=CoordinationFailureCode.LEASE_RELEASE_FAILED,
                message=str(exc),
                partial_result=exc.result,
            ),
        )
    except CoordinationError as exc:
        return FanOutItemOutcome(
            item_id=item.item_id,
            status=FanOutItemStatus.FAILURE,
            failure=FanOutItemFailure(
                failure_code=exc.failure_code,
                message=str(exc),
            ),
        )
    return FanOutItemOutcome(
        item_id=item.item_id,
        status=FanOutItemStatus.SUCCESS,
        result=coordination_result,
    )


def _normalize_executor_outcomes(
    request_items: tuple[FanOutItem[RequestT], ...],
    outcomes: tuple[FanOutItemOutcome[ResultT], ...],
) -> tuple[FanOutItemOutcome[ResultT], ...]:
    """Validate executor output and project outcomes in request order."""
    if len(outcomes) != len(request_items):
        raise FanOutExecutorContractError(
            "executor outcome count must match request item count",
        )

    outcome_by_id: dict[FanOutItemId, FanOutItemOutcome[ResultT]] = {}
    for outcome in outcomes:
        if outcome.item_id in outcome_by_id:
            raise FanOutExecutorContractError(
                f"duplicate executor outcome item_id: {outcome.item_id}",
            )
        outcome_by_id[outcome.item_id] = outcome

    ordered: list[FanOutItemOutcome[ResultT]] = []
    for item in request_items:
        outcome = outcome_by_id.get(item.item_id)
        if outcome is None:
            raise FanOutExecutorContractError(
                f"missing executor outcome for item_id: {item.item_id}",
            )
        ordered.append(outcome)

    return tuple(ordered)


class BoundedFanOutExecutor(Protocol[RequestT, ResultT]):
    """Variation point for bounded fan-out execution without lifecycle ownership."""

    async def execute(
        self,
        *,
        items: tuple[FanOutItem[RequestT], ...],
        max_concurrency: int,
        coordination: MultiAgentCoordinationService[RequestT, ResultT],
        principal: RequestIdentity,
    ) -> tuple[FanOutItemOutcome[ResultT], ...]: ...


class AsyncioSemaphoreBoundedFanOutExecutor(Generic[RequestT, ResultT]):
    """Default bounded fan-out executor using local asyncio concurrency control."""

    async def execute(
        self,
        *,
        items: tuple[FanOutItem[RequestT], ...],
        max_concurrency: int,
        coordination: MultiAgentCoordinationService[RequestT, ResultT],
        principal: RequestIdentity,
    ) -> tuple[FanOutItemOutcome[ResultT], ...]:
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _run_item(
            item: FanOutItem[RequestT],
        ) -> FanOutItemOutcome[ResultT]:
            async with semaphore:
                return await _coordinate_item(
                    coordination,
                    item,
                    principal=principal,
                )

        return tuple(
            await asyncio.gather(*(_run_item(item) for item in items)),
        )


class BoundedMultiAgentFanOutService(Generic[RequestT, ResultT]):
    """Coordinate many bounded specialist delegations with deterministic fan-in."""

    __slots__ = ("_coordination", "_executor")

    def __init__(
        self,
        *,
        coordination: MultiAgentCoordinationService[RequestT, ResultT],
        executor: BoundedFanOutExecutor[RequestT, ResultT] | None = None,
    ) -> None:
        self._coordination = coordination
        self._executor = executor or AsyncioSemaphoreBoundedFanOutExecutor[
            RequestT,
            ResultT,
        ]()

    async def fan_out(
        self,
        request: FanOutRequest[RequestT],
        *,
        principal: RequestIdentity,
    ) -> FanOutResult[ResultT]:
        validate_fan_out_request(request)
        raw_outcomes = await self._executor.execute(
            items=request.items,
            max_concurrency=request.max_concurrency,
            coordination=self._coordination,
            principal=principal,
        )
        outcomes = _normalize_executor_outcomes(request.items, raw_outcomes)
        return FanOutResult(
            fan_out_id=request.fan_out_id,
            items=outcomes,
        )


__all__ = [
    "AsyncioSemaphoreBoundedFanOutExecutor",
    "BoundedFanOutExecutor",
    "BoundedMultiAgentFanOutService",
    "FanOutError",
    "FanOutExecutorContractError",
    "FanOutFailureCode",
    "FanOutId",
    "FanOutItem",
    "FanOutItemFailure",
    "FanOutItemId",
    "FanOutItemOutcome",
    "FanOutItemStatus",
    "FanOutRequest",
    "FanOutResult",
    "InvalidFanOutError",
    "MAX_FAN_OUT_CONCURRENCY",
    "MAX_FAN_OUT_ITEMS",
    "validate_fan_out_id",
    "validate_fan_out_item_id",
    "validate_fan_out_request",
]
