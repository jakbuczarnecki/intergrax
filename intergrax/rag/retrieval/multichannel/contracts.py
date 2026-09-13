# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Generic multi-channel retrieval coordination contracts (platform-owned envelope)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Generic, Protocol, TypeVar

from intergrax.rag.retrieval.multichannel.errors import MultiChannelRetrievalContractError

TResult = TypeVar("TResult")


class RetrievalChannelStatus(str, Enum):
    SUCCEEDED = "succeeded"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class RetrievalChannelKey:
    """Stable, scenario-defined channel identity (not a platform enum of channel kinds)."""

    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str):
            raise TypeError("RetrievalChannelKey.value must be str")
        if self.value != self.value.strip():
            raise MultiChannelRetrievalContractError(
                "RetrievalChannelKey.value must be trimmed"
            )
        if not self.value:
            raise MultiChannelRetrievalContractError(
                "RetrievalChannelKey.value must be non-empty"
            )


@dataclass(frozen=True, slots=True)
class RetrievalChannelFailure:
    failure_code: str
    message: str
    retryable: bool

    def __post_init__(self) -> None:
        if not isinstance(self.failure_code, str) or not self.failure_code.strip():
            raise MultiChannelRetrievalContractError(
                "RetrievalChannelFailure.failure_code must be a non-empty string"
            )
        if not isinstance(self.message, str) or not self.message.strip():
            raise MultiChannelRetrievalContractError(
                "RetrievalChannelFailure.message must be a non-empty string"
            )
        if not isinstance(self.retryable, bool):
            raise TypeError("RetrievalChannelFailure.retryable must be bool")


@dataclass(frozen=True, slots=True)
class RetrievalChannelOutcome(Generic[TResult]):
    channel_key: RetrievalChannelKey
    status: RetrievalChannelStatus
    result: TResult | None = None
    failure: RetrievalChannelFailure | None = None
    skip_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.channel_key, RetrievalChannelKey):
            raise TypeError("channel_key must be RetrievalChannelKey")
        if not isinstance(self.status, RetrievalChannelStatus):
            raise TypeError("status must be RetrievalChannelStatus")

        if self.status is RetrievalChannelStatus.SUCCEEDED:
            if self.result is None:
                raise MultiChannelRetrievalContractError(
                    "SUCCEEDED outcome requires result"
                )
            if self.failure is not None or self.skip_reason is not None:
                raise MultiChannelRetrievalContractError(
                    "SUCCEEDED outcome must not include failure or skip_reason"
                )
        elif self.status is RetrievalChannelStatus.FAILED:
            if self.failure is None:
                raise MultiChannelRetrievalContractError(
                    "FAILED outcome requires failure"
                )
            if self.result is not None or self.skip_reason is not None:
                raise MultiChannelRetrievalContractError(
                    "FAILED outcome must not include result or skip_reason"
                )
        elif self.status is RetrievalChannelStatus.SKIPPED:
            if self.skip_reason is None or not self.skip_reason.strip():
                raise MultiChannelRetrievalContractError(
                    "SKIPPED outcome requires a non-empty skip_reason"
                )
            if self.result is not None or self.failure is not None:
                raise MultiChannelRetrievalContractError(
                    "SKIPPED outcome must not include result or failure"
                )

    @classmethod
    def succeeded(
        cls,
        *,
        channel_key: RetrievalChannelKey,
        result: TResult,
    ) -> RetrievalChannelOutcome[TResult]:
        return cls(
            channel_key=channel_key,
            status=RetrievalChannelStatus.SUCCEEDED,
            result=result,
        )

    @classmethod
    def failed(
        cls,
        *,
        channel_key: RetrievalChannelKey,
        failure: RetrievalChannelFailure,
    ) -> RetrievalChannelOutcome[TResult]:
        return cls(
            channel_key=channel_key,
            status=RetrievalChannelStatus.FAILED,
            failure=failure,
        )

    @classmethod
    def skipped(
        cls,
        *,
        channel_key: RetrievalChannelKey,
        skip_reason: str,
    ) -> RetrievalChannelOutcome[TResult]:
        return cls(
            channel_key=channel_key,
            status=RetrievalChannelStatus.SKIPPED,
            skip_reason=skip_reason,
        )


@dataclass(frozen=True, slots=True)
class MultiChannelRetrievalResult(Generic[TResult]):
    """Immutable aggregate of per-channel outcomes in execution order."""

    outcomes: tuple[RetrievalChannelOutcome[TResult], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.outcomes, tuple):
            raise TypeError("outcomes must be a tuple")
        for outcome in self.outcomes:
            if not isinstance(outcome, RetrievalChannelOutcome):
                raise TypeError(
                    "outcomes must contain only RetrievalChannelOutcome values"
                )

    @property
    def successful(self) -> tuple[RetrievalChannelOutcome[TResult], ...]:
        return tuple(
            outcome
            for outcome in self.outcomes
            if outcome.status is RetrievalChannelStatus.SUCCEEDED
        )

    @property
    def failed(self) -> tuple[RetrievalChannelOutcome[TResult], ...]:
        return tuple(
            outcome
            for outcome in self.outcomes
            if outcome.status is RetrievalChannelStatus.FAILED
        )

    @property
    def skipped(self) -> tuple[RetrievalChannelOutcome[TResult], ...]:
        return tuple(
            outcome
            for outcome in self.outcomes
            if outcome.status is RetrievalChannelStatus.SKIPPED
        )


class RetrievalChannelOperation(Protocol[TResult]):
    @property
    def channel_key(self) -> RetrievalChannelKey:
        ...

    def execute(self) -> RetrievalChannelOutcome[TResult]:
        ...


class MultiChannelRetrievalCoordinator(Protocol[TResult]):
    def execute(
        self,
        operations: tuple[RetrievalChannelOperation[TResult], ...],
    ) -> MultiChannelRetrievalResult[TResult]:
        ...
