# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass

import pytest

from intergrax.rag.retrieval.multichannel import (
    MultiChannelRetrievalContractError,
    MultiChannelRetrievalCoordinator,
    MultiChannelRetrievalResult,
    RetrievalChannelFailure,
    RetrievalChannelKey,
    RetrievalChannelOutcome,
    RetrievalChannelStatus,
    SequentialMultiChannelRetrievalCoordinator,
)
from intergrax.rag.retrieval.multichannel.contracts import RetrievalChannelOperation


@dataclass(frozen=True, slots=True)
class SampleDocumentHit:
    document_id: str


def _key(name: str) -> RetrievalChannelKey:
    return RetrievalChannelKey(value=name)


@dataclass
class _RecordingOperation:
    channel_key: RetrievalChannelKey
    outcome: RetrievalChannelOutcome[SampleDocumentHit]
    calls: int = 0

    def execute(self) -> RetrievalChannelOutcome[SampleDocumentHit]:
        self.calls += 1
        return self.outcome


def _success_op(channel: str, document_id: str) -> _RecordingOperation:
    key = _key(channel)
    return _RecordingOperation(
        channel_key=key,
        outcome=RetrievalChannelOutcome.succeeded(
            channel_key=key,
            result=SampleDocumentHit(document_id=document_id),
        ),
    )


def test_single_channel_success() -> None:
    coordinator = SequentialMultiChannelRetrievalCoordinator[SampleDocumentHit]()
    op = _success_op("alpha", "doc-1")
    result = coordinator.execute((op,))
    assert len(result.outcomes) == 1
    assert result.outcomes[0].status is RetrievalChannelStatus.SUCCEEDED
    assert result.outcomes[0].result == SampleDocumentHit(document_id="doc-1")
    assert op.calls == 1


def test_multiple_successes_preserve_order() -> None:
    coordinator = SequentialMultiChannelRetrievalCoordinator[SampleDocumentHit]()
    ops = (
        _success_op("a", "1"),
        _success_op("b", "2"),
        _success_op("c", "3"),
    )
    result = coordinator.execute(ops)
    assert tuple(o.channel_key.value for o in result.outcomes) == ("a", "b", "c")


def test_mixed_outcomes_all_returned() -> None:
    coordinator = SequentialMultiChannelRetrievalCoordinator[SampleDocumentHit]()
    key_b = _key("b")
    key_c = _key("c")
    ops = (
        _success_op("a", "1"),
        _RecordingOperation(
            channel_key=key_b,
            outcome=RetrievalChannelOutcome.failed(
                channel_key=key_b,
                failure=RetrievalChannelFailure(
                    failure_code="lookup_error",
                    message="channel b failed",
                    retryable=False,
                ),
            ),
        ),
        _RecordingOperation(
            channel_key=key_c,
            outcome=RetrievalChannelOutcome.skipped(
                channel_key=key_c,
                skip_reason="not applicable",
            ),
        ),
    )
    result = coordinator.execute(ops)
    assert len(result.outcomes) == 3
    assert result.outcomes[1].status is RetrievalChannelStatus.FAILED
    assert result.outcomes[2].status is RetrievalChannelStatus.SKIPPED
    assert len(result.successful) == 1
    assert len(result.failed) == 1
    assert len(result.skipped) == 1


def test_failed_first_channel_does_not_abort_later() -> None:
    coordinator = SequentialMultiChannelRetrievalCoordinator[SampleDocumentHit]()
    key_a = _key("a")
    op_a = _RecordingOperation(
        channel_key=key_a,
        outcome=RetrievalChannelOutcome.failed(
            channel_key=key_a,
            failure=RetrievalChannelFailure(
                failure_code="x",
                message="first failed",
                retryable=True,
            ),
        ),
    )
    op_b = _success_op("b", "still-runs")
    result = coordinator.execute((op_a, op_b))
    assert op_b.calls == 1
    assert result.outcomes[1].status is RetrievalChannelStatus.SUCCEEDED


def test_duplicate_channel_keys_fail_before_execute() -> None:
    coordinator = SequentialMultiChannelRetrievalCoordinator[SampleDocumentHit]()
    op1 = _success_op("dup", "1")
    op2 = _success_op("dup", "2")
    with pytest.raises(MultiChannelRetrievalContractError, match="duplicate"):
        coordinator.execute((op1, op2))
    assert op1.calls == 0
    assert op2.calls == 0


def test_empty_operations_tuple() -> None:
    coordinator = SequentialMultiChannelRetrievalCoordinator[SampleDocumentHit]()
    result = coordinator.execute(())
    assert result.outcomes == ()
    assert result.successful == ()
    assert result.failed == ()
    assert result.skipped == ()


def test_operation_invoked_exactly_once() -> None:
    coordinator = SequentialMultiChannelRetrievalCoordinator[SampleDocumentHit]()
    op = _success_op("once", "x")
    coordinator.execute((op,))
    coordinator.execute((op,))
    assert op.calls == 2


def test_result_collection_immutable() -> None:
    coordinator = SequentialMultiChannelRetrievalCoordinator[SampleDocumentHit]()
    result = coordinator.execute((_success_op("a", "1"),))
    with pytest.raises(FrozenInstanceError):
        result.outcomes = ()


def test_invalid_succeeded_without_result() -> None:
    key = _key("a")
    with pytest.raises(MultiChannelRetrievalContractError, match="SUCCEEDED"):
        RetrievalChannelOutcome(
            channel_key=key,
            status=RetrievalChannelStatus.SUCCEEDED,
            result=None,
        )


def test_invalid_failed_without_failure() -> None:
    key = _key("a")
    with pytest.raises(MultiChannelRetrievalContractError, match="FAILED"):
        RetrievalChannelOutcome(
            channel_key=key,
            status=RetrievalChannelStatus.FAILED,
        )


def test_invalid_skipped_without_reason() -> None:
    key = _key("a")
    with pytest.raises(MultiChannelRetrievalContractError, match="SKIPPED"):
        RetrievalChannelOutcome.skipped(channel_key=key, skip_reason="   ")


class _ReverseOrderCoordinator:
    def execute(
        self,
        operations: tuple[RetrievalChannelOperation[SampleDocumentHit], ...],
    ) -> MultiChannelRetrievalResult[SampleDocumentHit]:
        outcomes = tuple(operation.execute() for operation in reversed(operations))
        return MultiChannelRetrievalResult(outcomes=outcomes)


def _run_with_coordinator(
    coordinator: MultiChannelRetrievalCoordinator[SampleDocumentHit],
) -> tuple[str, ...]:
    ops = (
        _success_op("first", "1"),
        _success_op("second", "2"),
    )
    result = coordinator.execute(ops)
    return tuple(o.channel_key.value for o in result.outcomes)


def test_replacement_implementation_contract_proof() -> None:
    sequential_keys = _run_with_coordinator(
        SequentialMultiChannelRetrievalCoordinator[SampleDocumentHit]()
    )
    reverse_keys = _run_with_coordinator(_ReverseOrderCoordinator())
    assert sequential_keys == ("first", "second")
    assert reverse_keys == ("second", "first")


def test_genericity_no_vpi_payload() -> None:
    coordinator = SequentialMultiChannelRetrievalCoordinator[SampleDocumentHit]()
    result = coordinator.execute(
        (
            _success_op("lexical", "hit-9"),
            _success_op("vector", "hit-10"),
        )
    )
    assert all(isinstance(o.result, SampleDocumentHit) for o in result.successful)
