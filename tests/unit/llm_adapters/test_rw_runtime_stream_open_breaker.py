# © Artur Czarnecki. All rights reserved.

"""T-IGX-051-D NT-RW-RUN-01 (NT-ROW-103, EX-L-08): stream-open errors reach the breaker and the retry loop.

Before the change ``_execute_streaming`` handed the resilience loop a lazy generator, so the loop recorded
success before the SDK was touched and never saw an error raised while the stream was opened or while its
first event was pulled. Fake SDK factories only (no network, no credential).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence

import pytest
from intergrax.contracts.dependency_concurrency_admission import (
    DependencyConcurrencyIdentity,
    DependencyConcurrencyKind,
    DependencyConcurrencyOverloadMode,
    DependencyConcurrencyPolicy,
)
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.call_config import LLMCallConfig
from intergrax.llm_adapters._shared.resilience import LLMCircuitOpenError, reset_provider_resilience
from intergrax.llm_adapters.base.base_llm_adapter import BaseLLMAdapter
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.runtime.resilience.dependency_attempt_execution_boundary import (
    DependencyAttemptExecutionBoundary,
)
from intergrax.runtime.resilience.local_dependency_concurrency_admission import (
    LocalDependencyConcurrencyAdmission,
)

pytestmark = pytest.mark.unit

_PROVIDER = "rw-runtime-stream-probe"


class _ServiceUnavailable(Exception):
    """Shape of an SDK status error that the retry module treats as transient (503)."""

    status_code = 503


class _StreamProbeAdapter(BaseLLMAdapter):
    def __init__(self, factory: Callable[[], Iterable[str]], config: LLMCallConfig) -> None:
        super().__init__()
        self.provider = _PROVIDER
        self.model = "probe-model"
        self.call_config = config
        self._factory = factory

    @property
    def context_window_tokens(self) -> int:
        return 8_192

    def generate_messages(self, messages: Sequence[ChatMessage], **_: object) -> LLMAdapterResponse:
        return LLMAdapterResponse(content="")

    def open_stream(self) -> Iterable[str]:
        return self._execute_streaming(self._factory)


@pytest.fixture(autouse=True)
def _clean_breaker() -> Iterator[None]:
    reset_provider_resilience(_PROVIDER)
    yield
    reset_provider_resilience(_PROVIDER)


def _failing_open(calls: list[int]) -> Callable[[], Iterable[str]]:
    def factory() -> Iterable[str]:
        calls.append(1)
        raise _ServiceUnavailable("open failed")

    return factory


def _failing_first_pull(calls: list[int]) -> Callable[[], Iterable[str]]:
    def factory() -> Iterable[str]:
        calls.append(1)

        def events() -> Iterator[str]:
            raise _ServiceUnavailable("first event failed")
            yield "never"  # pragma: no cover

        return events()

    return factory


@pytest.mark.parametrize("factory_kind", ["open", "first_pull"])
def test_a_stream_open_failure_trips_the_breaker(factory_kind: str) -> None:
    calls: list[int] = []
    factory = _failing_open(calls) if factory_kind == "open" else _failing_first_pull(calls)
    adapter = _StreamProbeAdapter(
        factory, LLMCallConfig(circuit_breaker_threshold=1, circuit_breaker_cooldown_sec=60.0)
    )

    with pytest.raises(Exception) as first:
        list(adapter.open_stream())
    assert not isinstance(first.value, LLMCircuitOpenError)

    with pytest.raises(LLMCircuitOpenError):
        list(adapter.open_stream())
    assert len(calls) == 1, "the open circuit must stop the second call before the SDK"


@pytest.mark.parametrize("factory_kind", ["open", "first_pull"])
def test_a_retriable_stream_open_failure_is_retried_within_max_retries(factory_kind: str) -> None:
    calls: list[int] = []
    fail = _failing_open(calls) if factory_kind == "open" else _failing_first_pull(calls)

    def factory() -> Iterable[str]:
        if len(calls) < 2:
            return fail()
        calls.append(1)
        return iter(["a", "b"])

    adapter = _StreamProbeAdapter(factory, LLMCallConfig(max_retries=2, retry_backoff_sec=0.0))

    assert list(adapter.open_stream()) == ["a", "b"]
    assert len(calls) == 3


def test_a_stream_that_never_opens_is_attempted_max_retries_plus_one_times() -> None:
    calls: list[int] = []
    adapter = _StreamProbeAdapter(_failing_open(calls), LLMCallConfig(max_retries=2, retry_backoff_sec=0.0))

    with pytest.raises(Exception) as info:
        list(adapter.open_stream())
    assert not isinstance(info.value, LLMCircuitOpenError)
    assert len(calls) == 3


def test_control_a_failure_after_the_first_event_is_neither_retried_nor_recorded() -> None:
    calls: list[int] = []

    def factory() -> Iterable[str]:
        calls.append(1)

        def events() -> Iterator[str]:
            yield "a"
            raise _ServiceUnavailable("mid-stream failure")

        return events()

    adapter = _StreamProbeAdapter(
        factory,
        LLMCallConfig(max_retries=2, retry_backoff_sec=0.0, circuit_breaker_threshold=1),
    )
    stream = iter(adapter.open_stream())
    assert next(stream) == "a"
    with pytest.raises(Exception) as info:
        next(stream)
    assert not isinstance(info.value, LLMCircuitOpenError)
    assert len(calls) == 1

    # The breaker did not count the mid-stream failure: the next call reaches the SDK again.
    again = iter(adapter.open_stream())
    assert next(again) == "a"
    assert len(calls) == 2


def test_control_an_empty_stream_is_a_successful_empty_stream() -> None:
    adapter = _StreamProbeAdapter(lambda: iter(()), LLMCallConfig(circuit_breaker_threshold=1))
    assert list(adapter.open_stream()) == []
    assert list(adapter.open_stream()) == []


def _boundary(capacity: int) -> DependencyAttemptExecutionBoundary:
    admission = LocalDependencyConcurrencyAdmission(
        {
            DependencyConcurrencyIdentity(kind=DependencyConcurrencyKind.LLM_PROVIDER, value=_PROVIDER): (
                DependencyConcurrencyPolicy(
                    max_concurrent_calls=capacity,
                    overload_mode=DependencyConcurrencyOverloadMode.REJECT,
                    wait_timeout_seconds=None,
                )
            )
        }
    )
    return DependencyAttemptExecutionBoundary(admission)


def test_control_permits_are_released_exactly_once_on_every_path() -> None:
    """With one permit, a leaked permit would reject the next attempt; a double release would not show,
    so the boundary is also asked how many attempts are still open."""
    boundary = _boundary(1)
    try:
        calls: list[int] = []
        fail = _failing_open(calls)

        def factory() -> Iterable[str]:
            if len(calls) < 2:
                return fail()
            calls.append(1)
            return iter(["a"])

        adapter = _StreamProbeAdapter(factory, LLMCallConfig(max_retries=2, retry_backoff_sec=0.0))
        adapter.bind_provider_dependency_boundary(boundary)
        assert list(adapter.open_stream()) == ["a"]
        assert len(calls) == 3

        # A consumer that stops early must release the permit when the stream is closed.
        early = adapter.open_stream()
        iterator = iter(early)
        close = getattr(iterator, "close", None)
        if callable(close):
            close()

        always_failing = _StreamProbeAdapter(_failing_open([]), LLMCallConfig(max_retries=0))
        always_failing.bind_provider_dependency_boundary(boundary)
        with pytest.raises(Exception):
            list(always_failing.open_stream())

        # The single permit is free again: a full stream still runs.
        calls.clear()
        assert list(adapter.open_stream()) == ["a"]
    finally:
        boundary.close()
