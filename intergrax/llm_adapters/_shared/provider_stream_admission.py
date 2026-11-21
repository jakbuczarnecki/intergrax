# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Hold provider dependency permits through iterable stream consumption (W2-B3).

R1: ``stream_with_external_operation_lifecycle`` assumes ``ext_op`` already passed
admission governance when ``admission_attempt`` is bound on ``LlmExternalOperationAttempt``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING, TypeVar

from intergrax.runtime.resilience.dependency_attempt_execution_boundary import (
    DependencyAttemptExecutionBoundary,
    DependencyAttemptHandle,
)

if TYPE_CHECKING:
    from intergrax.llm_adapters._shared.provider_stream_transport_registry import (
        ProviderStreamTransportRegistry,
    )
    from intergrax.runtime.external_operations.llm_external_operation_attempt import (
        LlmExternalOperationAttempt,
    )

T = TypeVar("T")


def admission_bounded_iterable[T](
    *,
    boundary: DependencyAttemptExecutionBoundary,
    handle: DependencyAttemptHandle,
    iterable: Iterable[T],
) -> Iterable[T]:
    """Release permit exactly once when iteration ends, closes, or errors."""

    def _gen() -> Iterator[T]:
        try:
            yield from iterable
        finally:
            boundary.complete_direct(handle)

    return _gen()


def stream_factory_with_admission[T](
    *,
    boundary: DependencyAttemptExecutionBoundary,
    handle: DependencyAttemptHandle,
    factory: Callable[[], Iterable[T]],
) -> Iterable[T]:
    try:
        stream = factory()
    except BaseException:
        boundary.complete_direct(handle)
        raise
    return admission_bounded_iterable(boundary=boundary, handle=handle, iterable=stream)


def stream_with_external_operation_lifecycle[T](
    *,
    ext_op: LlmExternalOperationAttempt,
    factory: Callable[[], Iterable[T]],
    stream_registry: ProviderStreamTransportRegistry | None,
    map_error: Callable[[Exception], BaseException | None] | None = None,
) -> Iterable[T]:
    """Wrap stream consumption with W4-D transport registration and terminal CAS.

    ``map_error`` (optional) converts an ``Exception`` raised *after* the first event was yielded into
    the typed provider error (raised ``from`` the original). An error raised while the stream is opened
    or while its first event is pulled is re-raised unchanged, so the resilience loop that primes the
    stream (``open_primed_stream``) sees the SDK exception itself.
    """

    def _gen() -> Iterator[T]:
        from intergrax.utils import attribute_access

        operation_id = ext_op.operation_id
        try:
            stream = factory()
        except BaseException:
            ext_op.mark_failed()
            raise
        if stream_registry is not None and operation_id is not None:
            close_method = attribute_access.optional(stream, "close", None)
            if callable(close_method):

                def _close_transport() -> None:
                    close_method()

                stream_registry.register(operation_id, _close_transport)
        iterator = iter(stream)
        yielded = False
        try:
            for item in iterator:
                yielded = True
                yield item
        except GeneratorExit:
            ext_op.mark_failed()
            _close_iterator(iterator)
            raise
        except BaseException as exc:
            ext_op.mark_failed()
            if yielded and map_error is not None and isinstance(exc, Exception):
                mapped = map_error(exc)
                if mapped is not None:
                    raise mapped from exc
            raise
        else:
            ext_op.mark_succeeded()
        finally:
            if stream_registry is not None and operation_id is not None:
                stream_registry.close_transport(operation_id)

    return _gen()


def _close_iterator(iterator: object) -> None:
    from intergrax.utils import attribute_access

    close_method = attribute_access.optional(iterator, "close", None)
    if callable(close_method):
        close_method()


class _PrimedStream[S]:
    """An opened stream whose first event was already pulled; ``close`` closes the underlying stream."""

    __slots__ = ("_head", "_iterator")

    def __init__(self, first: S, iterator: Iterator[S]) -> None:
        # The pulled first event waits here until it is yielded; popping it drops the reference.
        self._head: list[S] = [first]
        self._iterator = iterator

    def __iter__(self) -> _PrimedStream[S]:
        return self

    def __next__(self) -> S:
        if self._head:
            return self._head.pop()
        return next(self._iterator)

    def close(self) -> None:
        self._head.clear()
        _close_iterator(self._iterator)


def open_primed_stream[S](stream: Iterable[S]) -> Iterable[S]:
    """Open ``stream`` and pull its first event now (EX-L-08).

    Called inside one physical attempt of the resilience loop, so an error raised while the provider
    stream is opened, or while its first event is pulled, fails that attempt: the circuit breaker records
    it and the retry loop classifies it. The returned iterable yields the first event and then the rest;
    its ``close`` closes the underlying stream (permit release, transport close and the terminal state of
    the external operation run in the wrapper's ``finally``). An empty stream returns an empty iterable.
    An error at the first pull closes the underlying iterator and is re-raised unchanged.
    """
    iterator = iter(stream)
    try:
        first = next(iterator)
    except StopIteration:
        return iter(())
    except BaseException:
        _close_iterator(iterator)
        raise
    return _PrimedStream(first, iterator)
