# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Hold provider dependency permits through iterable stream consumption (W2-B3)."""

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


def admission_bounded_iterable(
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


def stream_factory_with_admission(
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


def stream_with_external_operation_lifecycle(
    *,
    ext_op: LlmExternalOperationAttempt,
    factory: Callable[[], Iterable[T]],
    stream_registry: ProviderStreamTransportRegistry | None,
) -> Iterable[T]:
    """Wrap stream consumption with W4-D transport registration and terminal CAS."""

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
        try:
            yield from stream
        except BaseException:
            ext_op.mark_failed()
            raise
        else:
            ext_op.mark_succeeded()
        finally:
            if stream_registry is not None and operation_id is not None:
                stream_registry.close_transport(operation_id)

    return _gen()
