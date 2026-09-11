# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Hold provider dependency permits through iterable stream consumption (W2-B3)."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import TypeVar

from intergrax.runtime.resilience.dependency_attempt_execution_boundary import (
    DependencyAttemptExecutionBoundary,
    DependencyAttemptHandle,
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
