# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""Per-call SDK clients for adapters whose SDK takes its timeout at client construction (EX-D3-01).

Some SDKs (boto3 ``Config(read_timeout=)``, ``ollama.Client(timeout=)``, ``ChatOllama(client_kwargs=)``)
accept a timeout only when the client is built. When the adapter built its own client, a call that
carries an effective timeout runs on a client derived for that call, and the derived client is released
when the call ends: after a buffered call returns or raises, and after a stream is exhausted, fails or
is closed. The adapter's long-lived client serves every call without a timeout. An injected client
cannot be rebuilt; such an adapter declares ``supports_call_timeout() -> False`` and fails closed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class _SupportsClose(Protocol):
    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class CallClient[C]:
    """A client derived for one call and the function that releases it."""

    client: C
    release: Callable[[], None]


class ReleasingIterator[T]:
    """Iterates ``inner`` and runs ``release`` exactly once when it is exhausted, fails or is closed."""

    __slots__ = ("_inner", "_release")

    def __init__(self, inner: Iterator[T], release: Callable[[], None]) -> None:
        self._inner = inner
        self._release: Callable[[], None] | None = release

    def __iter__(self) -> ReleasingIterator[T]:
        return self

    def __next__(self) -> T:
        try:
            return next(self._inner)
        except BaseException:
            self._finish()
            raise

    def close(self) -> None:
        try:
            if isinstance(self._inner, _SupportsClose):
                self._inner.close()
        finally:
            self._finish()

    def _finish(self) -> None:
        release, self._release = self._release, None
        if release is not None:
            release()


def call_on_client[C, R](
    *,
    shared: C,
    derive: Callable[[float], CallClient[C]] | None,
    timeout: float | None,
    call: Callable[[C], R],
) -> R:
    """Run ``call`` on ``shared``, or on a client derived for ``timeout`` that is released afterwards."""
    if timeout is None or derive is None:
        return call(shared)
    derived = derive(timeout)
    try:
        return call(derived.client)
    finally:
        derived.release()


def stream_on_client[C, T](
    *,
    shared: C,
    derive: Callable[[float], CallClient[C]] | None,
    timeout: float | None,
    open_stream: Callable[[C], Iterable[T]],
) -> Iterable[T]:
    """Open a stream on ``shared``, or on a derived client that is released when the stream ends."""
    if timeout is None or derive is None:
        return open_stream(shared)
    derived = derive(timeout)
    try:
        stream = iter(open_stream(derived.client))
    except BaseException:
        derived.release()
        raise
    return ReleasingIterator(stream, derived.release)


__all__ = [
    "CallClient",
    "ReleasingIterator",
    "call_on_client",
    "stream_on_client",
]
