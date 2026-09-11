# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Adapter-instance stream transport registry (W4-D) — not process-global."""

from __future__ import annotations

from collections.abc import Callable

from intergrax.runtime.external_operations.operation_termination import (
    pop_stream_transport_closer,
    register_stream_transport_closer,
)


class ProviderStreamTransportRegistry:
    """Maps operation_id to a transport close hook for in-flight streams."""

    __slots__ = ("_closers",)

    def __init__(self) -> None:
        self._closers: dict[str, Callable[[], None]] = {}

    def register(self, operation_id: str, closer: Callable[[], None]) -> None:
        register_stream_transport_closer(
            self._closers,
            operation_id=operation_id,
            closer=closer,
        )

    def close_transport(self, operation_id: str) -> bool:
        closer = pop_stream_transport_closer(
            self._closers,
            operation_id=operation_id,
        )
        if closer is None:
            return False
        closer()
        return True

    def clear(self) -> None:
        self._closers.clear()
