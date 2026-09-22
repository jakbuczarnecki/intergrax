# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Adapter-instance stream transport registry (W4-D) — not process-global."""

from __future__ import annotations

from collections.abc import Callable


class ProviderStreamTransportRegistry:
    """Maps operation_id to a transport close hook for in-flight streams."""

    __slots__ = ("_closers",)

    def __init__(self) -> None:
        self._closers: dict[str, Callable[[], None]] = {}

    def register(self, operation_id: str, closer: Callable[[], None]) -> None:
        if type(operation_id) is not str or not operation_id:
            raise ValueError("operation_id must be a non-empty str")
        self._closers[operation_id] = closer

    def close_transport(self, operation_id: str) -> bool:
        closer = self._closers.pop(operation_id, None)
        if closer is None:
            return False
        closer()
        return True

    def clear(self) -> None:
        self._closers.clear()
