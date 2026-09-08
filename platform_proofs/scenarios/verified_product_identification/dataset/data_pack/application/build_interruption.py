"""Graceful interruption handling for long-running data pack builds."""

from __future__ import annotations

import signal
from collections.abc import Iterator
from contextlib import contextmanager
from types import FrameType


class DataPackBuildInterrupted(BaseException):
    """Raised when the operator interrupts a build via SIGTERM or KeyboardInterrupt."""


@contextmanager
def graceful_build_interruption_scope() -> Iterator[None]:
    """Convert SIGTERM into a catchable interruption while preserving prior handlers."""
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def _handle_sigterm(signum: int, frame: FrameType | None) -> None:
        raise DataPackBuildInterrupted(f"received signal {signum}")

    signal.signal(signal.SIGTERM, _handle_sigterm)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
