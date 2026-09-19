# © Artur Czarnecki. All rights reserved.

"""Neutral clock ports for execution deadline projection."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol


class UtcClockPort(Protocol):
    def now_utc(self) -> datetime:
        """Return current instant as timezone-aware UTC."""


class MonotonicClockPort(Protocol):
    def monotonic(self) -> float:
        """Return monotonic seconds suitable for intra-process deadlines."""
