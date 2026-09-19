# © Artur Czarnecki. All rights reserved.

"""System clock implementations for deadline projection."""

from __future__ import annotations

import time
from datetime import datetime, timezone

from intergrax.contracts.execution_deadline.clock import MonotonicClockPort, UtcClockPort


class SystemUtcClock:
    def now_utc(self) -> datetime:
        return datetime.now(timezone.utc)


class SystemMonotonicClock:
    def monotonic(self) -> float:
        return time.monotonic()
