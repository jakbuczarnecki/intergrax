# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Lock
from typing import Dict

from intergrax.fastapi_core.rate_limit.keys import RateLimitKey


@dataclass
class _Bucket:
    tokens: float
    last_refill: float


class InMemoryRateLimitPolicy:
    """
    Simple in-memory token bucket rate limiting policy.

    Notes:
    - Process-local (not suitable for multi-instance).
    - Thread-safe.
    """

    def __init__(
        self,
        capacity: int,
        refill_rate_per_sec: float,
    ) -> None:
        self._capacity = float(capacity)
        self._refill_rate = float(refill_rate_per_sec)
        self._buckets: Dict[str, _Bucket] = {}
        self._lock = Lock()

    def allow(self, key: RateLimitKey, identity: str) -> bool:
        """
        Check and consume a token for the given identity.
        """
        now = time.monotonic()

        with self._lock:
            bucket = self._buckets.get(identity)
            if bucket is None:
                bucket = _Bucket(tokens=self._capacity, last_refill=now)
                self._buckets[identity] = bucket

            # Refill tokens
            elapsed = now - bucket.last_refill
            bucket.tokens = min(
                self._capacity,
                bucket.tokens + elapsed * self._refill_rate,
            )
            bucket.last_refill = now

            if bucket.tokens < 1.0:
                return False

            bucket.tokens -= 1.0
            return True
