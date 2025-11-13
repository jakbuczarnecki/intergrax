# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
import asyncio
import time
from typing import Optional


class TokenBucket:
    """
    Simple asyncio-compatible token bucket rate limiter.

    Usage:
        bucket = TokenBucket(rate_per_sec=2.0, capacity=5)
        await bucket.acquire()  # waits until at least 1 token is available

    Parameters:
      rate_per_sec : average token refill rate (tokens per second)
      capacity     : maximum number of tokens stored in the bucket

    Behavior:
      - Tokens accumulate over time up to 'capacity'.
      - 'acquire(n)' waits until at least n tokens are available,
        then consumes them atomically.
      - Designed to be used across concurrent coroutines (single process).
    """

    def __init__(self, rate_per_sec: float, capacity: int) -> None:
        if rate_per_sec <= 0:
            raise ValueError("TokenBucket: rate_per_sec must be > 0.")
        if capacity <= 0:
            raise ValueError("TokenBucket: capacity must be > 0.")

        self._rate = float(rate_per_sec)
        self._capacity = int(capacity)
        self._tokens: float = float(capacity)
        self._timestamp: float = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> None:
        """
        Waits until 'tokens' tokens are available and consumes them.

        This method is safe to call from multiple coroutines concurrently.
        It guarantees that the total consumption never exceeds capacity
        and that the average request rate does not exceed 'rate_per_sec'.
        """
        if tokens <= 0:
            return

        async with self._lock:
            while True:
                now = time.monotonic()
                elapsed = now - self._timestamp
                self._timestamp = now

                # Refill tokens based on elapsed time.
                self._tokens = min(
                    self._capacity,
                    self._tokens + elapsed * self._rate,
                )

                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return

                # Not enough tokens: compute wait time until enough tokens accumulate.
                missing = tokens - self._tokens
                wait_seconds = max(missing / self._rate, 0.001)
                await asyncio.sleep(wait_seconds)

    def try_acquire(self, tokens: int = 1) -> bool:
        """
        Non-blocking attempt to consume 'tokens' tokens.

        Returns:
          True  if tokens were available and consumed,
          False if not enough tokens are currently available.

        This method does not wait; it is intended for lightweight checks
        in paths where best-effort limiting is sufficient.
        """
        if tokens <= 0:
            return True

        now = time.monotonic()
        elapsed = now - self._timestamp
        self._timestamp = now

        self._tokens = min(
            self._capacity,
            self._tokens + elapsed * self._rate,
        )

        if self._tokens >= tokens:
            self._tokens -= tokens
            return True

        return False
