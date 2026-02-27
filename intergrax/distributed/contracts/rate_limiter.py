# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class RateLimitResult:
    """
    Result of a distributed rate limit acquire() call.
    """

    allowed: bool
    remaining_tokens: float
    retry_after_seconds: float


class DistributedRateLimiter(ABC):
    """
    Distributed rate limiter contract (token bucket).

    Must be safe under multi-worker concurrency.
    Must operate correctly in distributed environments (e.g., Redis-backed).
    """

    @abstractmethod
    def acquire(
        self,
        *,
        tenant_id: str,
        key: str,
        capacity: int,
        refill_rate_per_second: float,
    ) -> RateLimitResult:
        """
        Attempts to consume one token from a distributed token bucket.

        Parameters:
            tenant_id: Tenant scope.
            key: Logical rate limit key (e.g., "llm", "websearch").
            capacity: Maximum burst size (bucket capacity).
            refill_rate_per_second: Token refill rate per second.

        Returns:
            RateLimitResult
        """
        raise NotImplementedError