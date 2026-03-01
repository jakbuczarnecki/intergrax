# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from intergrax.distributed.contracts.rate_limiter import (
    DistributedRateLimiter,
    RateLimitResult,
)
from intergrax.fastapi_core.rate_limit.keys import RateLimitKey
from intergrax.fastapi_core.rate_limit.policy import RateLimitPolicy


@dataclass(frozen=True)
class RateLimitConfig:
    capacity: int
    refill_rate_per_second: float


class DistributedRateLimitPolicy(RateLimitPolicy):
    """
    Production-grade RateLimitPolicy backed by a DistributedRateLimiter.

    Responsibilities:
    - Map FastAPI RateLimitKey → distributed bucket key.
    - Delegate token bucket logic to DistributedRateLimiter.
    - Return boolean decision only (no algorithm duplication).

    Does NOT:
    - Implement token bucket logic.
    - Store state.
    - Perform retries.
    """

    def __init__(
        self,
        *,
        limiter: DistributedRateLimiter,
        configs: Mapping[RateLimitKey, RateLimitConfig],
    ) -> None:
        self._limiter = limiter
        self._configs = configs

    def allow(self, key: RateLimitKey, identity: str) -> bool:
        config = self._configs.get(key)

        # If no configuration defined for key → deny by design (explicit config required)
        if config is None:
            return False

        result: RateLimitResult = self._limiter.acquire(
            tenant_id=identity,
            key=key.value,
            capacity=config.capacity,
            refill_rate_per_second=config.refill_rate_per_second,
        )

        return result.allowed