# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Optional

import pytest

from intergrax.distributed.contracts.rate_limiter import (
    DistributedRateLimiter,
    RateLimitResult,
)
from intergrax.fastapi_core.rate_limit.distributed_policy import (
    DistributedRateLimitPolicy,
    RateLimitConfig,
)
from intergrax.fastapi_core.rate_limit.keys import RateLimitKey


pytestmark = pytest.mark.unit

class DummyLimiter(DistributedRateLimiter):
    def __init__(self, result: RateLimitResult) -> None:
        self._result = result
        self.last_call: Optional[dict[str, object]] = None

    def acquire(
        self,
        *,
        tenant_id: str,
        key: str,
        capacity: int,
        refill_rate_per_second: float,
    ) -> RateLimitResult:
        self.last_call = {
            "tenant_id": tenant_id,
            "key": key,
            "capacity": capacity,
            "refill_rate_per_second": refill_rate_per_second,
        }
        return self._result


def test_allow_returns_true_when_limiter_allows() -> None:
    result = RateLimitResult(
        allowed=True,
        remaining_tokens=5.0,
        retry_after_seconds=0.0,
    )
    limiter = DummyLimiter(result)

    configs = {
        RateLimitKey.REQUEST: RateLimitConfig(
            capacity=10,
            refill_rate_per_second=1.0,
        )
    }

    policy = DistributedRateLimitPolicy(
        limiter=limiter,
        configs=configs,
    )

    allowed = policy.allow(RateLimitKey.REQUEST, "tenant_A")

    assert allowed is True
    assert limiter.last_call is not None
    assert limiter.last_call["tenant_id"] == "tenant_A"
    assert limiter.last_call["key"] == RateLimitKey.REQUEST.value
    assert limiter.last_call["capacity"] == 10
    assert limiter.last_call["refill_rate_per_second"] == 1.0


def test_allow_returns_false_when_limiter_denies() -> None:
    result = RateLimitResult(
        allowed=False,
        remaining_tokens=0.0,
        retry_after_seconds=1.5,
    )
    limiter = DummyLimiter(result)

    configs = {
        RateLimitKey.REQUEST: RateLimitConfig(
            capacity=5,
            refill_rate_per_second=0.5,
        )
    }

    policy = DistributedRateLimitPolicy(
        limiter=limiter,
        configs=configs,
    )

    allowed = policy.allow(RateLimitKey.REQUEST, "tenant_B")

    assert allowed is False


def test_allow_returns_false_when_no_config_defined() -> None:
    result = RateLimitResult(
        allowed=True,
        remaining_tokens=10.0,
        retry_after_seconds=0.0,
    )
    limiter = DummyLimiter(result)

    policy = DistributedRateLimitPolicy(
        limiter=limiter,
        configs={},  # no key configured
    )

    allowed = policy.allow(RateLimitKey.REQUEST, "tenant_X")

    assert allowed is False
    assert limiter.last_call is None