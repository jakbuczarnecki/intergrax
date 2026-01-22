# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from fastapi import Depends, HTTPException, status

from intergrax.fastapi_core.rate_limit.keys import RateLimitKey
from intergrax.fastapi_core.rate_limit.policy import RateLimitPolicy



class NoOpRateLimitPolicy(RateLimitPolicy):
    """
    Default rate limit policy that allows all requests.

    Used as a safe skeleton implementation.
    """

    def allow(self, key: RateLimitKey) -> bool:
        return True


def rate_limit(
    key: RateLimitKey,
    policy: RateLimitPolicy = Depends(NoOpRateLimitPolicy),
) -> None:
    """
    FastAPI dependency enforcing rate limiting.

    Usage:
        Depends(rate_limit(RateLimitKey.TENANT))
    """
    if not policy.allow(key):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
        )
