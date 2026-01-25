# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from typing import Callable

from fastapi import Depends, HTTPException, Request, status

from intergrax.fastapi_core.context import RequestContext, get_request_context
from intergrax.fastapi_core.rate_limit.errors import RateLimitExceededError
from intergrax.fastapi_core.rate_limit.keys import RateLimitKey
from intergrax.fastapi_core.rate_limit.policy import RateLimitPolicy



class NoOpRateLimitPolicy(RateLimitPolicy):
    """
    Default rate limit policy that allows all requests.

    Used as a safe skeleton implementation.
    """

    def allow(self, key: RateLimitKey, identity: str) -> bool:
        return True
    
    
class RateLimitRequired:
    """
    FastAPI dependency enforcing rate limiting using RateLimitPolicy.

    Notes:
    - Policy is injected via DI (create_app).
    - This dependency only enforces the decision.
    """

    def __call__(
        self,
        policy: RateLimitPolicy = Depends(),
        context: RequestContext = Depends(get_request_context),
    ) -> None:
        allowed: bool = policy.allow(context)

        if not allowed:
            raise RateLimitExceededError()
        


def rate_limit(
    key: RateLimitKey,
    policy: RateLimitPolicy = Depends(),
) -> Callable[[Request], None]:
    """
    FastAPI dependency enforcing rate limiting.
    """
    def _dependency(request: Request) -> None:
        ctx = get_request_context(request)

        if key == RateLimitKey.TENANT:
            identity = ctx.tenant_id
        elif key == RateLimitKey.USER:
            identity = ctx.user_id
        else:
            identity = ctx.request_id

        if identity is None:
            # No identity -> treat as anonymous bucket
            identity = "anonymous"

        if not policy.allow(key, identity):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
            )

    return _dependency

