# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod

from intergrax.fastapi_core.rate_limit.keys import RateLimitKey


class RateLimitPolicy(ABC):
    """
    Abstract rate limiting policy.

    Implementations decide whether a request is allowed
    for a given rate limit key.
    """

    @abstractmethod
    def allow(self, key: RateLimitKey) -> bool:
        """
        Return True if request is allowed, False otherwise.
        """
        raise NotImplementedError
