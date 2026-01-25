# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations


class RateLimitExceededError(Exception):
    """
    Raised when a rate limit policy denies the request.
    """

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or "Rate limit exceeded.")
