# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from enum import Enum


class RateLimitKey(str, Enum):
    """
    Rate limiting key types.

    Used to determine how requests are grouped for limiting.
    """
    REQUEST = "request"
    USER = "user"
    TENANT = "tenant"
