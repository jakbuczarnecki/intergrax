# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from enum import Enum


class ApiErrorType(str, Enum):
    """
    Minimal API-level error taxonomy.

    This taxonomy is transport-oriented and intentionally decoupled
    from runtime / agent error codes.

    It will be mapped to domain-specific error taxonomies
    (runtime, agents, tools) in higher layers.
    """

    BAD_REQUEST = "bad_request"
    UNAUTHORIZED = "unauthorized"
    FORBIDDEN = "forbidden"
    NOT_FOUND = "not_found"

    RATE_LIMITED = "rate_limited"

    INTERNAL_ERROR = "internal_error"
