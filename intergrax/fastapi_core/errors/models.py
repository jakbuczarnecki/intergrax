# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ErrorResponse:
    """
    Stable API error envelope.

    Notes:
    - This is an API-facing contract.
    - Do NOT include stack traces or internal details.
    """
    error_type: str
    message: str
    request_id: str
    details: Optional[str] = None
