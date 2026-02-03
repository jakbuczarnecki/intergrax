# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class FailureKind(str, Enum):
    TRANSIENT = "transient"
    PERMANENT = "permanent"
    TIMEOUT = "timeout"
    CANCELED = "canceled"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class FailureInfo:
    """
    Typed classification result used by retry policy.

    NOTE:
    - error_type/message are persisted today by RunService.mark_failed()
    - kind/retry_after_seconds drive governance decisions (retry / backoff)
    """
    error_type: str
    error_message: str
    kind: FailureKind
    retry_after_seconds: Optional[float] = None

    @property
    def is_retryable(self) -> bool:
        return self.kind in (FailureKind.TRANSIENT, FailureKind.TIMEOUT)
