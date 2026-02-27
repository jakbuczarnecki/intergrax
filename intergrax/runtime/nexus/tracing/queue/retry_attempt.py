# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload


@dataclass(frozen=True)
class RetryAttemptDiagnosticPayload(DiagnosticPayload):
    """
    Emitted when execution plane schedules a retry.

    This payload is infrastructure-level and PII-safe.
    """

    logical_task_name: str
    exception_type: str
    current_retries: int
    max_retries: int
    countdown_seconds: float
    reason: str  # "lock_conflict" | "handler_transient"

    @classmethod
    def schema_id(cls) -> str:
        return "intergrax.diag.queue.retry_attempt"
    
    def redact(self) -> "RetryAttemptDiagnosticPayload":
        """
        This payload does not contain PII, so redaction returns self.
        """
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "logical_task_name": self.logical_task_name,
            "exception_type": self.exception_type,
            "current_retries": self.current_retries,
            "max_retries": self.max_retries,
            "countdown_seconds": self.countdown_seconds,
            "reason": self.reason,
        }