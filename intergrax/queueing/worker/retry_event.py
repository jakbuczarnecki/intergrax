# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetryEvent:
    """
    Infrastructure-level retry event emitted by the execution plane.

    This model is PII-safe and does not depend on runtime or tracing layers.
    """

    logical_task_name: str
    exception_type: str
    current_retries: int
    max_retries: int
    countdown_seconds: float
    reason: str  # "lock_conflict" | "handler_transient"