# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RateLimitEvent:
    """
    Event emitted when execution is denied by distributed rate limiter.

    Designed for observability, metrics and telemetry integration.
    """

    logical_task_name: str
    tenant_id: str
    retry_after_seconds: float
    remaining_tokens: float
    current_retries: int
    max_retries: int