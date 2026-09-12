"""External payment processing lifecycle — distinct from application knowledge."""

from __future__ import annotations

from enum import StrEnum


class ExternalPaymentLifecycleState(StrEnum):
    """States tracked inside the external acquirer / PSP simulator."""

    REQUESTED = "REQUESTED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


class IntegrationResponseState(StrEnum):
    """What the integration channel returns to the commerce application."""

    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"
