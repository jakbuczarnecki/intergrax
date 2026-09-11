# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""External operation provider SPI — execution only after admission (R1)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from intergrax.contracts.external_operations.attempt import ExternalOperationAttempt
from intergrax.contracts.external_operations.evidence import ProviderExecutionOutcome


class ProviderRiskProfile(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    PRODUCTION = "PRODUCTION"


@dataclass(frozen=True, slots=True)
class ProviderPayloadBounds:
    max_payload_bytes: int
    timeout_seconds: float
    max_retries: int

    def __post_init__(self) -> None:
        if self.max_payload_bytes < 1:
            raise ValueError("max_payload_bytes must be >= 1")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")


@runtime_checkable
class ExternalOperationProvider(Protocol):
    """Provider adapter — cannot bypass admission or own audit stores."""

    @property
    def provider_id(self) -> str:
        ...

    @property
    def version(self) -> str:
        ...

    @property
    def capabilities(self) -> frozenset[str]:
        ...

    @property
    def tenant_scope(self) -> frozenset[str] | None:
        """None means tenant resolved at admission time only."""

    @property
    def risk_profile(self) -> ProviderRiskProfile:
        ...

    @property
    def payload_bounds(self) -> ProviderPayloadBounds:
        ...

    def execute_admitted(
        self,
        attempt: ExternalOperationAttempt,
    ) -> ProviderExecutionOutcome:
        """Run only for ADMITTED → EXECUTING attempts; no hidden retries."""
        ...
