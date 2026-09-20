# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Capability qualification provider SPI (UCA-4)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.capability_qualification.qualification_request import (
    CapabilityQualificationRequest,
)
from intergrax.contracts.capability_qualification.qualification_result import (
    CapabilityQualificationResult,
)


@runtime_checkable
class CapabilityQualificationProvider(Protocol):
    """Replaceable qualification adapter — verification facts, not execution."""

    @property
    def provider_id(self) -> str:
        """Stable provider identity."""
        ...

    def supports(self, request: CapabilityQualificationRequest) -> bool:
        """Technical compatibility only — side-effect free, no trust decision."""
        ...

    def qualify(
        self, request: CapabilityQualificationRequest
    ) -> CapabilityQualificationResult:
        """Produce typed qualification facts for one request.

        Idempotency: for the same ``qualification_request_id``, the provider MUST NOT
        duplicate unsafe side effects.

        Providers MUST NOT execute workloads, mutate domain lifecycle, or mint authority.
        """
        ...


__all__ = ["CapabilityQualificationProvider"]
