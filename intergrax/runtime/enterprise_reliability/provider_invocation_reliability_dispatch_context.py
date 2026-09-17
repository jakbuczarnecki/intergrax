# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Internal dispatch-time bundle for GR-7-A8 early lifecycle evidence (projection only)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.enterprise_reliability.provider_invocation_reliability_evidence import (
    ProviderInvocationReliabilityEvidenceObserver,
)


@dataclass(frozen=True, slots=True)
class ProviderInvocationReliabilityDispatchContext:
    """Optional dispatch-time correlation for early lifecycle evidence emission."""

    tenant_id: str
    effect_contract_id: str | None
    execution_id: str | None
    attempt_id: str | None
    observer: ProviderInvocationReliabilityEvidenceObserver | None


__all__ = ["ProviderInvocationReliabilityDispatchContext"]
