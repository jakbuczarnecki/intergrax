# © Artur Czarnecki. All rights reserved.

"""Pluggable enterprise adaptation contracts (DS-E2E-15J-L13)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from testing_support.decision_e2e.model_matrix.autonomous_enterprise_adaptation.contracts import (
    AdaptationAuditMetadata,
    AdaptationExecutionStatus,
    ApprovedAdaptationRequest,
)


@dataclass(frozen=True, slots=True)
class AdaptationApplyOutcome:
    status: AdaptationExecutionStatus
    applied_change_reference: str | None
    summary: str


class EnterpriseAdaptationProvider(Protocol):
    """Pluggable adaptation mechanism — new strategy = new provider, no engine edits."""

    @property
    def provider_id(self) -> str: ...

    @property
    def provider_version(self) -> str: ...

    def apply_adaptation(
        self, approved_change: ApprovedAdaptationRequest
    ) -> AdaptationApplyOutcome: ...


class AdaptationAuditProvider(Protocol):
    """Records auditable trace for every adaptation attempt."""

    @property
    def provider_id(self) -> str: ...

    @property
    def provider_version(self) -> str: ...

    def build_audit(
        self,
        approved_change: ApprovedAdaptationRequest,
        *,
        outcome_status: AdaptationExecutionStatus,
        adaptation_provider_id: str,
        adaptation_provider_version: str,
        applied_change_reference: str | None,
        executed_at: datetime,
        outcome_summary: str,
    ) -> AdaptationAuditMetadata: ...


__all__ = [
    "AdaptationApplyOutcome",
    "AdaptationAuditProvider",
    "EnterpriseAdaptationProvider",
]
