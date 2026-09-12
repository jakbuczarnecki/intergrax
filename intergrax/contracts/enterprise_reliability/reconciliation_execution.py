# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Reconciliation probe execution contracts (ERL Phase 3 — execution foundation)."""

from __future__ import annotations

from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.enterprise_reliability.evidence import ExternalEffectEvidenceVerdict
from intergrax.contracts.enterprise_reliability.reconciliation import (
    ReconciliationDisposition,
    ReconciliationPlan,
)

SCHEMA_RECONCILIATION_PROBE_REQUEST_V1: Final = "reconciliation_probe_request.v1"
SCHEMA_RECONCILIATION_PROBE_RESULT_V1: Final = "reconciliation_probe_result.v1"


class ReconciliationExecutionDisposition(StrEnum):
    """Platform outcome after reconcile execution orchestration — no provider I/O here."""

    PROBE_EXECUTED = "probe_executed"
    SKIPPED_NOT_SCHEDULED = "skipped_not_scheduled"
    PLUGIN_PROBE_UNAVAILABLE = "plugin_probe_unavailable"


class ReconciliationExecutionError(ValueError):
    """Probe execution cannot proceed under contract and plan rules."""


class ReconciliationProbeRequest(BaseModel):
    """Provider-neutral probe invocation — integrations interpret ``probe_ref``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_RECONCILIATION_PROBE_REQUEST_V1
    tenant_id: str = Field(min_length=1, max_length=256)
    correlation_id: str = Field(min_length=1, max_length=256)
    contract_id: str = Field(min_length=1, max_length=256)
    probe_ref: str = Field(min_length=1, max_length=256)
    plugin_id: str = Field(min_length=1, max_length=256)
    attempt_index: int = Field(default=1, ge=1, le=256)


class ReconciliationProbeResult(BaseModel):
    """Evidence returned by a plugin probe — core stores refs and verdict only."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_RECONCILIATION_PROBE_RESULT_V1
    verdict: ExternalEffectEvidenceVerdict
    evidence_ref: str = Field(min_length=1, max_length=512)
    rationale: str = Field(default="", max_length=512)


class ExternalEffectReconciliationExecution(BaseModel):
    """One bounded reconcile attempt — probe result and updated uncertainty state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    disposition: ReconciliationExecutionDisposition
    plan: ReconciliationPlan
    probe_request: ReconciliationProbeRequest | None = None
    probe_result: ReconciliationProbeResult | None = None
    rationale: str = Field(default="", max_length=512)


def build_reconciliation_probe_request(
    *,
    plan: ReconciliationPlan,
    tenant_id: str,
    correlation_id: str,
    contract_id: str,
    attempt_index: int = 1,
) -> ReconciliationProbeRequest:
    """Materialize a probe request from a scheduled reconcile plan."""
    if plan.disposition is not ReconciliationDisposition.SCHEDULE_PROBE:
        raise ReconciliationExecutionError(
            "probe request requires schedule_probe disposition",
        )
    assert plan.probe_ref is not None and plan.plugin_id is not None
    return ReconciliationProbeRequest(
        tenant_id=tenant_id.strip(),
        correlation_id=correlation_id.strip(),
        contract_id=contract_id.strip(),
        probe_ref=plan.probe_ref,
        plugin_id=plan.plugin_id,
        attempt_index=attempt_index,
    )


__all__ = [
    "ExternalEffectReconciliationExecution",
    "ReconciliationExecutionDisposition",
    "ReconciliationExecutionError",
    "ReconciliationProbeRequest",
    "ReconciliationProbeResult",
    "SCHEMA_RECONCILIATION_PROBE_REQUEST_V1",
    "SCHEMA_RECONCILIATION_PROBE_RESULT_V1",
    "build_reconciliation_probe_request",
]
