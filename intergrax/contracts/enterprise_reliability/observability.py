# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Semantic facts for Observability spine emission (ERL Phase 1 — contracts only)."""

from __future__ import annotations

from datetime import datetime
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.enterprise_reliability.gating import DependentExecutionGateAction
from intergrax.contracts.enterprise_reliability.lifecycle import (
    UncertaintyLifecyclePhase,
    UncertaintyResolutionKind,
)
from intergrax.contracts.enterprise_reliability.outcome import ExternalEffectOutcome
from intergrax.contracts.enterprise_reliability.evidence import ExternalEffectEvidenceVerdict
from intergrax.contracts.enterprise_reliability.reconciliation import ReconciliationDisposition
from intergrax.contracts.enterprise_reliability.resolution_decision import ResolutionPlatformAction

SCHEMA_UNCERTAINTY_ADMISSION_FACT_V1: Final = "uncertainty_admission_fact.v1"
SCHEMA_UNCERTAINTY_GATING_FACT_V1: Final = "uncertainty_gating_fact.v1"
SCHEMA_RECONCILIATION_PLAN_FACT_V1: Final = "reconciliation_plan_fact.v1"
SCHEMA_RECONCILIATION_ATTEMPT_FACT_V1: Final = "reconciliation_attempt_fact.v1"
SCHEMA_RESOLUTION_DECISION_FACT_V1: Final = "resolution_decision_fact.v1"


class UncertaintyAdmissionFact(BaseModel):
    """Record that execution admitted UNKNOWN rather than guessing."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    correlation_id: str = Field(min_length=1, max_length=256)
    effect_outcome: ExternalEffectOutcome
    lifecycle_phase: UncertaintyLifecyclePhase
    recorded_at: datetime


class ReconciliationPlanFact(BaseModel):
    """Record reconcile planning — attempts execute in later orchestration phases."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    correlation_id: str = Field(min_length=1, max_length=256)
    contract_id: str = Field(min_length=1, max_length=256)
    disposition: ReconciliationDisposition
    probe_ref: str | None = Field(default=None, max_length=256)
    plugin_id: str | None = Field(default=None, max_length=256)
    lifecycle_phase: UncertaintyLifecyclePhase
    recorded_at: datetime


class ReconciliationAttemptFact(BaseModel):
    """Record one reconcile probe execution and evidence verdict."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    correlation_id: str = Field(min_length=1, max_length=256)
    contract_id: str = Field(min_length=1, max_length=256)
    probe_ref: str = Field(min_length=1, max_length=256)
    plugin_id: str = Field(min_length=1, max_length=256)
    attempt_index: int = Field(ge=1, le=256)
    verdict: ExternalEffectEvidenceVerdict
    evidence_ref: str = Field(min_length=1, max_length=512)
    lifecycle_phase: UncertaintyLifecyclePhase
    recorded_at: datetime


class ResolutionDecisionFact(BaseModel):
    """Record resolution strategy outcome after reconciliation evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    correlation_id: str = Field(min_length=1, max_length=256)
    contract_id: str = Field(min_length=1, max_length=256)
    plugin_id: str | None = Field(default=None, max_length=256)
    evidence_ref: str = Field(min_length=1, max_length=512)
    platform_action: ResolutionPlatformAction | None = None
    resolution_kind: UncertaintyResolutionKind
    lifecycle_phase: UncertaintyLifecyclePhase
    recorded_at: datetime


class UncertaintyGatingFact(BaseModel):
    """Record that dependents were gated while truth was uncertain."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    correlation_id: str = Field(min_length=1, max_length=256)
    gate_action: DependentExecutionGateAction
    reason: str = Field(default="", max_length=512)
    recorded_at: datetime


__all__ = [
    "SCHEMA_RECONCILIATION_ATTEMPT_FACT_V1",
    "SCHEMA_RECONCILIATION_PLAN_FACT_V1",
    "SCHEMA_RESOLUTION_DECISION_FACT_V1",
    "SCHEMA_UNCERTAINTY_ADMISSION_FACT_V1",
    "SCHEMA_UNCERTAINTY_GATING_FACT_V1",
    "ReconciliationAttemptFact",
    "ResolutionDecisionFact",
    "ReconciliationPlanFact",
    "UncertaintyAdmissionFact",
    "UncertaintyGatingFact",
]
