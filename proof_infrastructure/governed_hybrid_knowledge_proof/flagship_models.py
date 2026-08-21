# © Artur Czarnecki. All rights reserved.

"""Typed presentation models for COMM-5 F3-F flagship proof output."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from intergrax.runtime.evidence.obligation_derivation_contracts import (
    PolicyEvidenceBasisV1,
    TemporalConstraintV1,
)
from local_workspace_application.workspaces.hybrid_ask_models import (
    EvidenceAdmissibilityStatusV1,
    RequirementAdmissibilityReasonCodeV1,
    RequirementEvaluationStatusV1,
)


class FlagshipScenarioIdV1(StrEnum):
    REV17_ALL_SATISFIED = "rev17_all_satisfied"
    REV18_STALE_SECURITY = "rev18_stale_security"
    REV18_FRESH_SECURITY = "rev18_fresh_security"
    AUTHORITY_REVOKED = "authority_revoked"
    PROVIDER_503 = "provider_503"
    MALFORMED_RESPONSE = "malformed_response"
    VENDOR_RESTART = "vendor_restart"
    STRUCTURAL_HISTORY = "structural_history"


class FlagshipRequirementProofV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    requirement_id: str = Field(..., min_length=1, max_length=256)
    source_connection_ref: str | None = Field(default=None, max_length=128)
    capability_id: str | None = Field(default=None, max_length=128)
    call_id: str | None = Field(default=None, max_length=256)
    policy_document_id: str | None = Field(default=None, max_length=128)
    policy_revision_id: str | None = Field(default=None, max_length=128)
    policy_rule_id: str | None = Field(default=None, max_length=128)
    temporal_constraint: TemporalConstraintV1 | None = None
    temporal_effective_at: str | None = Field(default=None, max_length=64)
    temporal_evaluated_at: str | None = Field(default=None, max_length=64)
    outcome: RequirementEvaluationStatusV1 | None = None
    reason: RequirementAdmissibilityReasonCodeV1 | None = None


class FlagshipScenarioProofV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: FlagshipScenarioIdV1
    policy_basis: PolicyEvidenceBasisV1 | None = None
    derivation_snapshot_id: str | None = Field(default=None, max_length=256)
    requirements: tuple[FlagshipRequirementProofV1, ...] = ()
    overall_admissibility: EvidenceAdmissibilityStatusV1 | None = None
    llm_calls: int = Field(..., ge=0)
    answer: str | None = Field(default=None, max_length=4096)
    run_id: str = Field(..., min_length=1, max_length=128)
    passed: bool
    detail: str = Field(..., min_length=1, max_length=1024)


class FlagshipSummaryRowV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_label: str = Field(..., min_length=1, max_length=128)
    evidence: str = Field(..., min_length=1, max_length=32)
    authority: str = Field(..., min_length=1, max_length=32)
    temporal: str = Field(..., min_length=1, max_length=32)
    result: str = Field(..., min_length=1, max_length=32)
    llm: int = Field(..., ge=0)


class AdvancedFlagshipProofResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    flagship_question: str = Field(..., min_length=1, max_length=512)
    scenarios: tuple[FlagshipScenarioProofV1, ...] = ()
    summary_rows: tuple[FlagshipSummaryRowV1, ...] = ()
    distinct_providers: int = Field(..., ge=0)
    distinct_connections: int = Field(..., ge=0)
    distinct_capabilities: int = Field(..., ge=0)
    distinct_call_ids: int = Field(..., ge=0)
    all_passed: bool
    history_comparison: str = Field(default="", max_length=4096)
