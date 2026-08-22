# © Artur Czarnecki. All rights reserved.
# Intergrax platform proof — TOOLS-ITERATIVE-SQL-INVESTIGATION (PP-3C).

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from intergrax.runtime.nexus.tools.investigation_proof import InvestigationProof
    from intergrax.runtime.nexus.engine.runtime_state import ToolCallTrace

from platform_proofs.tools.iterative_sql_investigation.dataset_identity import (
    DatasetFingerprint,
    DatasetIdentity,
)
from platform_proofs.tools.iterative_sql_investigation.scenarios import ScenarioId


class ScenarioAOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    identifies_north_anomalous_segment: bool
    rejects_volume_only_explanation: bool
    conclusion_supported: bool


class ScenarioBOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    detects_global_association: bool
    verifies_segmented_evidence: bool
    claims_direct_causation: bool = Field(
        description="Must be false for PASS — model must not assert direct causation."
    )


class ScenarioCOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    staffing_evidence_available: bool = Field(
        description="Must be false for PASS — dataset has no staffing evidence."
    )
    claims_staffing_cause: bool = Field(
        description="Must be false for PASS."
    )
    reports_missing_evidence: bool = Field(
        description="Must be true for PASS."
    )


class ScenarioRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: ScenarioId
    passed: bool
    stop_reason: str
    successful_tool_calls: int
    investigation_proof_steps: int
    platform_invariants_pass: bool
    outcome_a: ScenarioAOutcome | None = None
    outcome_b: ScenarioBOutcome | None = None
    outcome_c: ScenarioCOutcome | None = None
    failure_reasons: tuple[str, ...] = ()


class ModelProviderIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str
    model: str
    supports_native_tools: bool


class ToolsSqlInvestigationProofResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    proof_id: str
    dataset_identity: dict[str, Any]
    dataset_fingerprint_sha256: str
    db_verification_stats: dict[str, Any]
    model_provider: ModelProviderIdentity
    scenarios: tuple[ScenarioRunResult, ...]
    overall_pass: bool
    blocked_reason: str | None = None

    @classmethod
    def blocked(
        cls,
        *,
        proof_id: str,
        identity: DatasetIdentity,
        fingerprint: DatasetFingerprint,
        reason: str,
    ) -> ToolsSqlInvestigationProofResult:
        return cls(
            proof_id=proof_id,
            dataset_identity=identity.as_dict(),
            dataset_fingerprint_sha256=fingerprint.sha256,
            db_verification_stats={},
            model_provider=ModelProviderIdentity(
                provider="unknown",
                model="unknown",
                supports_native_tools=False,
            ),
            scenarios=(),
            overall_pass=False,
            blocked_reason=reason,
        )


@dataclass(frozen=True, slots=True)
class ScenarioExecutionSnapshot:
    stop_reason: str
    successful_tool_calls: int
    sql_texts: tuple[str, ...]
    output_texts: tuple[str, ...]
    investigation_proof_steps: int
    follow_up_has_valid_basis: bool
    final_answer: str
    tool_traces: tuple[ToolCallTrace, ...] = ()
    investigation_proof: InvestigationProof | None = None
