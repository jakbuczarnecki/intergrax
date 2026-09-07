# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Work-stage capability discovery loop evidence (CAPABILITY-CATALOG-1 Stage 14)."""

from __future__ import annotations

from enum import StrEnum
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from intergrax.contracts.capability_catalog._validation import require_non_empty_text
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.capability_catalog.work_stage import WorkStageCapabilityNeed

SCHEMA_WORK_STAGE_CAPABILITY_OBSERVATION_V1: Final = "work_stage_capability_observation.v1"
SCHEMA_WORK_STAGE_CAPABILITY_LOOP_ITERATION_EVIDENCE_V1: Final = (
    "work_stage_capability_loop_iteration_evidence.v1"
)
SCHEMA_WORK_STAGE_CAPABILITY_LOOP_RESULT_V1: Final = "work_stage_capability_loop_result.v1"
_NON_EMPTY = Field(min_length=1)


class WorkStageCapabilityLoopDisposition(StrEnum):
    """Terminal disposition for a bounded reference discovery loop."""

    COMPLETED = "COMPLETED"
    BLOCKED = "BLOCKED"
    ESCALATED = "ESCALATED"
    UNAVAILABLE = "UNAVAILABLE"
    CONFLICT = "CONFLICT"


class WorkStageDomainAuthorityKind(StrEnum):
    """Domain authority route used after governed selection — not execution semantics."""

    TOOL = "TOOL"
    AGENT = "AGENT"
    SKILL = "SKILL"


class WorkStageCapabilityExecutionCorrelation(BaseModel):
    """Correlation fields from canonical Tool execution authority — not audit evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["work_stage_capability_execution_correlation.v1"] = (
        "work_stage_capability_execution_correlation.v1"
    )
    run_id: str = _NON_EMPTY
    step_id: str = _NON_EMPTY
    tool_id: str = _NON_EMPTY

    @model_validator(mode="after")
    def _validate_correlation(self) -> WorkStageCapabilityExecutionCorrelation:
        require_non_empty_text(self.run_id, label="run_id")
        require_non_empty_text(self.step_id, label="step_id")
        require_non_empty_text(self.tool_id, label="tool_id")
        return self


class WorkStageCapabilityObservation(BaseModel):
    """Typed execution observation — may carry the next stage need for rediscovery."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["work_stage_capability_observation.v1"] = (
        SCHEMA_WORK_STAGE_CAPABILITY_OBSERVATION_V1
    )
    execution_succeeded: bool
    outcome_summary: str = _NON_EMPTY
    next_need: WorkStageCapabilityNeed | None = None

    @model_validator(mode="after")
    def _validate_outcome_summary(self) -> WorkStageCapabilityObservation:
        require_non_empty_text(self.outcome_summary, label="outcome_summary")
        return self


class WorkStageCapabilityLoopIterationEvidence(BaseModel):
    """Immutable evidence for one governed discovery → execute → observe iteration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["work_stage_capability_loop_iteration_evidence.v1"] = (
        SCHEMA_WORK_STAGE_CAPABILITY_LOOP_ITERATION_EVIDENCE_V1
    )
    iteration_index: int = Field(ge=0)
    need: WorkStageCapabilityNeed
    discovery_evidence_schema: str = _NON_EMPTY
    selected_identity_key: CapabilityIdentityKey | None = None
    domain_authority_kind: WorkStageDomainAuthorityKind | None = None
    execution_correlation: WorkStageCapabilityExecutionCorrelation | None = None
    observation: WorkStageCapabilityObservation | None = None

    @model_validator(mode="after")
    def _validate_iteration_evidence(self) -> WorkStageCapabilityLoopIterationEvidence:
        require_non_empty_text(
            self.discovery_evidence_schema,
            label="discovery_evidence_schema",
        )
        if self.selected_identity_key is not None:
            if self.domain_authority_kind is None:
                raise ValueError(
                    "domain_authority_kind is required when selected_identity_key is set",
                )
        if self.execution_correlation is not None:
            if self.selected_identity_key is None:
                raise ValueError(
                    "execution_correlation requires selected_identity_key",
                )
            if self.domain_authority_kind is None:
                raise ValueError(
                    "execution_correlation requires domain_authority_kind",
                )
            if self.observation is None:
                raise ValueError("execution_correlation requires observation")
            if (
                self.domain_authority_kind is WorkStageDomainAuthorityKind.TOOL
                and self.execution_correlation.tool_id != self.selected_identity_key.logical_id
            ):
                raise ValueError(
                    "execution_correlation.tool_id must match selected_identity_key.logical_id",
                )
        return self


class WorkStageCapabilityLoopResult(BaseModel):
    """Ordered immutable loop evidence with explicit terminal disposition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["work_stage_capability_loop_result.v1"] = (
        SCHEMA_WORK_STAGE_CAPABILITY_LOOP_RESULT_V1
    )
    disposition: WorkStageCapabilityLoopDisposition
    iterations: tuple[WorkStageCapabilityLoopIterationEvidence, ...]

    @model_validator(mode="after")
    def _validate_iteration_order(self) -> WorkStageCapabilityLoopResult:
        for index, iteration in enumerate(self.iterations):
            if iteration.iteration_index != index:
                raise ValueError(
                    "iterations must be ordered by contiguous iteration_index starting at 0",
                )
        return self
