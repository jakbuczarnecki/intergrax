# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Work-stage capability need contracts (CAPABILITY-CATALOG-1 Stage 8)."""

from __future__ import annotations

from typing import Final, Literal, NewType

from pydantic import BaseModel, ConfigDict, model_validator

from intergrax.contracts.capability_catalog._validation import require_non_empty_text
from intergrax.contracts.capability_catalog.query import CapabilityDiscoveryQuery

SCHEMA_WORK_STAGE_CAPABILITY_NEED_V1: Final = "work_stage_capability_need.v1"

WorkContextReference = NewType("WorkContextReference", str)
WorkStageReference = NewType("WorkStageReference", str)


def validate_work_context_reference(value: object) -> WorkContextReference:
    return WorkContextReference(require_non_empty_text(value, label="work_reference"))


def validate_work_stage_reference(value: object) -> WorkStageReference:
    return WorkStageReference(require_non_empty_text(value, label="stage_reference"))


class WorkStageCapabilityNeed(BaseModel):
    """Read-only stage-scoped capability need — not a lifecycle aggregate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["work_stage_capability_need.v1"] = (
        SCHEMA_WORK_STAGE_CAPABILITY_NEED_V1
    )
    work_reference: str
    stage_reference: str
    goal_objective: str
    stage_objective: str
    discovery_query: CapabilityDiscoveryQuery | None = None

    @model_validator(mode="after")
    def _validate_references_and_objectives(self) -> WorkStageCapabilityNeed:
        validate_work_context_reference(self.work_reference)
        validate_work_stage_reference(self.stage_reference)
        require_non_empty_text(self.goal_objective, label="goal_objective")
        require_non_empty_text(self.stage_objective, label="stage_objective")
        return self

    @property
    def requests_capabilities(self) -> bool:
        return self.discovery_query is not None
