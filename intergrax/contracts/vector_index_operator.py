# © Artur Czarnecki. All rights reserved.

"""Operator contracts for governed vector index prepare (GR-12-A4-R2-R1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.control_plane_mutation import (
    ControlPlaneMutationAuthorizationEvidence,
)
from intergrax.integrations.contracts.vector_index_administration import (
    VectorIndexPrepareOutcome,
    VectorIndexSpec,
)

SCHEMA_VECTOR_INDEX_PREPARE_OPERATOR_REQUEST_V1 = (
    "vector_index_prepare_operator_request.v1"
)
SCHEMA_VECTOR_INDEX_PREPARE_OPERATOR_RESULT_V1 = "vector_index_prepare_operator_result.v1"


@dataclass(frozen=True, slots=True)
class VectorIndexPrepareOperatorRequest:
    """Live operator prepare invocation (typed spec; not a loose payload)."""

    mutation_id: str
    spec: VectorIndexSpec


class VectorIndexPrepareOperatorResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["vector_index_prepare_operator_result.v1"] = (
        SCHEMA_VECTOR_INDEX_PREPARE_OPERATOR_RESULT_V1
    )
    mutation_id: str = Field(min_length=1)
    changed: bool
    outcome: VectorIndexPrepareOutcome | None = None
    before_revision: str = Field(min_length=1)
    after_revision: str = Field(min_length=1)
    authorization_evidence: ControlPlaneMutationAuthorizationEvidence | None = None
    blocker_code: str | None = None
    policy_action: str | None = None
