# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Experiment registry contracts (architecture §35)."""

from __future__ import annotations

from enum import Enum
from typing import List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class ExperimentDecision(str, Enum):
    """Laboratory verdict after observing traces and outputs (§35 step 10)."""

    PENDING = "pending"
    KEEP = "keep"
    IMPROVE = "improve"
    PAUSE = "pause"
    DELETE = "delete"


class RegisterExperimentRequest(BaseModel):
    hypothesis: str = Field(min_length=1)
    capability: str = Field(min_length=1)
    agent_id: Optional[str] = None
    expected_output: str = ""
    validation_criteria: str = ""
    notes: str = ""


class ExperimentRecord(BaseModel):
    experiment_id: str
    hypothesis: str
    capability: str
    agent_id: Optional[str] = None
    expected_output: str = ""
    validation_criteria: str = ""
    decision: ExperimentDecision = ExperimentDecision.PENDING
    notes: str = ""
    run_ids: List[str] = Field(default_factory=list)
    created_at_utc: str
    updated_at_utc: str

    @classmethod
    def new_from_request(
        cls,
        request: RegisterExperimentRequest,
        *,
        experiment_id: Optional[str] = None,
        created_at_utc: str,
        updated_at_utc: str,
    ) -> ExperimentRecord:
        return cls(
            experiment_id=experiment_id or uuid4().hex,
            hypothesis=request.hypothesis.strip(),
            capability=request.capability.strip(),
            agent_id=request.agent_id,
            expected_output=request.expected_output.strip(),
            validation_criteria=request.validation_criteria.strip(),
            notes=request.notes.strip(),
            created_at_utc=created_at_utc,
            updated_at_utc=updated_at_utc,
        )


class SetExperimentDecisionRequest(BaseModel):
    decision: ExperimentDecision
    notes: Optional[str] = None
