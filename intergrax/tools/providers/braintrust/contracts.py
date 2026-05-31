# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class BraintrustLogEvalInput(BaseModel):
    name: str = Field(..., min_length=1)
    score: float = Field(..., ge=0.0, le=1.0)
    project: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class BraintrustLogEvalOutput(BaseModel):
    log_id: str
    provider: str = "braintrust"
