# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PagerDutyTriggerIncidentInput(BaseModel):
    summary: str = Field(..., min_length=1)
    severity: str = Field(default="error")
    source: str = Field(default="intergrax")
    dedup_key: str = ""
    custom_details: dict[str, Any] = Field(default_factory=dict)


class PagerDutyTriggerIncidentOutput(BaseModel):
    dedup_key: str
    triggered: bool = True
