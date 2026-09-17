# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Typed optional source failures for federated runtime inspection."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class RuntimeInspectionSourceFailureCode(StrEnum):
    UNAVAILABLE = "unavailable"
    INTEGRITY = "integrity"
    REDACTED = "redacted"


class RuntimeInspectionSourceFailure(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    code: RuntimeInspectionSourceFailureCode
    reason_code: str = Field(min_length=1)


__all__ = [
    "RuntimeInspectionSourceFailure",
    "RuntimeInspectionSourceFailureCode",
]
