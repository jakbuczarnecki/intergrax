# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Recommendation evidence and context contracts (CAPABILITY-CATALOG-1 / ME-5)."""

from __future__ import annotations

from enum import StrEnum
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.contracts.capability_catalog._validation import require_non_empty_text

SCHEMA_CAPABILITY_RECOMMENDATION_CONTEXT_V1: Final = "capability_recommendation_context.v1"
SCHEMA_CAPABILITY_RECOMMENDATION_EVIDENCE_V1: Final = "capability_recommendation_evidence.v1"

_NON_EMPTY = Field(min_length=1)


class CapabilityRecommendationReasonCode(StrEnum):
    """Advisory recommendation basis — not selection or authorization."""

    TOP_RANKED = "top_ranked"


class CapabilityRecommendationContext(BaseModel):
    """Read-only recommendation configuration — advisory output only."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["capability_recommendation_context.v1"] = (
        SCHEMA_CAPABILITY_RECOMMENDATION_CONTEXT_V1
    )
    top_n: int = Field(default=10, ge=1)


class CapabilityRecommendationEvidence(BaseModel):
    """Immutable recommendation metadata — not lifecycle or governance authority."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["capability_recommendation_evidence.v1"] = (
        SCHEMA_CAPABILITY_RECOMMENDATION_EVIDENCE_V1
    )
    recommendation_strategy_id: str = _NON_EMPTY
    reason_codes: tuple[CapabilityRecommendationReasonCode, ...] = Field(min_length=1)
    reason_text: str | None = None
    rank_position: int | None = Field(default=None, ge=1)

    @field_validator("recommendation_strategy_id")
    @classmethod
    def _validate_recommendation_strategy_id(cls, value: str) -> str:
        return require_non_empty_text(value, label="recommendation_strategy_id")
