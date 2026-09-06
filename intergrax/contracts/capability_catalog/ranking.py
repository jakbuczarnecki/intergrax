# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Ranking evidence and context contracts (CAPABILITY-CATALOG-1 Stage 4)."""

from __future__ import annotations

from enum import StrEnum
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.contracts.capability_catalog._validation import require_non_empty_text

SCHEMA_CAPABILITY_RANKING_EVIDENCE_V1: Final = "capability_ranking_evidence.v1"
SCHEMA_CAPABILITY_RANKING_CONTEXT_V1: Final = "capability_ranking_context.v1"

_NON_EMPTY = Field(min_length=1)


class CapabilityRankingSignal(StrEnum):
    """Typed ranking basis — evidence only, not authority."""

    STABLE_IDENTITY_ORDER = "stable_identity_order"
    KEYWORD_OVERLAP = "keyword_overlap"


class CapabilityRankingEvidence(BaseModel):
    """Immutable ranking metadata attached to a discovery candidate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["capability_ranking_evidence.v1"] = (
        SCHEMA_CAPABILITY_RANKING_EVIDENCE_V1
    )
    ranker_id: str = _NON_EMPTY
    rank_position: int = Field(ge=1)
    signal: CapabilityRankingSignal
    score: float | None = None
    original_stage3_position: int | None = Field(default=None, ge=1)

    @field_validator("ranker_id")
    @classmethod
    def _validate_ranker_id(cls, value: str) -> str:
        return require_non_empty_text(value, label="ranker_id")


class CapabilityRankingContext(BaseModel):
    """Read-only facts for domain rankers — no permissions or mutable handles."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["capability_ranking_context.v1"] = (
        SCHEMA_CAPABILITY_RANKING_CONTEXT_V1
    )
    semantic_need: str | None = None
