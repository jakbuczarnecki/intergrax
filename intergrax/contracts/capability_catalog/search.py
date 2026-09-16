# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Search query and evidence contracts (CAPABILITY-CATALOG-1 / ME-5)."""

from __future__ import annotations

from enum import StrEnum
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.contracts.capability_catalog._validation import require_non_empty_text

SCHEMA_CAPABILITY_SEARCH_QUERY_V1: Final = "capability_search_query.v1"
SCHEMA_CAPABILITY_SEARCH_CONTEXT_V1: Final = "capability_search_context.v1"
SCHEMA_CAPABILITY_SEARCH_EVIDENCE_V1: Final = "capability_search_evidence.v1"

_NON_EMPTY = Field(min_length=1)


class CapabilitySearchSignal(StrEnum):
    """Typed search basis — evidence only, not eligibility or authority."""

    PASS_THROUGH = "pass_through"
    TEXT_SUBSTRING_MATCH = "text_substring_match"


class CapabilitySearchQuery(BaseModel):
    """Read-only search request over a discovery candidate corpus."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["capability_search_query.v1"] = SCHEMA_CAPABILITY_SEARCH_QUERY_V1
    text: str | None = None


class CapabilitySearchContext(BaseModel):
    """Optional facts for search strategies — no permissions or mutable handles."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["capability_search_context.v1"] = (
        SCHEMA_CAPABILITY_SEARCH_CONTEXT_V1
    )


class CapabilitySearchEvidence(BaseModel):
    """Immutable search metadata attached to a discovery candidate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["capability_search_evidence.v1"] = (
        SCHEMA_CAPABILITY_SEARCH_EVIDENCE_V1
    )
    search_strategy_id: str = _NON_EMPTY
    signal: CapabilitySearchSignal
    matched_field: str | None = None
    score: float | None = None

    @field_validator("search_strategy_id")
    @classmethod
    def _validate_search_strategy_id(cls, value: str) -> str:
        return require_non_empty_text(value, label="search_strategy_id")
