# © Artur Czarnecki. All rights reserved.

"""Structured output for Problem Radar (architecture canon §36)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ProblemCluster(BaseModel):
    """One clustered pain signal group."""

    cluster_id: str
    title: str
    representative_quotes: list[str] = Field(default_factory=list)
    source_links: list[str] = Field(default_factory=list)
    frequency_estimate: float = Field(ge=0.0, le=1.0)
    intensity_score: float = Field(ge=0.0, le=1.0)
    affected_user_group: str = ""
    possible_product_ideas: list[str] = Field(default_factory=list)
    mom_test_risk_notes: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class ProblemRadarOutput(BaseModel):
    """Agent response payload — serialized to the Nexus answer string."""

    clusters: list[ProblemCluster] = Field(default_factory=list)
    summary: str = ""
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
