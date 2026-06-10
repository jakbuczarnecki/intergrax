# © Artur Czarnecki. All rights reserved.

"""Structured output for Vendor Discovery (Phase K.2 prototype)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class VendorCandidate(BaseModel):
    vendor_id: str
    name: str
    category: str = ""
    fit_score: float = Field(ge=0.0, le=1.0)
    risk_notes: list[str] = Field(default_factory=list)
    source_links: list[str] = Field(default_factory=list)


class VendorDiscoveryOutput(BaseModel):
    candidates: list[VendorCandidate] = Field(default_factory=list)
    summary: str = ""
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
