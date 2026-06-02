# © Artur Czarnecki. All rights reserved.

"""Prompt registry governance contracts (Phase V-PE.1)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class PromptRiskTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PromptRegistryEntry(BaseModel):
    prompt_id: str
    version: str
    owner_team: str
    owner_contact: str
    risk_tier: PromptRiskTier
    change_ticket_ref: str = ""


class PromptRegistryValidationResult(BaseModel):
    prompt_id: str
    valid: bool
    reasons: list[str] = Field(default_factory=list)


class PromptRegistryGovernanceReport(BaseModel):
    schema_version: str = "1.0.0"
    entries: list[PromptRegistryEntry] = Field(default_factory=list)
    validations: list[PromptRegistryValidationResult] = Field(default_factory=list)


def validate_prompt_registry_entry(entry: PromptRegistryEntry) -> PromptRegistryValidationResult:
    reasons: list[str] = []
    if not entry.owner_team:
        reasons.append("Missing owner team")
    if not entry.owner_contact:
        reasons.append("Missing owner contact")
    if not entry.version:
        reasons.append("Missing prompt version")
    if entry.risk_tier in {PromptRiskTier.HIGH} and not entry.change_ticket_ref:
        reasons.append("High-risk prompt requires change ticket reference")
    return PromptRegistryValidationResult(
        prompt_id=entry.prompt_id,
        valid=not reasons,
        reasons=reasons,
    )


def evaluate_prompt_registry(entries: list[PromptRegistryEntry]) -> PromptRegistryGovernanceReport:
    validations = [validate_prompt_registry_entry(entry) for entry in entries]
    return PromptRegistryGovernanceReport(entries=entries, validations=validations)
