# © Artur Czarnecki. All rights reserved.

"""Prompt governance validation for YAML catalog entries (Phase V-REM-PE.1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.prompts.schema.prompt_governance import PromptRiskTier
from intergrax.prompts.schema.prompt_schema import LocalizedPromptDocument


@dataclass(frozen=True, slots=True)
class PromptGovernanceValidationResult:
    prompt_id: str
    valid: bool
    reasons: tuple[str, ...] = ()


def validate_prompt_document_governance(
    document: LocalizedPromptDocument,
) -> PromptGovernanceValidationResult:
    """Validate governance fields carried on ``PromptMeta``."""
    meta = document.meta
    reasons: list[str] = []
    if not meta.owner_team:
        reasons.append("Missing owner team")
    if not meta.owner_contact:
        reasons.append("Missing owner contact")
    if meta.risk_tier == PromptRiskTier.HIGH and not meta.change_ticket_ref:
        reasons.append("High-risk prompt requires change ticket reference")
    return PromptGovernanceValidationResult(
        prompt_id=document.id,
        valid=not reasons,
        reasons=tuple(reasons),
    )
