# © Artur Czarnecki. All rights reserved.

"""Prompt injection defense profile contracts (Phase V-SEC.1)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class PromptRiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PromptInjectionRule(BaseModel):
    rule_id: str
    pattern: str
    risk_level: PromptRiskLevel
    block: bool


class PromptInspectionResult(BaseModel):
    blocked: bool
    risk_level: PromptRiskLevel
    matched_rule_ids: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class PromptDefenseProfile(BaseModel):
    profile_id: str
    version: str
    rules: list[PromptInjectionRule] = Field(default_factory=list)


def inspect_prompt_for_injection(*, prompt: str, profile: PromptDefenseProfile) -> PromptInspectionResult:
    matched_rules: list[PromptInjectionRule] = [
        rule for rule in profile.rules if rule.pattern.lower() in prompt.lower()
    ]
    if not matched_rules:
        return PromptInspectionResult(
            blocked=False,
            risk_level=PromptRiskLevel.LOW,
            reasons=["No prompt injection patterns matched"],
        )

    highest_risk = _max_risk_level([rule.risk_level for rule in matched_rules])
    blocked = any(rule.block for rule in matched_rules)
    reasons = [f"Matched injection pattern: {rule.rule_id}" for rule in matched_rules]
    return PromptInspectionResult(
        blocked=blocked,
        risk_level=highest_risk,
        matched_rule_ids=[rule.rule_id for rule in matched_rules],
        reasons=reasons,
    )


def _max_risk_level(levels: list[PromptRiskLevel]) -> PromptRiskLevel:
    order: dict[PromptRiskLevel, int] = {
        PromptRiskLevel.LOW: 0,
        PromptRiskLevel.MEDIUM: 1,
        PromptRiskLevel.HIGH: 2,
        PromptRiskLevel.CRITICAL: 3,
    }
    return max(levels, key=lambda level: order[level])
