# © Artur Czarnecki. All rights reserved.

"""Unit tests for Token Optimization LLM router contracts (TOKEN-9)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from intergrax.runtime.token_optimization.llm_router_contracts import (
    TokenOptimizationLLMRouterPolicy,
    TokenOptimizationRouterConfigurationId,
    TokenOptimizationRouterReasonCode,
    TokenOptimizationRouterRisk,
    TokenOptimizationRouterToolInput,
)

pytestmark = pytest.mark.unit


def _valid_input(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "configuration_id": TokenOptimizationRouterConfigurationId.EXACT_ONLY,
        "reason_code": TokenOptimizationRouterReasonCode.EXACT_DUPLICATES,
        "risk": TokenOptimizationRouterRisk.LOW,
        "review_required": False,
        "confidence": 0.9,
    }
    payload.update(overrides)
    return payload


def test_extra_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        TokenOptimizationRouterToolInput.model_validate(
            {**_valid_input(), "max_chars": 80}
        )


def test_invalid_confidence_rejected() -> None:
    with pytest.raises(ValidationError):
        TokenOptimizationRouterToolInput.model_validate(
            _valid_input(confidence=1.5)
        )


def test_high_risk_without_review_rejected() -> None:
    with pytest.raises(ValidationError):
        TokenOptimizationRouterToolInput.model_validate(
            _valid_input(risk=TokenOptimizationRouterRisk.HIGH, review_required=False)
        )


def test_protected_reason_without_review_rejected() -> None:
    with pytest.raises(ValidationError):
        TokenOptimizationRouterToolInput.model_validate(
            _valid_input(
                reason_code=TokenOptimizationRouterReasonCode.PROTECTED_OR_HIGH_RISK,
                review_required=False,
            )
        )


def test_enums_serialize_deterministically() -> None:
    parsed = TokenOptimizationRouterToolInput.model_validate(_valid_input())
    dumped = parsed.model_dump()
    assert dumped["configuration_id"] == "exact_only"
    assert dumped["reason_code"] == "exact_duplicates"
    assert dumped["risk"] == "low"


def test_router_policy_confidence_bounds() -> None:
    with pytest.raises(ValueError):
        TokenOptimizationLLMRouterPolicy(minimum_confidence=1.5)


def test_router_policy_execute_when_review_required_forbidden() -> None:
    with pytest.raises(ValueError):
        TokenOptimizationLLMRouterPolicy(execute_when_review_required=True)
