# © Artur Czarnecki. All rights reserved.

"""Contract stability tests for Decision qualification taxonomy (DS-E2E-14.3)."""

from __future__ import annotations

import pytest

from intergrax.decision_system.qualification.taxonomy import (
    BOUNDARY_ORDER,
    CATEGORY_PRECEDENCE,
    DecisionFailureBoundary,
    DecisionFailureCategory,
    DecisionFailureDiagnosticCode,
    DecisionFailureOwner,
    boundary_rank,
    earliest_boundary,
)


def test_failure_category_enum_stability() -> None:
    assert tuple(item.value for item in DecisionFailureCategory) == (
        "platform_contract",
        "model_behavior",
        "evaluator_semantics",
        "provider_infrastructure",
        "environment",
        "observability_gap",
        "unclassified",
    )


def test_failure_owner_enum_stability() -> None:
    assert tuple(item.value for item in DecisionFailureOwner) == (
        "decision_system",
        "model",
        "evaluator",
        "provider",
        "environment",
        "execution_engine",
        "observability",
    )


def test_failure_boundary_ordering() -> None:
    assert BOUNDARY_ORDER[0] is DecisionFailureBoundary.ENVIRONMENT_RESOLUTION
    assert BOUNDARY_ORDER[-1] is DecisionFailureBoundary.EVALUATOR
    assert len(BOUNDARY_ORDER) == len(DecisionFailureBoundary)
    assert boundary_rank(DecisionFailureBoundary.TOOL_DISPATCH) < boundary_rank(
        DecisionFailureBoundary.REASONING
    )
    assert earliest_boundary(
        DecisionFailureBoundary.REASONING,
        DecisionFailureBoundary.TOOL_DISPATCH,
    ) is DecisionFailureBoundary.TOOL_DISPATCH


def test_category_precedence_order() -> None:
    assert CATEGORY_PRECEDENCE[0] is DecisionFailureCategory.ENVIRONMENT
    assert CATEGORY_PRECEDENCE[-1] is DecisionFailureCategory.UNCLASSIFIED


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (
            DecisionFailureDiagnosticCode.MODEL_TOOL_USE_DEFICIENCY,
            "decision.model.tool_use_deficiency",
        ),
        (
            DecisionFailureDiagnosticCode.MODEL_EPISTEMIC_CONTRADICTION,
            "decision.model.epistemic_contradiction",
        ),
        (
            DecisionFailureDiagnosticCode.PLATFORM_TRACE_NOT_FINALIZED,
            "decision.platform.trace_not_finalized",
        ),
        (
            DecisionFailureDiagnosticCode.PROVIDER_RATE_LIMIT,
            "decision.provider.rate_limit",
        ),
    ],
)
def test_diagnostic_code_stability(
    code: DecisionFailureDiagnosticCode,
    expected: str,
) -> None:
    assert code.value == expected


def test_category_and_owner_are_distinct_axes() -> None:
    assert DecisionFailureCategory.PLATFORM_CONTRACT is not DecisionFailureOwner.DECISION_SYSTEM
    assert DecisionFailureCategory.PLATFORM_CONTRACT.value != DecisionFailureOwner.EXECUTION_ENGINE.value
