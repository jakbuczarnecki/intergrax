"""Explicit per-run scenario execution context — no global mutable state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.domain.failures import (
    ApplicationFailure,
    ApplicationFailureCode,
    InvalidScenarioContextError,
)

_SCENARIO_ID = "ERL-QUAL-004"
_SCENARIO_SLUG = "enterprise_payment_uncertainty_recovery"


@dataclass(frozen=True, slots=True)
class ScenarioExecutionContext:
    """Harness-provided scope for one scenario application execution."""

    scenario_id: str
    scenario_slug: str
    variant_id: str
    execution_reference: str
    correlation_ids: Mapping[str, str]


def validate_scenario_execution_context(context: ScenarioExecutionContext) -> None:
    """Validate required identifiers before business workflow execution."""
    missing: list[str] = []
    if not context.scenario_id.strip():
        missing.append("scenario_id")
    if not context.scenario_slug.strip():
        missing.append("scenario_slug")
    if not context.variant_id.strip():
        missing.append("variant_id")
    if not context.execution_reference.strip():
        missing.append("execution_reference")
    if not context.correlation_ids:
        missing.append("correlation_ids")
    required_keys = ("order_logical_id", "payment_correlation_id")
    for key in required_keys:
        value = context.correlation_ids.get(key, "").strip()
        if not value:
            missing.append(f"correlation_ids.{key}")
    if context.scenario_id != _SCENARIO_ID:
        raise InvalidScenarioContextError(
            ApplicationFailure(
                code=ApplicationFailureCode.INVALID_SCENARIO_CONTEXT,
                message=f"unsupported scenario_id: {context.scenario_id}",
            )
        )
    if context.scenario_slug != _SCENARIO_SLUG:
        raise InvalidScenarioContextError(
            ApplicationFailure(
                code=ApplicationFailureCode.INVALID_SCENARIO_CONTEXT,
                message=f"unsupported scenario_slug: {context.scenario_slug}",
            )
        )
    if missing:
        raise InvalidScenarioContextError(
            ApplicationFailure(
                code=ApplicationFailureCode.INVALID_SCENARIO_CONTEXT,
                message=f"missing or empty context fields: {', '.join(missing)}",
            )
        )
