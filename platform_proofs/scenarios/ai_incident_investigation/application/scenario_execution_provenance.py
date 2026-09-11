# © Artur Czarnecki. All rights reserved.

"""Typed execution identity for AI Incident scenario production boundaries."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.execution_identity import RunId, validate_run_id


@dataclass(frozen=True, slots=True)
class ScenarioExecutionProvenance:
    """Immutable platform execution identity for post-run trace correlation."""

    platform_run_id: RunId
    execution_tenant_id: str


def scenario_execution_provenance(
    platform_run_id: str,
    execution_tenant_id: str,
) -> ScenarioExecutionProvenance:
    return ScenarioExecutionProvenance(
        platform_run_id=validate_run_id(platform_run_id),
        execution_tenant_id=execution_tenant_id,
    )
