# © Artur Czarnecki. All rights reserved.

"""Tier-3 lab harness wiring passed into reference agents (Phase U-Pol.1)."""

from __future__ import annotations

from pathlib import Path

from intergrax.agents.reference_harness import (
    LabHarnessContext,
    lab_harness_context_from_modality_tooling,
)
from intergrax.applications._shared.application_composition_context import (
    optional_factory_composition,
)
from intergrax.applications._shared.policy_wiring import build_runtime_policy_bundle
from intergrax.applications.contracts.build_context import ApplicationBuildContext

__all__ = [
    "LabHarnessContext",
    "lab_harness_context_from_build_context",
]


def lab_harness_context_from_build_context(
    ctx: ApplicationBuildContext,
    *,
    trace_db_path: Path | None = None,
) -> LabHarnessContext:
    """Build harness context from public factory context plus active composition scope."""
    composition = optional_factory_composition()
    bundle = (
        composition.policy_bundle
        if composition is not None and composition.policy_bundle is not None
        else build_runtime_policy_bundle()
    )
    tool_wiring_context = (
        composition.tool_wiring_context if composition is not None else None
    )
    resolved_trace = trace_db_path if trace_db_path is not None else ctx.trace_db_path
    return lab_harness_context_from_modality_tooling(
        policy_bundle=bundle,
        strict_harness=ctx.strict_harness,
        trace_db_path=resolved_trace,
        tool_wiring_context=tool_wiring_context,
    )
