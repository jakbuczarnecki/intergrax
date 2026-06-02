# © Artur Czarnecki. All rights reserved.

"""Tier-3 lab harness wiring passed into reference agents (Phase U-Pol.1)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from intergrax.applications._shared.policy_wiring import build_runtime_policy_bundle
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle


@dataclass(frozen=True, slots=True)
class LabHarnessContext:
    """Policy and strict-mode options for lab reference agents."""

    policy_bundle: RuntimePolicyBundle
    strict_harness: bool = False
    trace_db_path: Path | None = None


def lab_harness_context_from_build_context(
    ctx: ApplicationBuildContext,
    *,
    trace_db_path: Path | None = None,
) -> LabHarnessContext:
    """Build harness context from Tier-3 ``ApplicationBuildContext``."""
    bundle = ctx.policy_bundle or build_runtime_policy_bundle()
    resolved_trace = trace_db_path if trace_db_path is not None else ctx.trace_db_path
    return LabHarnessContext(
        policy_bundle=bundle,
        strict_harness=ctx.strict_harness,
        trace_db_path=resolved_trace,
    )
