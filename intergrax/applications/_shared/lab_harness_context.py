# © Artur Czarnecki. All rights reserved.

"""Tier-3 lab harness wiring passed into reference agents (Phase U-Pol.1)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from intergrax.applications._shared.policy_wiring import build_runtime_policy_bundle
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.runtime.modality.modality_profile import ModalityProfile, lab_default_modality_profile
from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle
from intergrax.tools.registry.wiring import ToolWiringContext


@dataclass(frozen=True, slots=True)
class LabHarnessContext:
    """Policy and strict-mode options for lab reference agents."""

    policy_bundle: RuntimePolicyBundle
    strict_harness: bool = False
    trace_db_path: Path | None = None
    modality_profile: ModalityProfile | None = None
    tool_wiring_context: ToolWiringContext | None = None


def lab_harness_context_from_build_context(
    ctx: ApplicationBuildContext,
    *,
    trace_db_path: Path | None = None,
) -> LabHarnessContext:
    """Build harness context from Tier-3 ``ApplicationBuildContext``."""
    bundle = ctx.policy_bundle or build_runtime_policy_bundle()
    resolved_trace = trace_db_path if trace_db_path is not None else ctx.trace_db_path
    modality_profile = None
    if ctx.tool_wiring_context is not None:
        from intergrax.runtime.modality.modality_profile import MODALITY_PROFILE_EXTRA_KEY

        raw_profile = ctx.tool_wiring_context.extras.get(MODALITY_PROFILE_EXTRA_KEY)
        if isinstance(raw_profile, ModalityProfile):
            modality_profile = raw_profile
    if modality_profile is None and ctx.strict_harness:
        modality_profile = lab_default_modality_profile()
    return LabHarnessContext(
        policy_bundle=bundle,
        strict_harness=ctx.strict_harness,
        trace_db_path=resolved_trace,
        modality_profile=modality_profile,
        tool_wiring_context=ctx.tool_wiring_context,
    )
