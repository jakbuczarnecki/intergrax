# © Artur Czarnecki. All rights reserved.

"""Adaptive tool engine mode hook wiring (TOOL-ENG-10)."""

from __future__ import annotations

from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.adaptive.tool_engine_selection_engine import ToolEngineSelectionEngine
from intergrax.runtime.architecture.adaptive_governance import AdaptiveLoopKind
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.tools.tool_engine_hook import ToolEngineHook


def resolve_tool_engine_hook(env: ApplicationEnvironmentProfile) -> ToolEngineHook:
    """Expose per-run tool mode resolution when adaptive routing tuning is enabled."""
    profile = env.adaptive_profile
    loop_enabled = AdaptiveLoopKind.ROUTING_TUNING in profile.enabled_loops
    is_product = env.application_profile is ApplicationProfile.PRODUCT
    enabled = is_product and profile.enabled and loop_enabled
    engine = ToolEngineSelectionEngine()
    return ToolEngineHook(enabled=enabled, engine_id=engine.engine_id)


def apply_tool_engine_hook_to_runtime_config(
    config: RuntimeConfig,
    env: ApplicationEnvironmentProfile,
) -> RuntimeConfig:
    config.tool_engine_hook = resolve_tool_engine_hook(env)
    return config
