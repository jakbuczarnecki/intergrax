# © Artur Czarnecki. All rights reserved.

"""Dynamic execution strategy L4 hook wiring (AUDIT-IDEAL-9.3)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.adaptive.execution_strategy_engine import ExecutionStrategyEngine
from intergrax.runtime.architecture.adaptive_governance import AdaptiveLoopKind


@dataclass(frozen=True, slots=True)
class ExecutionStrategyHook:
    enabled: bool
    engine_id: str
    loop_kind: str


def resolve_execution_strategy_hook(
    env: ApplicationEnvironmentProfile,
) -> ExecutionStrategyHook:
    """Expose execution-strategy tuning when adaptive profile enables the loop."""
    profile = env.adaptive_profile
    loop_enabled = AdaptiveLoopKind.EXECUTION_STRATEGY_TUNING in profile.enabled_loops
    is_product = env.application_profile is ApplicationProfile.PRODUCT
    enabled = is_product and profile.enabled and loop_enabled
    engine = ExecutionStrategyEngine()
    return ExecutionStrategyHook(
        enabled=enabled,
        engine_id=engine.engine_id,
        loop_kind=AdaptiveLoopKind.EXECUTION_STRATEGY_TUNING.value,
    )
