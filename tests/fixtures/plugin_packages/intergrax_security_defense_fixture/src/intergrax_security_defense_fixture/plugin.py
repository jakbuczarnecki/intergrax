# © Artur Czarnecki. All rights reserved.

"""Lab security defense plugin for EP discovery tests."""

from __future__ import annotations

from intergrax.runtime.hooks.hook_context import HookContext
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.security.defense_plugin import (
    SecurityFailMode,
    SecurityInspectionResult,
)


class FixtureDefensePlugin:
    plugin_id = "fixture_ep.defense"
    version = "1.0.0"
    hook_points = frozenset({HookPoint.BEFORE_TOOL_CALL})
    priority = 58
    fail_mode = SecurityFailMode.FAIL_CLOSED

    def inspect(self, point: HookPoint, ctx: HookContext) -> SecurityInspectionResult:
        return SecurityInspectionResult(allowed=True, plugin_id=self.plugin_id, hook_point=point.value)
