#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""TOOL-ENG-10 — adaptive tool engine hook gate."""

from __future__ import annotations

import sys

from intergrax.applications._shared.tool_engine_wiring import resolve_tool_engine_hook
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.adaptive.tool_engine_selection_engine import ToolEngineSelectionEngine
from intergrax.runtime.architecture.adaptive_governance import AdaptiveLoopKind


def main() -> int:
    env = ApplicationEnvironmentProfile.product_defaults()
    env = env.model_copy(
        update={
            "adaptive_profile": env.adaptive_profile.model_copy(
                update={
                    "enabled": True,
                    "enabled_loops": [AdaptiveLoopKind.ROUTING_TUNING],
                }
            )
        }
    )
    hook = resolve_tool_engine_hook(env)
    if not hook.enabled:
        print("product host must enable tool engine hook", file=sys.stderr)
        return 1
    if hook.engine_id != ToolEngineSelectionEngine().engine_id:
        print("tool engine hook must reference tool_engine_selection engine", file=sys.stderr)
        return 1

    print("OK: tool engine AHI hook")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
