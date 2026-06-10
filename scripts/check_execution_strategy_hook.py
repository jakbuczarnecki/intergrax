#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-9.3 — dynamic execution strategy selection (L4 hook)."""

from __future__ import annotations

import sys

from intergrax.applications._shared.execution_strategy_wiring import resolve_execution_strategy_hook
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.adaptive.execution_strategy_engine import ExecutionStrategyEngine


def main() -> int:
    env = ApplicationEnvironmentProfile.product_defaults()
    hook = resolve_execution_strategy_hook(env)
    if not hook.enabled:
        print("product host must enable execution strategy hook", file=sys.stderr)
        return 1
    if hook.engine_id != ExecutionStrategyEngine().engine_id:
        print("execution strategy hook must reference execution_strategy engine", file=sys.stderr)
        return 1

    print("OK: execution strategy L4 hook")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
