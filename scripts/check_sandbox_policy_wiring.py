#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-11.1 — sandbox required for product side-effect tool profiles."""

from __future__ import annotations

import sys

from intergrax.applications._shared.sandbox_wiring import (
    product_requires_sandbox,
    tool_profile_with_sandbox,
    wire_sandbox_sessions,
)
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    SandboxProfile,
)
def main() -> int:
    base = ApplicationEnvironmentProfile.product_defaults(
        tool_ids=["sandbox.exec", "workspace.write_file"],
    )
    if not product_requires_sandbox(base):
        print("product_requires_sandbox must detect side-effect tools", file=sys.stderr)
        return 1

    env = base.model_copy(update={"sandbox": SandboxProfile(enable_exec_tool=True)})
    if wire_sandbox_sessions(env) is None:
        print("wire_sandbox_sessions must return manager when sandbox profile set", file=sys.stderr)
        return 1

    tool_profile = tool_profile_with_sandbox(env)
    if "sandbox.exec" not in tool_profile.enabled:
        print("tool_profile_with_sandbox must include sandbox.exec", file=sys.stderr)
        return 1

    no_tools = ApplicationEnvironmentProfile.product_defaults()
    if product_requires_sandbox(no_tools):
        print("empty product tool profile must not require sandbox", file=sys.stderr)
        return 1

    print("OK: sandbox policy wiring")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
