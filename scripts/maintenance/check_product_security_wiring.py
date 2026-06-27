#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-14.2 / 23.2 — retrieval poisoning + tool injection on product hosts."""

from __future__ import annotations

import sys

from intergrax.applications._shared.security_wiring import wire_application_security
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile


def main() -> int:
    profile = ApplicationEnvironmentProfile.product_defaults()
    sec = profile.security_profile
    if not sec.retrieval_poisoning_defense_enabled:
        print("product_defaults: retrieval_poisoning_defense_enabled must be True", file=sys.stderr)
        return 1
    if not sec.tool_injection_defense_enabled:
        print("product_defaults: tool_injection_defense_enabled must be True", file=sys.stderr)
        return 1

    wiring = wire_application_security(profile)
    expected = {"PromptDefenseMiddleware", "ToolInjectionDefenseMiddleware", "TenantSecurityMiddleware"}
    missing = expected.difference(set(wiring.enabled_middleware))
    if missing:
        print(f"product security wiring missing middleware: {sorted(missing)}", file=sys.stderr)
        return 1

    print("OK: product security wiring (retrieval poisoning + tool injection)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
