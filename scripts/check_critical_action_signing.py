#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-4.1 — cryptographic signing / audit-protect for critical actions."""

from __future__ import annotations

import sys

from intergrax.applications._shared.critical_action_signing_wiring import (
    resolve_critical_action_signing_wiring,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile


def main() -> int:
    wiring = resolve_critical_action_signing_wiring(ApplicationEnvironmentProfile.product_defaults())
    if not wiring.enabled:
        print("product host must enable critical action signing", file=sys.stderr)
        return 1
    if wiring.bootstrap_signature is None:
        print("critical action bootstrap signature missing", file=sys.stderr)
        return 1
    print(f"OK: critical action signing ({wiring.bootstrap_signature.action_kind.value})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
