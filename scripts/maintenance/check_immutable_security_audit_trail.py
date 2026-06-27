#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-23.1 — immutable multi-region security audit trail."""

from __future__ import annotations

import sys

from intergrax.applications._shared.security_audit_trail_wiring import resolve_security_audit_trail_wiring
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile


def main() -> int:
    wiring = resolve_security_audit_trail_wiring(ApplicationEnvironmentProfile.product_defaults())
    if not wiring.enabled:
        print("product host must enable immutable multi-region audit trail", file=sys.stderr)
        return 1
    report = wiring.report
    if report is None or not report.replicated:
        print("multi-region audit trail replication failed", file=sys.stderr)
        return 1
    if len(report.regions) < 2:
        print("audit trail must replicate to at least two regions", file=sys.stderr)
        return 1
    print(f"OK: immutable security audit trail ({len(report.regions)} regions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
