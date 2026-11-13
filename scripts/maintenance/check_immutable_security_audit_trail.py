#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-23.1 — immutable multi-region security audit trail (durable, fail closed)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from intergrax.applications._shared.security_audit_trail_wiring import resolve_security_audit_trail_wiring
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.core.security.security_configuration import AUDIT_SINK_MISSING, SecurityConfigurationError
from intergrax.runtime.events.store import resolve_runtime_event_persistence


def main() -> int:
    env = ApplicationEnvironmentProfile.product_defaults()

    # PDC-D6-R020/R024: no evidence sink means no trail; there is no in-memory fallback.
    try:
        resolve_security_audit_trail_wiring(env)
    except SecurityConfigurationError as exc:
        if exc.reason_code != AUDIT_SINK_MISSING:
            print(f"unexpected configuration error: {exc.reason_code}", file=sys.stderr)
            return 1
    else:
        print("product host without an evidence sink must fail closed", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db_path = Path(tmp) / "evidence.db"
        persistence = resolve_runtime_event_persistence(db_path=db_path)
        wiring = resolve_security_audit_trail_wiring(env, evidence_persistence=persistence)
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
        restarted = resolve_runtime_event_persistence(db_path=db_path)
        from intergrax.runtime.security.security_audit_trail import EvidencePlaneSecurityAuditTrail

        entries = EvidencePlaneSecurityAuditTrail(restarted).list_entries(tenant_id=env.profile_id)  # type: ignore[arg-type]
        if len(entries) < 2:
            print("audit trail did not survive a restart", file=sys.stderr)
            return 1
        regions = len(report.regions)
    print(f"OK: immutable security audit trail ({regions} regions, durable across restart)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
