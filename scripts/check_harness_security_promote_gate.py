#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""V-SEC STABLE promote gate using ``harness_security_stack`` scanners (Phase M.6 P6)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.security_scanner import SecurityScannerBackend
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.presets import harness_security_stack

CRITICAL_SEVERITIES = frozenset({"critical", "high", "CRITICAL", "HIGH"})


def _scan_repo(backend: SecurityScannerBackend, target: str) -> tuple[int, str]:
    report = backend.scan_repo(target)
    critical = sum(1 for item in report.findings if item.severity in CRITICAL_SEVERITIES)
    detail = f"{type(backend).__name__} status={report.status} findings={len(report.findings)} critical={critical}"
    return critical, detail


def main() -> int:
    register_default_integrations()
    profile = harness_security_stack()

    primary_slug = profile.slug_for_category(IntegrationCategory.SECURITY_SCANNER.value)
    if primary_slug != "trivy":
        print(f"security promote gate: expected primary scanner trivy, got {primary_slug!r}")
        return 1

    primary = profile.resolve(IntegrationCategory.SECURITY_SCANNER)
    if not isinstance(primary, SecurityScannerBackend):
        print("security promote gate: primary scanner not configured")
        return 1

    semgrep_slug = "semgrep"
    if semgrep_slug not in profile.options:
        print("security promote gate: semgrep not enabled in harness_security_stack options")
        return 1

    import os

    run_scan = os.getenv("INTERGRAX_SECURITY_PROMOTE_RUN_SCAN", "").strip().lower() in {"1", "true", "yes"}
    if not run_scan:
        print("security promote gate: wiring OK (set INTERGRAX_SECURITY_PROMOTE_RUN_SCAN=true to execute scans)")
        return 0

    repo_root = str(REPO_ROOT)
    failures: list[str] = []
    critical, detail = _scan_repo(primary, repo_root)
    print(f"{primary_slug} scan: {detail}")
    if critical > 0:
        failures.append(f"{primary_slug} reported {critical} critical/high findings")

    from intergrax.integrations.registry.factory import resolve

    semgrep = resolve(IntegrationCategory.SECURITY_SCANNER, slug="semgrep", profile=profile)
    if isinstance(semgrep, SecurityScannerBackend):
        critical, detail = _scan_repo(semgrep, repo_root)
        print(f"semgrep scan: {detail}")
        if critical > 0:
            failures.append(f"semgrep reported {critical} critical/high findings")

    if failures:
        for item in failures:
            print(f"FAIL: {item}")
        return 1

    print("security promote gate: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
