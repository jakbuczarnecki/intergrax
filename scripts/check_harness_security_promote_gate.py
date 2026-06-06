#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""V-SEC STABLE promote gate using ``harness_security_stack`` scanners (Phase M.6 P6)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.security_scanner import ScanFinding, ScanReport, SecurityScannerBackend
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.presets import harness_security_stack

CRITICAL_SEVERITIES = frozenset({"critical", "high", "CRITICAL", "HIGH"})


class TrivyCliScanner:
    """Local ``trivy fs`` adapter for release pipelines."""

    def scan_image(self, image_ref: str) -> ScanReport:
        del image_ref
        return ScanReport(target="image", status="skipped", findings=[])

    def scan_repo(self, repo_path: str) -> ScanReport:
        executable = shutil.which("trivy")
        if executable is None:
            raise RuntimeError("trivy CLI not found on PATH")
        completed = subprocess.run(
            [
                executable,
                "fs",
                "--format",
                "json",
                "--severity",
                "HIGH,CRITICAL",
                "--scanners",
                "vuln,secret",
                repo_path,
            ],
            capture_output=True,
            text=True,
            check=False,
            cwd=REPO_ROOT,
        )
        if completed.returncode not in {0, 1}:
            raise RuntimeError(f"trivy fs failed: {completed.stderr.strip() or completed.stdout.strip()}")
        payload = json.loads(completed.stdout or "{}")
        findings = _parse_trivy_findings(payload)
        return ScanReport(target=repo_path, status="completed", findings=findings)


class SemgrepCliScanner:
    """Local ``semgrep scan`` adapter for release pipelines."""

    def scan_image(self, image_ref: str) -> ScanReport:
        del image_ref
        return ScanReport(target="image", status="skipped", findings=[])

    def scan_repo(self, repo_path: str) -> ScanReport:
        executable = shutil.which("semgrep")
        if executable is None:
            raise RuntimeError("semgrep CLI not found on PATH")
        completed = subprocess.run(
            [executable, "scan", "--config", "auto", "--json", "--quiet", repo_path],
            capture_output=True,
            text=True,
            check=False,
            cwd=REPO_ROOT,
        )
        if completed.returncode not in {0, 1}:
            raise RuntimeError(f"semgrep scan failed: {completed.stderr.strip() or completed.stdout.strip()}")
        payload = json.loads(completed.stdout or "{}")
        findings = _parse_semgrep_findings(payload)
        return ScanReport(target=repo_path, status="completed", findings=findings)


def _parse_trivy_findings(payload: dict[str, Any]) -> list[ScanFinding]:
    findings: list[ScanFinding] = []
    results = payload.get("Results")
    if not isinstance(results, list):
        return findings
    index = 0
    for result in results:
        if not isinstance(result, dict):
            continue
        target = str(result.get("Target") or "")
        vulnerabilities = result.get("Vulnerabilities")
        if not isinstance(vulnerabilities, list):
            continue
        for item in vulnerabilities:
            if not isinstance(item, dict):
                continue
            index += 1
            findings.append(
                ScanFinding(
                    id=str(item.get("VulnerabilityID") or f"trivy-{index}"),
                    severity=str(item.get("Severity") or ""),
                    title=str(item.get("Title") or item.get("VulnerabilityID") or "trivy-finding"),
                    resource=target,
                    detail=str(item.get("Description") or ""),
                )
            )
    return findings


def _parse_semgrep_findings(payload: dict[str, Any]) -> list[ScanFinding]:
    findings: list[ScanFinding] = []
    results = payload.get("results")
    if not isinstance(results, list):
        return findings
    for index, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            continue
        extra = item.get("extra")
        extra_dict = extra if isinstance(extra, dict) else {}
        findings.append(
            ScanFinding(
                id=str(item.get("check_id") or f"semgrep-{index}"),
                severity=str(extra_dict.get("severity") or "HIGH"),
                title=str(extra_dict.get("message") or item.get("check_id") or "semgrep-finding"),
                resource=str(item.get("path") or ""),
                detail=str(item.get("rule_id") or ""),
            )
        )
    return findings


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

    semgrep_slug = "semgrep"
    if semgrep_slug not in profile.options:
        print("security promote gate: semgrep not enabled in harness_security_stack options")
        return 1

    scan_backend = os.getenv("INTERGRAX_SECURITY_PROMOTE_SCAN_BACKEND", "catalog").strip().lower()
    if scan_backend == "cli":
        primary = TrivyCliScanner()
        semgrep = SemgrepCliScanner()
    else:
        primary = profile.resolve(IntegrationCategory.SECURITY_SCANNER)
        if not isinstance(primary, SecurityScannerBackend):
            print("security promote gate: primary scanner not configured")
            return 1
        from intergrax.integrations.registry.factory import resolve

        semgrep_resolved = resolve(IntegrationCategory.SECURITY_SCANNER, slug="semgrep", profile=profile)
        semgrep = semgrep_resolved if isinstance(semgrep_resolved, SecurityScannerBackend) else None

    run_scan = os.getenv("INTERGRAX_SECURITY_PROMOTE_RUN_SCAN", "").strip().lower() in {"1", "true", "yes"}
    if not run_scan:
        print(
            "security promote gate: wiring OK "
            "(set INTERGRAX_SECURITY_PROMOTE_RUN_SCAN=true to execute scans; "
            "INTERGRAX_SECURITY_PROMOTE_SCAN_BACKEND=cli for local CLIs)"
        )
        return 0

    if scan_backend == "cli":
        missing = [name for name, path in (("trivy", shutil.which("trivy")), ("semgrep", shutil.which("semgrep"))) if not path]
        if missing:
            print(f"security promote gate: missing CLI tools: {', '.join(missing)}")
            return 1

    repo_root = str(REPO_ROOT)
    failures: list[str] = []
    critical, detail = _scan_repo(primary, repo_root)
    print(f"{primary_slug} scan: {detail}")
    if critical > 0:
        failures.append(f"{primary_slug} reported {critical} critical/high findings")

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
