# © Artur Czarnecki. All rights reserved.

"""EnterpriseAuditRunner — orchestrates allowlisted checks."""

from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from intergrax.platform.enterprise_audit.baseline import apply_drift, baseline_path, load_baseline
from intergrax.platform.enterprise_audit.checks.executor import execute_manifest_check
from intergrax.platform.enterprise_audit.contracts import (
    AuditExecutionManifest,
    AuditFinding,
    AuditMode,
    AuditReport,
    CheckRunOutcome,
    EnterpriseAuditExitCode,
    ManifestCheck,
)
from intergrax.platform.enterprise_audit.manifest import checks_for_mode, load_default_manifest
from intergrax.platform.enterprise_audit.modes import (
    aggregate_status,
    compute_exit_code,
    default_limitations,
)
from intergrax.platform.enterprise_audit.report import write_reports
from intergrax.platform.enterprise_audit.sandbox import CheckExecutionSandbox
from intergrax.platform.enterprise_audit.tenant import enrich_tenant_findings
from intergrax.platform.enterprise_audit.vbn import enrich_vbn_findings, vbn_domain_rollup


class EnterpriseAuditRunner:
    def __init__(
        self,
        repo_root: Path,
        *,
        manifest_checks: tuple[ManifestCheck, ...] | None = None,
        max_workers: int = 2,
        dry_run: bool = False,
    ) -> None:
        self._repo_root = repo_root.resolve()
        self._checks = manifest_checks or load_default_manifest()
        self._max_workers = max(1, max_workers)
        self._dry_run = dry_run

    def resolve_git_sha(self) -> str:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self._repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return completed.stdout.strip()

    def run(
        self,
        mode: AuditMode,
        *,
        output_dir: Path,
        baseline_dir: Path | None = None,
        changed_check_ids: frozenset[str] | None = None,
        check_ids_filter: frozenset[str] | None = None,
    ) -> tuple[AuditReport, AuditExecutionManifest, EnterpriseAuditExitCode]:
        current_sha = self.resolve_git_sha()
        planned = checks_for_mode(self._checks, mode, changed_check_ids=changed_check_ids)
        if check_ids_filter:
            planned = tuple(c for c in planned if c.check_id in check_ids_filter)

        sandbox = CheckExecutionSandbox(self._repo_root)
        outcomes: list[CheckRunOutcome] = []

        if self._dry_run:
            for check in planned:
                outcomes.append(
                    CheckRunOutcome(
                        check_id=check.check_id,
                        executed=False,
                        skipped=True,
                        skip_reason="dry_run",
                    )
                )
        elif len(planned) <= 1 or self._max_workers == 1:
            for check in planned:
                outcomes.append(execute_manifest_check(check, sandbox, mode=mode))
        else:
            with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
                futures = {pool.submit(execute_manifest_check, check, sandbox, mode=mode): check for check in planned}
                for future in as_completed(futures):
                    outcomes.append(future.result())

        outcomes_tuple = tuple(sorted(outcomes, key=lambda o: o.check_id))
        findings: list[AuditFinding] = []
        for outcome in outcomes_tuple:
            findings.extend(outcome.findings)

        findings_tuple = enrich_tenant_findings(findings)
        findings_tuple = enrich_vbn_findings(findings_tuple)
        rollup = vbn_domain_rollup(findings_tuple)
        coverage_summary = {
            "findings_total": len(findings_tuple),
            "checks_planned": len(planned),
            "vbn_domains_accounted": rollup.get("domains_accounted", 31),
        }

        baseline_file = baseline_path(baseline_dir or self._default_baseline_dir())
        baseline = load_baseline(baseline_file)
        baseline_sha = baseline.accepted_baseline_sha if baseline else None
        drifted = apply_drift(findings_tuple, baseline)

        incomplete = any(o.skipped for o in outcomes_tuple) and mode is not AuditMode.FAST
        internal_error = any(f.severity.value == "BLOCKER" and f.category.value == "internal" for f in drifted)
        status = aggregate_status(drifted, incomplete=incomplete, internal_error=internal_error)
        exit_code = compute_exit_code(status, outcomes_tuple)

        manifest = AuditExecutionManifest(
            audit_mode=mode,
            qualification_sha=current_sha,
            audit_execution_sha=current_sha,
            checks_planned=len(planned),
            checks_executed=sum(1 for o in outcomes_tuple if o.executed),
            checks_skipped=sum(1 for o in outcomes_tuple if o.skipped),
            coverage_label="FULL" if len(planned) == len(self._checks) else "PARTIAL",
            outcomes=outcomes_tuple,
        )

        report = AuditReport(
            status=status,
            exit_code=exit_code,
            audit_mode=mode,
            current_sha=current_sha,
            baseline_sha=baseline_sha,
            findings=drifted,
            limitations=default_limitations(mode),
            coverage_matrix_summary=coverage_summary,
        )

        write_reports(output_dir, report, manifest, outcomes_tuple)
        return report, manifest, exit_code

    def _default_baseline_dir(self) -> Path:
        return self._repo_root / "tests" / "qualification" / "ent_audit_x"
