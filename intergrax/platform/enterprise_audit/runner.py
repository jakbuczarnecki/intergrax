# © Artur Czarnecki. All rights reserved.

"""EnterpriseAuditRunner — orchestrates allowlisted checks."""

from __future__ import annotations

from pathlib import Path

from intergrax.platform.enterprise_audit.baseline import (
    EnterpriseAuditBaseline,
    apply_drift,
    assert_baseline_dir_jailed,
    baseline_path,
    candidate_baseline_path,
    findings_to_baseline_rows,
    load_baseline,
    save_candidate_baseline,
)
from intergrax.platform.enterprise_audit.checks.executor import execute_manifest_check, skip_prerequisite
from intergrax.platform.enterprise_audit.contracts import (
    AggregateAuditStatus,
    AuditCategory,
    AuditExecutionManifest,
    AuditFinding,
    AuditMode,
    AuditReport,
    AuditSeverity,
    CheckRunOutcome,
    EnterpriseAuditExitCode,
    ManifestCheck,
)
from intergrax.platform.enterprise_audit.coverage import compute_coverage_label
from intergrax.platform.enterprise_audit.digests import frz_coverage_artifact, manifest_digest
from intergrax.platform.enterprise_audit.evidence import build_evidence_manifest
from intergrax.platform.enterprise_audit.manifest import checks_for_mode, load_default_manifest
from intergrax.platform.enterprise_audit.modes import (
    aggregate_status,
    compute_exit_code,
    default_limitations,
    fast_floor_incomplete,
)
from intergrax.platform.enterprise_audit.repo_identity import assert_git_ancestor, read_repo_identity
from intergrax.platform.enterprise_audit.report import write_reports
from intergrax.platform.enterprise_audit.sandbox import CheckExecutionSandbox
from intergrax.platform.enterprise_audit.schedule import topological_order
from intergrax.platform.enterprise_audit.tenant import enrich_tenant_findings
from intergrax.platform.enterprise_audit.vbn import enrich_vbn_findings, load_vbn_domain_report, vbn_domain_rollup

_VBN_DOMAIN_TOTAL = 31


class EnterpriseAuditRunner:
    def __init__(
        self,
        repo_root: Path,
        *,
        manifest_checks: tuple[ManifestCheck, ...] | None = None,
        dry_run: bool = False,
        require_clean_repo: bool = False,
    ) -> None:
        self._repo_root = repo_root.resolve()
        self._checks = manifest_checks or load_default_manifest()
        self._dry_run = dry_run
        self._require_clean_repo = require_clean_repo

    def resolve_git_sha(self) -> str:
        return read_repo_identity(self._repo_root).head_sha

    def run(
        self,
        mode: AuditMode,
        *,
        output_dir: Path,
        baseline_dir: Path | None = None,
        changed_check_ids: frozenset[str] | None = None,
        check_ids_filter: frozenset[str] | None = None,
        bootstrap_baseline: bool = False,
        write_evidence_bundle: bool = False,
        certifying_run: bool = True,
    ) -> tuple[AuditReport, AuditExecutionManifest, EnterpriseAuditExitCode]:
        identity = read_repo_identity(self._repo_root)
        if self._require_clean_repo and identity.dirty:
            raise RuntimeError("dirty working tree not allowed for this audit mode")

        if certifying_run and check_ids_filter:
            return self._blocked_report(
                mode,
                identity,
                output_dir,
                finding_id="CERTIFYING:CHECK_FILTER_FORBIDDEN",
                message="certifying runs forbid --check-id; use --non-certifying for diagnostic filters",
                certifying_run=certifying_run,
            )

        baseline_root = assert_baseline_dir_jailed(
            baseline_dir or self._default_baseline_dir(),
            self._repo_root,
        )
        baseline_file = baseline_path(baseline_root)
        candidate_file = candidate_baseline_path(baseline_root)
        baseline = load_baseline(baseline_file)

        if mode is AuditMode.FREEZE and baseline is None and not bootstrap_baseline:
            return self._blocked_report(
                mode,
                identity,
                output_dir,
                finding_id="FREEZE:NO_BASELINE",
                message="FREEZE requires accepted baseline or certifying --bootstrap-candidate",
                certifying_run=certifying_run,
            )

        if mode is AuditMode.FREEZE and baseline is not None and certifying_run:
            try:
                assert_git_ancestor(
                    self._repo_root,
                    ancestor_sha=baseline.accepted_baseline_sha,
                    descendant_sha=identity.head_sha,
                )
            except ValueError as exc:
                return self._blocked_report(
                    mode,
                    identity,
                    output_dir,
                    finding_id="FREEZE:BASELINE_ANCESTRY",
                    message=str(exc),
                    certifying_run=certifying_run,
                    baseline_sha=baseline.accepted_baseline_sha,
                )

        planned_all = checks_for_mode(self._checks, mode, changed_check_ids=changed_check_ids)
        filtered = bool(check_ids_filter)
        if check_ids_filter:
            needed = set(check_ids_filter)
            by_id = {c.check_id: c for c in self._checks}
            stack = list(check_ids_filter)
            while stack:
                cid = stack.pop()
                check = by_id.get(cid)
                if check is None:
                    continue
                for dep in check.depends_on:
                    if dep not in needed:
                        needed.add(dep)
                        stack.append(dep)
            planned_all = tuple(c for c in planned_all if c.check_id in needed)
        if mode is AuditMode.CHANGED_SINCE and not planned_all:
            return self._blocked_report(
                mode,
                identity,
                output_dir,
                finding_id="CHANGED-SINCE:EMPTY",
                message="CHANGED-SINCE selection empty",
                certifying_run=certifying_run,
                baseline_sha=baseline.accepted_baseline_sha if baseline else None,
            )

        ordered = topological_order(planned_all)
        sandbox = CheckExecutionSandbox(self._repo_root)
        output_dir.mkdir(parents=True, exist_ok=True)
        domain_report = load_vbn_domain_report(self._repo_root, output_dir)
        outcomes: list[CheckRunOutcome] = []
        failed_prereqs: set[str] = set()

        if self._dry_run:
            for check in ordered:
                outcomes.append(
                    CheckRunOutcome(
                        check_id=check.check_id,
                        executed=False,
                        skipped=True,
                        skip_reason="dry_run",
                    )
                )
        else:
            for check in ordered:
                if check.depends_on & failed_prereqs:
                    outcome = skip_prerequisite(check)
                    outcomes.append(outcome)
                    continue
                outcome = execute_manifest_check(check, sandbox, mode=mode)
                outcomes.append(outcome)
                if _prerequisite_failed(outcome):
                    failed_prereqs.add(check.check_id)

        outcomes_tuple = tuple(sorted(outcomes, key=lambda o: o.check_id))
        findings: list[AuditFinding] = []
        for outcome in outcomes_tuple:
            findings.extend(outcome.findings)

        findings_tuple = enrich_tenant_findings(findings)
        findings_tuple = enrich_vbn_findings(findings_tuple)
        rollup = vbn_domain_rollup(findings_tuple, domain_report=domain_report)
        coverage_summary = {
            "findings_total": len(findings_tuple),
            "checks_planned": len(ordered),
            "vbn_domains_accounted": rollup.get("domains_accounted", 0),
            "manifest_digest": manifest_digest(),
        }

        baseline_sha = baseline.accepted_baseline_sha if baseline else None
        drifted = apply_drift(findings_tuple, baseline)

        if (
            certifying_run
            and not self._dry_run
            and mode in {AuditMode.FULL, AuditMode.FREEZE}
            and rollup.get("domains_accounted", 0) < _VBN_DOMAIN_TOTAL
        ):
            drifted = drifted + (
                AuditFinding(
                    finding_id="VBN:DOMAIN_COVERAGE_GAP",
                    check_id="VBN-ROLLUP",
                    severity=AuditSeverity.BLOCKER,
                    category=AuditCategory.VBN,
                    message=(
                        f"VBN domain coverage fail-closed: "
                        f"{rollup.get('domains_accounted', 0)}/{_VBN_DOMAIN_TOTAL} accounted"
                    ),
                ),
            )

        floor_incomplete = False if self._dry_run else fast_floor_incomplete(mode, self._checks, outcomes_tuple)
        incomplete = floor_incomplete or (
            mode in {AuditMode.FULL, AuditMode.FREEZE}
            and (any(o.skipped for o in outcomes_tuple) or len(ordered) < len(self._checks))
        )
        internal_error = any(f.severity.value == "BLOCKER" and f.category.value == "internal" for f in drifted)
        status = aggregate_status(drifted, incomplete=incomplete, internal_error=internal_error)
        exit_code = compute_exit_code(status, outcomes_tuple)

        coverage_label = compute_coverage_label(
            mode,
            total_checks=len(self._checks),
            planned=len(ordered),
            outcomes=outcomes_tuple,
            findings=drifted,
            certifying_run=certifying_run,
            filtered=filtered,
        )

        manifest = AuditExecutionManifest(
            audit_mode=mode,
            qualification_sha=identity.head_sha,
            audit_execution_sha=identity.head_sha,
            repo_remote=identity.remote_url,
            branch=identity.branch,
            checks_planned=len(ordered),
            checks_executed=sum(1 for o in outcomes_tuple if o.executed),
            checks_skipped=sum(1 for o in outcomes_tuple if o.skipped),
            coverage_label=coverage_label,
            certifying_run=certifying_run,
            outcomes=outcomes_tuple,
        )

        report = AuditReport(
            status=status,
            exit_code=exit_code,
            audit_mode=mode,
            current_sha=identity.head_sha,
            baseline_sha=baseline_sha,
            findings=drifted,
            limitations=default_limitations(mode),
            coverage_matrix_summary=coverage_summary,
            certifying_run=certifying_run,
        )

        write_reports(
            output_dir,
            report,
            manifest,
            outcomes_tuple,
            extra_artifacts={
                "frz_coverage.v1.json": frz_coverage_artifact(self._checks),
                "vbn_domains.v1.json": domain_report or rollup,
            },
        )

        if bootstrap_baseline and mode is AuditMode.FREEZE and exit_code is EnterpriseAuditExitCode.CLEAN:
            save_candidate_baseline(
                candidate_file,
                EnterpriseAuditBaseline(
                    accepted_baseline_sha=identity.head_sha,
                    findings=findings_to_baseline_rows(drifted),
                ),
            )

        if write_evidence_bundle:
            build_evidence_manifest(
                output_dir,
                run_id=output_dir.name,
                qualification_sha=identity.head_sha,
                certifying_run=certifying_run,
            )

        return report, manifest, exit_code

    def _blocked_report(
        self,
        mode: AuditMode,
        identity,
        output_dir: Path,
        *,
        finding_id: str,
        message: str,
        certifying_run: bool,
        baseline_sha: str | None = None,
    ) -> tuple[AuditReport, AuditExecutionManifest, EnterpriseAuditExitCode]:
        report = AuditReport(
            status=AggregateAuditStatus.BLOCKED,
            exit_code=EnterpriseAuditExitCode.VIOLATION,
            audit_mode=mode,
            current_sha=identity.head_sha,
            baseline_sha=baseline_sha,
            certifying_run=certifying_run,
            findings=(
                AuditFinding(
                    finding_id=finding_id,
                    check_id="SYS-INV",
                    severity=AuditSeverity.BLOCKER,
                    category=AuditCategory.INTERNAL,
                    message=message,
                ),
            ),
        )
        manifest = AuditExecutionManifest(
            audit_mode=mode,
            qualification_sha=identity.head_sha,
            audit_execution_sha=identity.head_sha,
            repo_remote=identity.remote_url,
            branch=identity.branch,
            certifying_run=certifying_run,
        )
        write_reports(output_dir, report, manifest, ())
        return report, manifest, EnterpriseAuditExitCode.VIOLATION

    def _default_baseline_dir(self) -> Path:
        return self._repo_root / "tests" / "qualification" / "ent_audit_x"


def _prerequisite_failed(outcome: CheckRunOutcome) -> bool:
    if outcome.skipped and outcome.skip_reason == "internal_pending":
        return True
    if outcome.exit_code != 0 and outcome.executed:
        return True
    if any(f.severity in {AuditSeverity.BLOCKER, AuditSeverity.ERROR} for f in outcome.findings):
        return True
    return False
