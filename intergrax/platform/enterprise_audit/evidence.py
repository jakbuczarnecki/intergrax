# © Artur Czarnecki. All rights reserved.

"""Evidence bundle builder for ENT-AUDIT-X qualification."""

from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
from pathlib import Path

from intergrax.platform.enterprise_audit.contracts import (
    AuditExecutionManifest,
    AuditMode,
    AuditSeverity,
    CheckRunOutcome,
)

_CATALOG_PATH = Path(__file__).resolve().parent / "data" / "evidence_artifact_catalog.v1.json"
_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_BLOCKING = frozenset({AuditSeverity.BLOCKER, AuditSeverity.ERROR})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_evidence_artifact_catalog() -> dict:
    return json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))


def _outcome_has_blocking_findings(outcome: CheckRunOutcome) -> bool:
    return any(f.severity in _BLOCKING for f in outcome.findings)


def assert_certifying_publication_eligible(manifest: AuditExecutionManifest) -> None:
    """Fail-closed eligibility for evidence bundle (C006/C026) — not counts alone."""
    if not manifest.certifying_run:
        raise ValueError("evidence bundle publication requires certifying_run=true on execution manifest")
    if manifest.run_status != "completed":
        raise ValueError(f"evidence bundle requires run_status=completed, got {manifest.run_status}")
    if manifest.checks_skipped != 0:
        raise ValueError(f"evidence bundle forbids skipped checks (checks_skipped={manifest.checks_skipped})")
    if not _SHA40.fullmatch(manifest.qualification_sha):
        raise ValueError("qualification_sha must be 40-char hex on execution manifest")
    if not _SHA40.fullmatch(manifest.audit_execution_sha):
        raise ValueError("audit_execution_sha must be 40-char hex on execution manifest")
    if manifest.checks_executed != manifest.checks_planned:
        raise ValueError(
            "evidence bundle requires checks_executed==checks_planned "
            f"({manifest.checks_executed}!={manifest.checks_planned})"
        )
    if manifest.audit_mode in {AuditMode.FULL, AuditMode.FREEZE}:
        if manifest.coverage_label != "FULL":
            raise ValueError(
                f"evidence bundle requires coverage_label FULL for {manifest.audit_mode.value}, "
                f"got {manifest.coverage_label}"
            )
        from intergrax.platform.enterprise_audit.manifest import checks_for_mode, load_default_manifest

        expected_checks = checks_for_mode(load_default_manifest(), manifest.audit_mode)
        if len(expected_checks) != 85:
            raise ValueError(f"closed-world FULL/FREEZE requires 85 planned checks, got {len(expected_checks)}")
        if manifest.checks_planned != 85:
            raise ValueError(f"execution manifest must plan 85 checks for FULL/FREEZE, got {manifest.checks_planned}")
        if len(manifest.outcomes) != 85:
            raise ValueError(f"execution manifest must include 85 outcomes, got {len(manifest.outcomes)}")
        by_id = {o.check_id: o for o in manifest.outcomes}
        for check in expected_checks:
            outcome = by_id.get(check.check_id)
            if outcome is None:
                raise ValueError(f"missing outcome for check_id {check.check_id}")
            if outcome.skipped or not outcome.executed:
                raise ValueError(f"check {check.check_id} not executed for evidence publication")
            if outcome.exit_code != 0:
                raise ValueError(f"check {check.check_id} exit_code={outcome.exit_code} blocks publication")
            if _outcome_has_blocking_findings(outcome):
                raise ValueError(f"check {check.check_id} has blocking findings on outcome")
    elif not manifest.outcomes:
        raise ValueError("execution manifest outcomes required for evidence publication")


def _assert_execution_manifest_allows_publication(manifest: AuditExecutionManifest) -> None:
    assert_certifying_publication_eligible(manifest)


def build_environment_fingerprint() -> dict[str, str]:
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "implementation": platform.python_implementation(),
    }


def build_evidence_manifest(
    output_dir: Path,
    *,
    run_id: str,
    qualification_sha: str,
    certifying_run: bool = True,
    execution_manifest: AuditExecutionManifest,
) -> Path:
    if not certifying_run:
        raise ValueError("evidence bundle publication requires certifying_run=true")
    _assert_execution_manifest_allows_publication(execution_manifest)
    if execution_manifest.qualification_sha != qualification_sha:
        raise ValueError("qualification_sha mismatch between bundle builder and execution manifest")
    from intergrax.platform.enterprise_audit.digests import manifest_digest

    artifacts: list[dict[str, str]] = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(output_dir).as_posix()
        if ".." in Path(rel).parts:
            raise ValueError(f"refusing to bundle path with traversal segment: {rel}")
        if rel == "evidence_manifest.v1.json":
            continue
        artifacts.append({"path": rel, "sha256": sha256_file(path)})
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "qualification_sha": qualification_sha,
        "audit_execution_sha": execution_manifest.audit_execution_sha,
        "check_manifest_digest": execution_manifest.check_manifest_digest or manifest_digest(),
        "certifying_run": True,
        "environment_fingerprint": build_environment_fingerprint(),
        "artifacts": artifacts,
    }
    out = output_dir / "evidence_manifest.v1.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def verify_evidence_manifest(output_dir: Path) -> bool:
    manifest_path = output_dir / "evidence_manifest.v1.json"
    if not manifest_path.is_file():
        return False
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in raw.get("artifacts", []):
        rel = str(item.get("path", ""))
        if ".." in Path(rel).parts:
            return False
        path = output_dir / rel
        if not path.is_file():
            return False
        if sha256_file(path) != item["sha256"]:
            return False
    return True


def list_bundle_files(output_dir: Path) -> set[str]:
    files: set[str] = set()
    for path in output_dir.rglob("*"):
        if path.is_file():
            rel = path.relative_to(output_dir).as_posix()
            if ".." in Path(rel).parts:
                raise ValueError(f"bundle path traversal rejected: {rel}")
            if rel != "evidence_manifest.v1.json":
                files.add(rel)
    return files
