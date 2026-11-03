# © Artur Czarnecki. All rights reserved.

"""Evidence bundle builder for ENT-AUDIT-X qualification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from intergrax.platform.enterprise_audit.contracts import AuditExecutionManifest, AuditMode

_CATALOG_PATH = Path(__file__).resolve().parent / "data" / "evidence_artifact_catalog.v1.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_evidence_artifact_catalog() -> dict:
    return json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))


def _assert_execution_manifest_allows_publication(manifest: AuditExecutionManifest) -> None:
    if not manifest.certifying_run:
        raise ValueError("evidence bundle publication requires certifying_run=true on execution manifest")
    if manifest.checks_executed != manifest.checks_planned:
        raise ValueError(
            "evidence bundle requires checks_executed==checks_planned "
            f"({manifest.checks_executed}!={manifest.checks_planned})"
        )
    if manifest.audit_mode in {AuditMode.FULL, AuditMode.FREEZE} and manifest.coverage_label != "FULL":
        raise ValueError(
            f"evidence bundle requires coverage_label FULL for {manifest.audit_mode.value}, "
            f"got {manifest.coverage_label}"
        )


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
    artifacts: list[dict[str, str]] = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(output_dir).as_posix()
        if rel == "evidence_manifest.v1.json":
            continue
        artifacts.append({"path": rel, "sha256": sha256_file(path)})
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "qualification_sha": qualification_sha,
        "certifying_run": True,
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
        path = output_dir / item["path"]
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
            if rel != "evidence_manifest.v1.json":
                files.add(rel)
    return files
