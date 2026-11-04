# © Artur Czarnecki. All rights reserved.

"""JSON Schema validators for ENT-AUDIT-X report and evidence artifacts (C069)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from intergrax.platform.enterprise_audit.evidence import sha256_file

_SCHEMA_DIR = Path(__file__).resolve().parent / "data" / "schemas"

_ARTIFACT_SCHEMA_FILES: dict[str, str] = {
    "enterprise_audit_report.json": "enterprise_audit_report.v1.schema.json",
    "enterprise_audit_outcomes.json": "enterprise_audit_outcomes.v1.schema.json",
    "frz_coverage.v1.json": "frz_coverage.v1.schema.json",
    "evidence_manifest.v1.json": "evidence_manifest.v1.schema.json",
    "vbn_domains.v1.json": "vbn_domains.v1.schema.json",
}

_BLOCKING_SEVERITIES = frozenset({"BLOCKER", "ERROR"})
_MD_REPORT_HEADER = "# Enterprise audit report"


def _load_schema(name: str) -> dict[str, Any]:
    return json.loads((_SCHEMA_DIR / name).read_text(encoding="utf-8"))


def _validator_for_schema(schema: dict[str, Any]) -> Draft202012Validator:
    return Draft202012Validator(schema)


def validate_against_schema(payload: dict[str, Any], schema: dict[str, Any]) -> tuple[str, ...]:
    """Validate a JSON object against a schema (fail-closed, including additionalProperties)."""
    validator = _validator_for_schema(schema)
    errors: list[str] = []
    for err in sorted(validator.iter_errors(payload), key=lambda e: list(e.path)):
        loc = ".".join(str(p) for p in err.path) or "<root>"
        errors.append(f"{loc}: {err.message}")
    return tuple(errors)


def _load_json(path: Path) -> tuple[Any | None, tuple[str, ...]]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        return None, (f"{path.name}: read failed: {exc}",)
    try:
        return json.loads(raw), ()
    except json.JSONDecodeError as exc:
        return None, (f"{path.name}: invalid JSON: {exc}",)


def validate_json_artifact(path: Path, *, schema_file: str) -> tuple[str, ...]:
    payload, load_errors = _load_json(path)
    if load_errors:
        return load_errors
    schema = _load_schema(schema_file)
    prefix = path.name
    return tuple(f"{prefix}: {msg}" for msg in validate_against_schema(payload, schema))


def validate_markdown_report(path: Path) -> tuple[str, ...]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return (f"{path.name}: read failed: {exc}",)
    if not text.strip():
        return (f"{path.name}: empty markdown report",)
    if not text.startswith(_MD_REPORT_HEADER):
        return (f"{path.name}: missing required report header",)
    return ()


def schema_file_for_artifact(name: str) -> str | None:
    return _ARTIFACT_SCHEMA_FILES.get(name)


def validate_named_artifact(output_dir: Path, artifact_name: str) -> tuple[str, ...]:
    path = output_dir / artifact_name
    if not path.is_file():
        return (f"missing artifact: {artifact_name}",)
    if artifact_name == "enterprise_audit_report.md":
        return validate_markdown_report(path)
    schema_file = schema_file_for_artifact(artifact_name)
    if schema_file is None:
        return (f"no schema registered for artifact: {artifact_name}",)
    return validate_json_artifact(path, schema_file=schema_file)


def validate_report_json(path: Path) -> tuple[str, ...]:
    return validate_json_artifact(path, schema_file="enterprise_audit_report.v1.schema.json")


def validate_evidence_manifest(path: Path) -> tuple[str, ...]:
    return validate_json_artifact(path, schema_file="evidence_manifest.v1.schema.json")


def validate_vbn_domains(path: Path) -> tuple[str, ...]:
    return validate_json_artifact(path, schema_file="vbn_domains.v1.schema.json")


def validate_outcomes(path: Path) -> tuple[str, ...]:
    payload, load_errors = _load_json(path)
    if load_errors:
        return load_errors
    schema = _load_schema("enterprise_audit_outcomes.v1.schema.json")
    validator = _validator_for_schema(schema)
    sorted_errors = sorted(validator.iter_errors(payload), key=lambda e: list(e.path))
    errors = [f"{path.name}: {err.message}" for err in sorted_errors]
    return tuple(errors)


def validate_frz_coverage(path: Path) -> tuple[str, ...]:
    return validate_json_artifact(path, schema_file="frz_coverage.v1.schema.json")


def validate_report_semantic_consistency(path: Path) -> tuple[str, ...]:
    """Detect status/exit_code tampering (ADV-31 / C044)."""
    payload, load_errors = _load_json(path)
    if load_errors:
        return load_errors
    errors = list(validate_report_json(path))
    status = str(payload.get("status", ""))
    exit_code = payload.get("exit_code")
    findings = payload.get("findings") or []
    blocking = [f for f in findings if str(f.get("severity", "")).upper() in _BLOCKING_SEVERITIES]
    if blocking and status in {"PASS", "CLEAN"}:
        errors.append("status PASS/CLEAN inconsistent with blocking findings")
    if exit_code == 0 and status not in {"PASS", "CLEAN", "INCOMPLETE"}:
        errors.append("exit_code 0 inconsistent with aggregate status")
    if exit_code != 0 and status in {"PASS", "CLEAN"}:
        errors.append("status PASS/CLEAN inconsistent with non-zero exit_code")
    return tuple(errors)


def validate_catalog_artifacts(output_dir: Path, artifact_names: tuple[str, ...]) -> tuple[str, ...]:
    errors: list[str] = []
    for name in artifact_names:
        errors.extend(validate_named_artifact(output_dir, name))
    return tuple(errors)


def validate_all_artifacts_in_dir(output_dir: Path) -> tuple[str, ...]:
    from intergrax.platform.enterprise_audit.evidence import load_evidence_artifact_catalog

    catalog = load_evidence_artifact_catalog()
    required = tuple(catalog.get("required_artifacts", []))
    optional = tuple(catalog.get("optional_artifacts", []))
    errors: list[str] = []
    for name in required:
        errors.extend(validate_named_artifact(output_dir, name))
    for name in optional:
        path = output_dir / name
        if path.is_file():
            errors.extend(validate_named_artifact(output_dir, name))
    report = output_dir / "enterprise_audit_report.json"
    if report.is_file():
        errors.extend(validate_report_semantic_consistency(report))
    return tuple(errors)


_B5_L3_ARTIFACT_NAMES: tuple[str, ...] = (
    "enterprise_audit_report.json",
    "enterprise_audit_report.md",
    "enterprise_audit_outcomes.json",
    "vbn_domains.v1.json",
    "frz_coverage.v1.json",
    "evidence_manifest.v1.json",
)
_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA64 = re.compile(r"^[0-9a-f]{64}$")


def _safe_repo_relative_path(repo_root: Path, relpath: str) -> Path | None:
    if not relpath or relpath.startswith("/") or "\\" in relpath:
        return None
    parts = Path(relpath).parts
    if ".." in parts:
        return None
    resolved = (repo_root / relpath).resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError:
        return None
    return resolved


def validate_b5_handoff_bundle(bundle_dir: Path) -> tuple[str, ...]:
    """Validate all known JSON artifacts under a published L3 bundle directory."""
    errors: list[str] = []
    for name, schema_file in _ARTIFACT_SCHEMA_FILES.items():
        path = bundle_dir / name
        if path.is_file():
            errors.extend(validate_json_artifact(path, schema_file=schema_file))
    md = bundle_dir / "enterprise_audit_report.md"
    if md.is_file():
        errors.extend(validate_markdown_report(md))
    return tuple(errors)


def verify_b5_publication_tree(
    repo_root: Path,
    qualification_dir: Path,
    *,
    expected_qualification_sha: str | None = None,
) -> tuple[str, ...]:
    """Strict closed-world verification of a published B5 qualification_at_<TES> tree."""
    repo_root = repo_root.resolve()
    qualification_dir = qualification_dir.resolve()
    errors: list[str] = []

    if not qualification_dir.is_dir():
        return ("qualification_dir missing",)

    index_path = qualification_dir / "publication_index.v1.json"
    if not index_path.is_file():
        return ("publication_index.v1.json missing",)
    if index_path.is_symlink():
        errors.append("publication_index.v1.json: symlink not allowed")

    payload, load_errors = _load_json(index_path)
    if load_errors:
        return load_errors
    schema = _load_schema("publication_index.v1.schema.json")
    errors.extend(f"publication_index: {msg}" for msg in validate_against_schema(payload, schema))

    qual_sha = str(payload.get("qualification_sha", ""))
    if expected_qualification_sha and qual_sha != expected_qualification_sha:
        errors.append("qualification_sha mismatch vs expected pin")
    dir_suffix = qualification_dir.name.removeprefix("qualification_at_")
    if qual_sha != dir_suffix:
        errors.append("qualification_sha inconsistent with qualification_at_<TES> directory name")

    l3_relpath = str(payload.get("l3_bundle_relpath", ""))
    l3_path = _safe_repo_relative_path(repo_root, l3_relpath)
    expected_l3 = (qualification_dir / "l3-cli-full").resolve()
    if l3_path is None:
        errors.append("l3_bundle_relpath invalid or escapes repo")
    elif l3_path != expected_l3:
        errors.append("l3_bundle_relpath does not match l3-cli-full under qualification_dir")
    elif not l3_path.is_dir():
        errors.append("l3 bundle directory missing")

    qw_meta = payload.get("qual_wave_report") or {}
    qw_relpath = str(qw_meta.get("path", ""))
    qw_expected = (qualification_dir / "qual_wave_report.json").resolve()
    qw_path = _safe_repo_relative_path(repo_root, qw_relpath)
    if qw_path is None:
        errors.append("qual_wave_report.path invalid or escapes repo")
    elif qw_path != qw_expected:
        errors.append("qual_wave_report.path must point to qual_wave_report.json in qualification_dir")
    elif not qw_path.is_file():
        errors.append("qual_wave_report.json missing")
    elif qw_path.is_symlink():
        errors.append("qual_wave_report.json: symlink not allowed")
    else:
        indexed_qw = str(qw_meta.get("sha256", ""))
        actual_qw = sha256_file(qw_path)
        if indexed_qw != actual_qw:
            errors.append("qual_wave_report sha256 mismatch (indexed vs file bytes)")

    artifacts = payload.get("artifacts") or []
    seen_paths: set[str] = set()
    indexed_names: set[str] = set()
    for idx, row in enumerate(artifacts):
        if not isinstance(row, dict):
            errors.append(f"artifacts[{idx}]: not an object")
            continue
        rel = str(row.get("path", ""))
        digest = str(row.get("sha256", ""))
        if rel in seen_paths:
            errors.append(f"duplicate indexed path: {rel}")
        seen_paths.add(rel)
        if not _SHA64.fullmatch(digest):
            errors.append(f"artifacts[{idx}]: invalid sha256")
        target = _safe_repo_relative_path(repo_root, rel)
        if target is None:
            errors.append(f"artifacts[{idx}]: path traversal or invalid path")
            continue
        if l3_path and not target.is_relative_to(l3_path):
            errors.append(f"artifacts[{idx}]: path outside l3 bundle")
        if target.is_symlink():
            errors.append(f"artifacts[{idx}]: symlink not allowed")
        if not target.is_file():
            errors.append(f"artifacts[{idx}]: missing file {rel}")
            continue
        indexed_names.add(target.name)
        actual = sha256_file(target)
        if actual != digest:
            errors.append(f"artifacts[{idx}]: sha256 mismatch for {target.name}")

    missing_l3 = set(_B5_L3_ARTIFACT_NAMES) - indexed_names
    if missing_l3:
        errors.append(f"mandatory L3 artifacts missing from index: {', '.join(sorted(missing_l3))}")

    if l3_path and l3_path.is_dir():
        errors.extend(validate_b5_handoff_bundle(l3_path))
        errors.extend(verify_closed_evidence_bundle(l3_path))

    allowed_files: set[Path] = {index_path.resolve(), qw_expected}
    if l3_path and l3_path.is_dir():
        allowed_files.update(p.resolve() for p in l3_path.iterdir() if p.is_file())
    present_files = {p.resolve() for p in qualification_dir.rglob("*") if p.is_file()}
    extra = present_files - allowed_files
    if extra:
        rel_extra = ", ".join(sorted(str(p.relative_to(qualification_dir)) for p in extra))
        errors.append(f"unexpected files under qualification tree: {rel_extra}")

    if qual_sha and _SHA40.fullmatch(qual_sha):
        try:
            import subprocess

            proc = subprocess.run(
                ["git", "merge-base", "--is-ancestor", qual_sha, "HEAD"],
                cwd=repo_root,
                capture_output=True,
                check=False,
            )
            if proc.returncode != 0:
                errors.append("qualification_sha is not an ancestor of HEAD")
        except OSError as exc:
            errors.append(f"git ancestry check failed: {exc}")

    return tuple(errors)


def verify_closed_evidence_bundle(output_dir: Path) -> tuple[str, ...]:
    from intergrax.platform.enterprise_audit.evidence import (
        list_bundle_files,
        load_evidence_artifact_catalog,
        verify_evidence_manifest,
    )

    errors: list[str] = []
    if not output_dir.is_dir():
        return ("output_dir missing",)
    catalog = load_evidence_artifact_catalog()
    required = catalog.get("required_artifacts", [])
    for name in required:
        if ".." in Path(str(name)).parts:
            errors.append(f"invalid catalog artifact name: {name}")
            continue
        target = output_dir / name
        if not target.is_file():
            errors.append(f"missing required artifact: {name}")
        elif target.is_symlink():
            errors.append(f"symlink not allowed in closed bundle: {name}")
        else:
            errors.extend(validate_named_artifact(output_dir, name))
    if not verify_evidence_manifest(output_dir):
        errors.append("evidence manifest digest mismatch or missing")
    manifest = output_dir / "evidence_manifest.v1.json"
    if manifest.is_file():
        errors.extend(validate_evidence_manifest(manifest))
    allowed = set(required) | set(catalog.get("optional_artifacts", []))
    allowed.add("evidence_manifest.v1.json")
    try:
        present = list_bundle_files(output_dir) | {"evidence_manifest.v1.json"}
    except ValueError as exc:
        errors.append(str(exc))
        present = set()
    extra = present - allowed
    if extra:
        errors.append(f"unexpected artifacts in closed bundle: {', '.join(sorted(extra))}")
    for name in catalog.get("optional_artifacts", []):
        if name in present:
            errors.extend(validate_named_artifact(output_dir, name))
    report = output_dir / "enterprise_audit_report.json"
    if report.is_file():
        errors.extend(validate_report_semantic_consistency(report))
    return tuple(errors)
