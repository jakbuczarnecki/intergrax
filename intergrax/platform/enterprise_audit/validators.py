# © Artur Czarnecki. All rights reserved.

"""Lightweight structural validators for ENT-AUDIT-X report artifacts."""

from __future__ import annotations

import json
from pathlib import Path

_SCHEMA_DIR = Path(__file__).resolve().parent / "data" / "schemas"


def _load_schema(name: str) -> dict:
    return json.loads((_SCHEMA_DIR / name).read_text(encoding="utf-8"))


def validate_against_schema(payload: dict, schema: dict) -> tuple[str, ...]:
    errors: list[str] = []
    required = schema.get("required", [])
    for key in required:
        if key not in payload:
            errors.append(f"missing required key: {key}")
    props = schema.get("properties", {})
    for key, spec in props.items():
        if key not in payload:
            continue
        expected = spec.get("type")
        value = payload[key]
        if expected == "string" and not isinstance(value, str):
            errors.append(f"{key}: expected string")
        elif expected == "integer" and not isinstance(value, int):
            errors.append(f"{key}: expected integer")
        elif expected == "array" and not isinstance(value, list):
            errors.append(f"{key}: expected array")
        elif expected == "object" and not isinstance(value, dict):
            errors.append(f"{key}: expected object")
    return tuple(errors)


def validate_report_json(path: Path) -> tuple[str, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return validate_against_schema(payload, _load_schema("enterprise_audit_report.v1.schema.json"))


def validate_evidence_manifest(path: Path) -> tuple[str, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return validate_against_schema(payload, _load_schema("evidence_manifest.v1.schema.json"))


def validate_vbn_domains(path: Path) -> tuple[str, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return validate_against_schema(payload, _load_schema("vbn_domains.v1.schema.json"))


def verify_closed_evidence_bundle(output_dir: Path) -> tuple[str, ...]:
    from intergrax.platform.enterprise_audit.evidence import (
        list_bundle_files,
        load_evidence_artifact_catalog,
        verify_evidence_manifest,
    )

    errors: list[str] = []
    if not output_dir.is_dir():
        return ("output_dir missing",)
    try:
        output_dir.resolve().relative_to(output_dir.resolve())
    except ValueError:
        errors.append("output_dir must be a normalized directory path")
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
    if not verify_evidence_manifest(output_dir):
        errors.append("evidence manifest digest mismatch or missing")
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
    return tuple(errors)
