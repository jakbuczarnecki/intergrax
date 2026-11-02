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
