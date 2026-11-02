# © Artur Czarnecki. All rights reserved.

"""Manifest digest and report artifact helpers."""

from __future__ import annotations

import hashlib
import json
from importlib import resources

from intergrax.platform.enterprise_audit.contracts import ManifestCheck


def canonical_manifest_bytes() -> bytes:
    pkg = resources.files("intergrax.platform.enterprise_audit")
    text = (pkg / "data" / "enterprise_audit_check_manifest.v1.json").read_text(encoding="utf-8")
    data = json.loads(text)
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")


def manifest_digest() -> str:
    return hashlib.sha256(canonical_manifest_bytes()).hexdigest()


def frz_coverage_artifact(checks: tuple[ManifestCheck, ...]) -> dict:
    rows = []
    for check in checks:
        rows.append(
            {
                "check_id": check.check_id,
                "pg_id": check.pg_id,
                "frz_aud": "FRZ-AUD-03",
                "executable": check.entrypoint_kind != "internal_pending",
            }
        )
    return {"schema_version": 1, "rows": rows, "row_count": len(rows)}
