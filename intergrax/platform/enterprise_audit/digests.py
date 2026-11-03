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


def _frz_for_check(check: ManifestCheck) -> tuple[str, ...]:
    cid = check.check_id
    if cid.startswith("TEN-"):
        return ("FRZ-TEN-*",)
    if cid.startswith("VBN-") or cid.startswith("ENT-A-VBN-"):
        return ("FRZ-RPL-*", "FRZ-CFG-*")
    if cid.startswith("ENT-A-BND-"):
        return ("FRZ-BND-*",)
    if cid.startswith("ENT-A-OWN-"):
        return ("FRZ-OWN-*",)
    if cid.startswith("ENT-A-CTR-"):
        return ("FRZ-CTR-*",)
    if cid.startswith("ENT-A-TYP-"):
        return ("FRZ-TYP-*",)
    if cid.startswith("ENT-A-EVOL-") or cid.startswith("ENT-A-INV-"):
        return ("FRZ-EVOL-*",)
    if cid.startswith("ENT-A-TEN-"):
        return ("FRZ-TEN-*",)
    if cid.startswith("ENT-A-REG-"):
        return ("FRZ-REG-*",)
    if cid.startswith("ENT-A-DOC-"):
        return ("FRZ-DOC-*",)
    if cid.startswith("ENT-A-PRD-"):
        return ("FRZ-PRD-*",)
    return ("FRZ-AUD-03",)


def frz_coverage_artifact(checks: tuple[ManifestCheck, ...]) -> dict:
    rows = []
    for check in checks:
        frz = _frz_for_check(check)
        rows.append(
            {
                "check_id": check.check_id,
                "pg_id": check.pg_id,
                "frz_aud": frz[0],
                "frz_mapping": list(frz),
                "executable": check.entrypoint_kind != "internal_pending",
            }
        )
    return {"schema_version": 1, "rows": rows, "row_count": len(rows)}
