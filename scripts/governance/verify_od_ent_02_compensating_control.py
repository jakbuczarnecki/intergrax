#!/usr/bin/env python3
"""Verify OD-ENT-02 Option B compensating control package (draft / Session D gate)."""

from __future__ import annotations

import json
import sys

from scripts.governance.iiac_common import (
    NEGATIVE_TEST_MODULE,
    OD_ENT_02_PACKAGE,
    OD_ENT_02_THREAT,
    REPO,
)

_REQUIRED_PACKAGE_MARKERS = (
    "Option B",
    "OPEN-BP-01",
    "no equivalence",
    "fail-closed",
    "igx_gov ticket",
)

_CI_WORKFLOW = REPO / ".github/workflows/enterprise-audit.yml"


def main() -> int:
    errors: list[str] = []

    if not OD_ENT_02_THREAT.is_file():
        errors.append(f"missing threat model: {OD_ENT_02_THREAT.relative_to(REPO)} (N-05)")
    else:
        threat = OD_ENT_02_THREAT.read_text(encoding="utf-8")
        for token in ("bypass", "residual risk", "OPEN-BP-01"):
            if token.lower() not in threat.lower():
                errors.append(f"threat model missing marker: {token}")

    if not OD_ENT_02_PACKAGE.is_file():
        errors.append(f"missing compensating package: {OD_ENT_02_PACKAGE.relative_to(REPO)}")
    else:
        pkg = OD_ENT_02_PACKAGE.read_text(encoding="utf-8")
        for marker in _REQUIRED_PACKAGE_MARKERS:
            if marker.lower() not in pkg.lower():
                errors.append(f"compensating package missing marker: {marker}")
        if "equivalent to live branch protection" in pkg.lower():
            errors.append("forbidden equivalence claim (N-06)")

    if not _CI_WORKFLOW.is_file():
        errors.append("missing enterprise-audit workflow for CI gate list")
    else:
        wf = _CI_WORKFLOW.read_text(encoding="utf-8")
        if "ent-audit-fast" not in wf or "ent-audit-pr-gate" not in wf:
            errors.append("enterprise-audit.yml missing required ENT job ids")

    if not NEGATIVE_TEST_MODULE.is_file():
        errors.append("missing negative test module test_iiac_negative_guards.py")
    else:
        text = NEGATIVE_TEST_MODULE.read_text(encoding="utf-8")
        for nid in ("N-01", "N-05", "N-08", "N-12"):
            if nid not in text:
                errors.append(f"negative tests missing {nid}")

    sub_manifest = REPO / "docs/project/maintainers/qualification/ent_audit_x/od_ent_02_compensating.v1.json"
    if sub_manifest.is_file():
        try:
            json.loads(sub_manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"od_ent_02_compensating.v1.json invalid: {exc}")

    if errors:
        print("verify_od_ent_02_compensating_control: FAIL", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print("verify_od_ent_02_compensating_control: PASS (package draft)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
