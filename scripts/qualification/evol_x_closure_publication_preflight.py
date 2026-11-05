#!/usr/bin/env python3
"""Fail-closed preflight before EVOL-X W7-05 formal closure publication."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_IIAC_ACTIVATION = _REPO / "docs/project/maintainers/qualification/IIAC_POLICY_ACTIVATION_RECORD.json"
_IIAC_EVOL_PASS_GLOB = "EVOL_X_IIAC_INDEPENDENT_AUDIT_PASS_RECORD.md"
_GOVERNANCE = _REPO / "docs/project/maintainers/qualification/evol_x/EVOL_X_AUTO_B6_GOVERNANCE_STATUS.json"
_LADDER = _REPO / "docs/project/maintainers/qualification/evol_x/EVOL_X_SHA_LADDER.md"
_REGISTER = _REPO / "docs/project/maintainers/qualification/evol_x/EVOL_X_OPERATOR_DECISION_REGISTER_W1.md"
_B6_PASS = _REPO / "docs/project/maintainers/qualification/evol_x/EVOL_X_PROTOCOL_03_INDEPENDENT_AUDIT_PASS_RECORD.md"
_B6_DRAFT = (
    _REPO
    / "docs/project/maintainers/qualification/evol_x/b6_technical_qualification"
    / "EVOL_X_PROTOCOL_03_INDEPENDENT_AUDIT_RECORD_DRAFT.md"
)


def _fail(msg: str, errors: list[str]) -> None:
    errors.append(msg)


def _iiac_in_force() -> bool:
    if not _IIAC_ACTIVATION.is_file():
        return False
    gov = json.loads(_IIAC_ACTIVATION.read_text(encoding="utf-8"))
    return str(gov.get("status") or "").upper() == "IN_FORCE"


def main() -> int:
    errors: list[str] = []

    if _iiac_in_force():
        iiac_passes = list((_REPO / "docs/project/maintainers/qualification/evol_x").glob(f"*{_IIAC_EVOL_PASS_GLOB}"))
        if not iiac_passes:
            _fail(
                "IIAC policy IN_FORCE: missing EVOL_X_IIAC_INDEPENDENT_AUDIT_PASS_RECORD.md",
                errors,
            )
        manifest_dir = _REPO / "docs/project/maintainers/qualification/evol_x"
        if not list(manifest_dir.glob("*iiac*manifest*.json")):
            _fail("IIAC policy IN_FORCE: missing EVOL IIAC audit manifest JSON", errors)
    elif not _B6_PASS.is_file():
        _fail(f"Missing signed B6 record: {_B6_PASS.relative_to(_REPO)}", errors)

    if _B6_DRAFT.is_file() and _B6_PASS.is_file():
        if _B6_DRAFT.read_text(encoding="utf-8") == _B6_PASS.read_text(encoding="utf-8"):
            _fail("B6 PASS record must not be identical to unsigned draft", errors)

    if not _GOVERNANCE.is_file():
        _fail("Missing governance JSON", errors)
    else:
        gov = json.loads(_GOVERNANCE.read_text(encoding="utf-8"))
        if gov.get("evol_x_closed") is True and not gov.get("b6_acceptance_publication_sha"):
            _fail("evol_x_closed true without b6_acceptance_publication_sha", errors)
        if not gov.get("b6_acceptance_publication_sha"):
            _fail("b6_acceptance_publication_sha unset in governance JSON", errors)
        if gov.get("ecap_amendment_id") == "ECAP-P4-EVOL-SCOPE-AUTO-B6":
            if gov.get("normative_effect") == "IN_FORCE":
                approvals = gov.get("required_approvals_present") or {}
                if not approvals.get("ecap_maintainer_publication"):
                    _fail("IN_FORCE without maintainer publication flag", errors)
                if not approvals.get("security_review"):
                    _fail("IN_FORCE without security_review flag", errors)
                if not approvals.get("independent_key_custody"):
                    _fail("IN_FORCE without independent_key_custody flag", errors)

    if not _LADDER.is_file():
        _fail("Missing SHA ladder", errors)
    else:
        ladder = _LADDER.read_text(encoding="utf-8")
        for line in ladder.splitlines():
            if "B6_ACCEPTANCE_PUBLICATION_SHA" in line and ("| —" in line or "PENDING" in line):
                _fail("SHA ladder B6_ACCEPTANCE_PUBLICATION_SHA still pending/empty", errors)
                break

    if _REGISTER.is_file():
        reg = _REGISTER.read_text(encoding="utf-8")
        if "**OD-EVOL-12** | **PENDING**" in reg or "OD-EVOL-12** | **PENDING**" in reg:
            _fail("OD-EVOL-12 still PENDING in register", errors)
        if "**OD-EVOLX-08** | **PENDING**" in reg or "OD-EVOLX-08** | **PENDING**" in reg:
            _fail("OD-EVOLX-08 still PENDING in register", errors)
        if "auditor identity NOT NAMED" in reg and "OD-EVOL-11" in reg:
            _fail("OD-EVOL-11 auditor identity still not named", errors)

    if errors:
        print("EVOL-X closure preflight: FAIL", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    print("EVOL-X closure preflight: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
