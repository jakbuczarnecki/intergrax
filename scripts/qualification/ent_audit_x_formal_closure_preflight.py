#!/usr/bin/env python3
"""Fail-closed preflight before ENT-AUDIT-X OD-ENT-08 formal closure publication."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_STATUS = _REPO / "docs/project/maintainers/qualification/ent_audit_x/ENT_AUDIT_X_CLOSURE_EXECUTION_STATUS.json"
_LADDER = _REPO / "docs/project/maintainers/qualification/ent_audit_x/ENT_AUDIT_X_SHA_LADDER.md"
_EVOL_STATUS = _REPO / "docs/project/maintainers/qualification/evol_x/EVOL_X_CLOSURE_EXECUTION_STATUS.json"
_ENT_B6_PASS = (
    _REPO
    / "docs/project/maintainers/qualification/ent_audit_x"
    / "ENT_AUDIT_X_PROTOCOL_03_INDEPENDENT_AUDIT_PASS_RECORD.md"
)
_OD_RECORDS = _REPO / "docs/project/maintainers/qualification/ent_audit_x/operator_records"
_ROADMAP = _REPO / "docs/project/maintainers/plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md"
_FRZ = _REPO / "docs/project/maintainers/qualification/PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md"


def _fail(msg: str, errors: list[str]) -> None:
    errors.append(msg)


def _ladder_sha_unset(ladder: str, label: str) -> bool:
    for line in ladder.splitlines():
        if label not in line:
            continue
        if re.search(r"\|\s*(—|-|PENDING|null|unset)\s*\|", line, re.I):
            return True
        if "PENDING" in line.upper():
            return True
    return False


def main() -> int:
    errors: list[str] = []

    if not _ENT_B6_PASS.is_file():
        _fail(f"Missing signed ENT B6 record: {_ENT_B6_PASS.relative_to(_REPO)}", errors)

    if not _STATUS.is_file():
        _fail("Missing ENT closure execution status JSON", errors)
    else:
        st = json.loads(_STATUS.read_text(encoding="utf-8"))
        if st.get("formal_b6_status") != "ACCEPTED":
            _fail("formal_b6_status is not ACCEPTED", errors)
        if not st.get("b6_acceptance_publication_sha"):
            _fail("b6_acceptance_publication_sha unset", errors)
        if not st.get("accepted_baseline_sha"):
            _fail("accepted_baseline_sha unset", errors)
        if st.get("frz_aud_status") == "OPEN":
            _fail("frz_aud_status still OPEN", errors)

    if _EVOL_STATUS.is_file():
        ev = json.loads(_EVOL_STATUS.read_text(encoding="utf-8"))
        formal = (ev.get("formal_closure_status") or "").upper()
        if "CLOSED" not in formal or "INDEPENDENTLY" not in formal:
            _fail("EVOL-X formal closure not independently accepted", errors)

    if not _LADDER.is_file():
        _fail("Missing ENT SHA ladder", errors)
    else:
        ladder = _LADDER.read_text(encoding="utf-8")
        for label in (
            "B6_ACCEPTANCE_PUBLICATION_SHA",
            "ACCEPTED_BASELINE_SHA",
            "FORMAL_CLOSURE_PUBLICATION_SHA",
        ):
            if _ladder_sha_unset(ladder, label):
                _fail(f"SHA ladder {label} still pending/empty", errors)

    for od_id in ("OD-ENT-00", "OD-ENT-01", "OD-ENT-02", "OD-ENT-08"):
        matches = list(_OD_RECORDS.glob(f"*{od_id.lower().replace('-', '_')}*")) if _OD_RECORDS.is_dir() else []
        signed = [p for p in matches if "SIGNED" in p.read_text(encoding="utf-8", errors="replace").upper()]
        if not signed:
            _fail(f"No signed operator record for {od_id} under operator_records/", errors)

    if _ROADMAP.is_file():
        road = _ROADMAP.read_text(encoding="utf-8")
        if "ENT-AUDIT-X — CLOSED — INDEPENDENTLY ACCEPTED — PUBLISHED" in road:
            _fail("Roadmap already claims formal ENT CLOSED (premature if preflight invoked)", errors)

    if _FRZ.is_file():
        frz = _FRZ.read_text(encoding="utf-8")
        for i in range(1, 11):
            tag = f"FRZ-AUD-{i:02d}"
            if tag in frz and "| OPEN |" in frz.split(tag, 1)[-1][:200]:
                _fail(f"{tag} still OPEN in freeze checklist", errors)
                break

    if errors:
        print("ENT-AUDIT-X formal closure preflight: FAIL", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    print("ENT-AUDIT-X formal closure preflight: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
