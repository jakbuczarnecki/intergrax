# © Artur Czarnecki. All rights reserved.

"""Validate final B5 triple replay transcripts and write rollup attestation."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_OUT = _REPO / "docs/project/maintainers/qualification/ebh_6_p03_independent_replay"
_EBH6 = _OUT / "final-b5-ebh6-pytest-transcript.txt"
_L9 = _OUT / "final-b5-l9-pytest-transcript.txt"
_T2 = _OUT / "final-b5-t2-pytest-transcript.txt"
_ROLLUP = _OUT / "final-b5-replay-attestation.json"

_SHA_LINE = re.compile(r"^# TEST_EXECUTION_SHA=([0-9a-f]{40})\s*$")


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        cwd=_REPO,
    )
    return result.stdout.strip()


def _read_transcript(path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8")
    sha = ""
    for line in text.splitlines()[:3]:
        match = _SHA_LINE.match(line.strip())
        if match:
            sha = match.group(1)
            break
    return sha, text


def main() -> int:
    sys.path.insert(0, str(_REPO))
    from tests.qualification.ebh_6._l9_blocker_execution import (
        attest_l9_transcript,
        mandatory_l9_gate_nodeids,
        parse_pytest_transcript_passed_nodeids,
    )
    from tests.qualification.ebh_6._t2_evidence_bundles import all_evidence_bundles

    head = _git_sha()
    missing = [p.name for p in (_EBH6, _L9, _T2) if not p.is_file()]
    if missing:
        print(f"Missing transcripts: {missing}", file=sys.stderr)
        return 1

    eb_sha, eb_text = _read_transcript(_EBH6)
    l9_sha, l9_text = _read_transcript(_L9)
    t2_sha, t2_text = _read_transcript(_T2)
    for label, recorded in (("ebh6", eb_sha), ("l9", l9_sha), ("t2", t2_sha)):
        if recorded != head:
            print(f"{label} transcript SHA {recorded} != HEAD {head}", file=sys.stderr)
            return 1

    eb_passed = parse_pytest_transcript_passed_nodeids(eb_text)
    l9_passed = parse_pytest_transcript_passed_nodeids(l9_text)
    t2_passed = parse_pytest_transcript_passed_nodeids(t2_text)
    all_passed = eb_passed | l9_passed | t2_passed

    l9_mandatory = mandatory_l9_gate_nodeids()
    l9_att = attest_l9_transcript(_L9)
    l9_att["TEST_EXECUTION_SHA"] = head
    l9_att["transcript_artifact"] = _L9.name

    gap_nodeids: list[str] = json.loads(
        (_OUT / "t2-primary-gates-attestation.json").read_text(encoding="utf-8"),
    )["gap_execution"]["mandatory_nodeids"]

    gates = sorted({g for bundle in all_evidence_bundles() for g in bundle.primary_behavioral_gates if "::" in g})
    primary_rows: list[dict[str, object]] = []
    for nodeid in gates:
        sources: list[str] = []
        if nodeid in eb_passed:
            sources.append(_EBH6.name)
        if nodeid in l9_passed:
            sources.append(_L9.name)
        if nodeid in t2_passed:
            sources.append(_T2.name)
        primary_rows.append(
            {
                "nodeid": nodeid,
                "status": "PASS" if nodeid in all_passed else "MISSING_OR_FAIL",
                "evidence_artifacts": sources,
            },
        )

    pre = json.loads((_OUT / "pre-cert-export.json").read_text(encoding="utf-8"))
    disp_by_id = {row["bundle_id"]: row["independent_disposition"] for row in pre["bundles"]}
    pair_by_id = {row["bundle_id"]: row["pair_count"] for row in pre["bundles"]}
    bundle_rows: list[dict[str, object]] = []
    for bundle in all_evidence_bundles():
        bid = bundle.bundle_id.value
        gate_nodeids = [g for g in bundle.primary_behavioral_gates if "::" in g]
        bundle_rows.append(
            {
                "bundle_id": bid,
                "pair_count": pair_by_id[bid],
                "independent_disposition": disp_by_id[bid],
                "primary_behavioral_gates": gate_nodeids,
                "all_gates_pass": all(g in all_passed for g in gate_nodeids),
            },
        )

    summary_line = next((line for line in eb_text.splitlines() if " passed in " in line and line.startswith("=")), "")
    ebh6_pass_count = None
    if " passed in " in summary_line:
        ebh6_pass_count = int(summary_line.split(" passed in ")[0].strip("= ").split()[-1])

    all_unique = all(row["status"] == "PASS" for row in primary_rows)
    gap_ok = all(n in t2_passed for n in gap_nodeids)
    l9_ok = l9_att["all_mandatory_pass"] is True and len(l9_mandatory) == 10

    payload: dict[str, object] = {
        "TEST_EXECUTION_SHA": head,
        "ebh6_qualification": {
            "artifact": _EBH6.name,
            "pytest_pass_count": ebh6_pass_count,
            "pytest_suite": "tests/qualification/ebh_6/",
            "result_summary": summary_line.strip("= ") if summary_line else None,
        },
        "l9_blocker_gates": l9_att,
        "t2_gap": {
            "artifact": _T2.name,
            "mandatory_nodeid_count": len(gap_nodeids),
            "all_mandatory_pass": gap_ok,
            "mandatory_nodeids": gap_nodeids,
        },
        "aj_group_count": 133,
        "bundle_count": 19,
        "all_unique_primary_gates_pass": all_unique,
        "bundles": bundle_rows,
        "primary_gates": primary_rows,
        "unique_primary_gate_count": len(gates),
        "independent_acceptance": "PENDING — auditor B6 only",
    }
    _ROLLUP.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "sha": head,
                "ebh6_pass": ebh6_pass_count,
                "l9_ok": l9_ok,
                "t2_gap_ok": gap_ok,
                "primary_25_ok": all_unique,
                "bundles_19_ok": all(b["all_gates_pass"] for b in bundle_rows),
            },
        ),
    )
    ok = all_unique and gap_ok and l9_ok and eb_sha == head
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
