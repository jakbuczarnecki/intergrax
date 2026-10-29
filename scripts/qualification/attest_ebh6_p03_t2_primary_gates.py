# © Artur Czarnecki. All rights reserved.

"""Parse T2 primary gate transcripts and write rollup attestation (no pytest re-run)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_OUT = _REPO / "docs/project/maintainers/qualification/ebh_6_p03_independent_replay"

_GAP_TRANSCRIPT = _OUT / "t2-primary-gates-gap-pytest-transcript.txt"
_QUAL_TRANSCRIPT = _OUT / "pytest-transcript.txt"
_L9_TRANSCRIPT = _OUT / "l9-pytest-transcript.txt"
_ATTESTATION = _OUT / "t2-primary-gates-attestation.json"


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        cwd=_REPO,
    )
    return result.stdout.strip()


def _read_transcript(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16")
    return raw.decode("utf-8")


def main() -> int:
    sys.path.insert(0, str(_REPO))
    from tests.qualification.ebh_6._l9_blocker_execution import parse_pytest_transcript_passed_nodeids
    from tests.qualification.ebh_6._t2_evidence_bundles import all_evidence_bundles

    for path in (_GAP_TRANSCRIPT, _QUAL_TRANSCRIPT, _L9_TRANSCRIPT):
        if not path.is_file():
            print(f"Missing {path}", file=sys.stderr)
            return 1

    sources = {
        _GAP_TRANSCRIPT.name: parse_pytest_transcript_passed_nodeids(_read_transcript(_GAP_TRANSCRIPT)),
        _QUAL_TRANSCRIPT.name: parse_pytest_transcript_passed_nodeids(_read_transcript(_QUAL_TRANSCRIPT)),
        _L9_TRANSCRIPT.name: parse_pytest_transcript_passed_nodeids(_read_transcript(_L9_TRANSCRIPT)),
    }
    passed = set().union(*sources.values())

    pre = json.loads((_OUT / "pre-cert-export.json").read_text(encoding="utf-8"))
    pair_by_id = {row["bundle_id"]: row["pair_count"] for row in pre["bundles"]}
    disp_by_id = {row["bundle_id"]: row["independent_disposition"] for row in pre["bundles"]}

    gates = sorted({g for bundle in all_evidence_bundles() for g in bundle.primary_behavioral_gates if "::" in g})
    primary_rows: list[dict[str, object]] = []
    for nodeid in gates:
        evidence = [name for name, nodeids in sources.items() if nodeid in nodeids]
        primary_rows.append(
            {
                "nodeid": nodeid,
                "status": "PASS" if nodeid in passed else "MISSING_OR_FAIL",
                "evidence_artifacts": evidence,
            },
        )

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
                "all_gates_pass": all(g in passed for g in gate_nodeids),
            },
        )

    manifest = json.loads((_OUT / "MANIFEST.json").read_text(encoding="utf-8"))
    gap_block = manifest.get("executions", {}).get("t2_primary_gates_gap", {})
    gap_nodeids: list[str] = list(gap_block.get("mandatory_nodeids") or [])
    gap_sha = str(gap_block.get("TEST_EXECUTION_SHA") or _git_sha())
    if not gap_nodeids and _ATTESTATION.is_file():
        prior = json.loads(_ATTESTATION.read_text(encoding="utf-8"))
        gap_nodeids = list(prior.get("gap_execution", {}).get("mandatory_nodeids") or [])

    sha = gap_sha
    all_unique_pass = all(row["status"] == "PASS" for row in primary_rows)
    gap_pass = all(nodeid in passed for nodeid in gap_nodeids) if gap_nodeids else False

    payload: dict[str, object] = {
        "TEST_EXECUTION_SHA": sha,
        "aj_group_count": 133,
        "all_unique_primary_gates_pass": all_unique_pass,
        "bundle_count": len(bundle_rows),
        "bundles": bundle_rows,
        "gap_execution": {
            "TEST_EXECUTION_SHA": sha,
            "all_mandatory_pass": gap_pass,
            "artifact": _GAP_TRANSCRIPT.name,
            "mandatory_nodeid_count": len(gap_nodeids),
            "mandatory_nodeids": gap_nodeids,
        },
        "l9_cross_reference": {
            "TEST_EXECUTION_SHA": manifest["executions"]["l9_blocker_gates_b3_b4_b5"]["TEST_EXECUTION_SHA"],
            "artifact": "l9-nodeid-attestation.json",
            "note": "L9 transcript supplies PASS for unit/arch gates absent from ebh_6 qual suite @ e174b5b9",
        },
        "primary_gates": primary_rows,
        "qual_suite_cross_reference": {
            "TEST_EXECUTION_SHA": manifest["executions"]["ebh6_qualification_suite"]["TEST_EXECUTION_SHA"],
            "artifact": _QUAL_TRANSCRIPT.name,
        },
        "unique_primary_gate_count": len(gates),
    }
    _ATTESTATION.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "all_unique_primary_gates_pass": all_unique_pass,
                "gap_all_mandatory_pass": gap_pass,
                "sha": sha,
            },
        ),
    )
    return 0 if all_unique_pass and gap_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
