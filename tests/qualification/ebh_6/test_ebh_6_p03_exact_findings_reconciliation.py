# © Artur Czarnecki. All rights reserved.

"""Protocol 03 exact findings reconciliation — committed replay bundle integrity."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_REPO = Path(__file__).resolve().parents[3]
_BUNDLE = _REPO / "docs/project/maintainers/qualification/ebh_6_p03_independent_replay"
_RECON = _REPO / "docs/project/maintainers/qualification/EBH_6_PROTOCOL_03_EXACT_FINDINGS_RECONCILIATION.md"


def test_ebh6_p03_committed_replay_bundle_present() -> None:
    for name in (
        "MANIFEST.json",
        "p03-handoff.json",
        "pre-cert-export.json",
        "pytest-transcript.txt",
        "l9-pytest-transcript.txt",
        "l9-pytest-transcript-fail-historical-653cdd93f.txt",
        "l9-nodeid-attestation.json",
        "t2-primary-gates-gap-pytest-transcript.txt",
        "t2-primary-gates-attestation.json",
        "final-b5-ebh6-pytest-transcript.txt",
        "final-b5-l9-pytest-transcript.txt",
        "final-b5-t2-pytest-transcript.txt",
        "final-b5-replay-attestation.json",
        "README.md",
    ):
        assert (_BUNDLE / name).is_file(), name


def test_ebh6_p03_committed_handoff_counts() -> None:
    manifest = json.loads((_BUNDLE / "MANIFEST.json").read_text(encoding="utf-8"))
    handoff = json.loads((_BUNDLE / "p03-handoff.json").read_text(encoding="utf-8"))
    assert manifest["aj_group_count"] == 133
    assert manifest["family_count"] == 23
    assert manifest["l9_blocker_count"] == 8
    assert handoff["aj_group_count"] == 133
    assert handoff["family_count"] == 23
    assert "PARTIAL" in handoff["certification_banner"]


def test_ebh6_p03_committed_pre_cert_pair_count() -> None:
    pre = json.loads((_BUNDLE / "pre-cert-export.json").read_text(encoding="utf-8"))
    assert len(pre["pairs"]) == 133
    assert len(pre["bundles"]) == 19


def test_ebh6_p03_exact_findings_reconciliation_doc_covers_findings() -> None:
    text = _RECON.read_text(encoding="utf-8")
    for finding in ("P03-01", "P03-02", "P03-03", "P03-04", "P03-05"):
        assert finding in text


def test_ebh6_p03_manifest_sha_is_ancestor_of_head() -> None:
    manifest = json.loads((_BUNDLE / "MANIFEST.json").read_text(encoding="utf-8"))
    recorded = (
        manifest.get("TEST_EXECUTION_SHA") or manifest["executions"]["ebh6_qualification_suite"]["TEST_EXECUTION_SHA"]
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        cwd=_REPO,
    ).stdout.strip()
    if recorded == head:
        return
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", recorded, head],
        cwd=_REPO,
    )
    assert result.returncode == 0, f"manifest SHA {recorded} must be ancestor of HEAD {head}"


def test_ebh6_p03_l9_execution_attestation_covers_mandatory_nodeids() -> None:
    from tests.qualification.ebh_6._l9_blocker_execution import mandatory_l9_gate_nodeids

    att = json.loads((_BUNDLE / "l9-nodeid-attestation.json").read_text(encoding="utf-8"))
    mandatory = mandatory_l9_gate_nodeids()
    assert att["mandatory_unique_nodeid_count"] == len(mandatory) == 10
    attested = {row["nodeid"]: row["status"] for row in att["results"]}
    assert set(attested) == set(mandatory)


def test_ebh6_p03_l9_execution_manifest_status_honest() -> None:
    manifest = json.loads((_BUNDLE / "MANIFEST.json").read_text(encoding="utf-8"))
    att = json.loads((_BUNDLE / "l9-nodeid-attestation.json").read_text(encoding="utf-8"))
    l9 = manifest["executions"]["l9_blocker_gates_b3_b4_b5"]
    assert l9["status"] in ("COMPLETE", "BLOCKED")
    if att["all_mandatory_pass"]:
        assert l9["status"] == "COMPLETE"
    else:
        assert l9["status"] == "BLOCKED"
        assert att["all_mandatory_pass"] is False


def test_ebh6_p03_t2_primary_gates_attestation_complete() -> None:
    att = json.loads((_BUNDLE / "t2-primary-gates-attestation.json").read_text(encoding="utf-8"))
    manifest = json.loads((_BUNDLE / "MANIFEST.json").read_text(encoding="utf-8"))
    t2 = manifest["executions"]["t2_primary_gates_gap"]
    assert att["unique_primary_gate_count"] == 25
    assert att["bundle_count"] == 19
    assert att["aj_group_count"] == 133
    assert att["all_unique_primary_gates_pass"] is True
    assert att["gap_execution"]["all_mandatory_pass"] is True
    assert t2["status"] == "COMPLETE"
    for bundle in att["bundles"]:
        assert "PENDING" in bundle["independent_disposition"]
        assert bundle["all_gates_pass"] is True


def test_ebh6_p03_final_b5_replay_attestation_complete() -> None:
    att = json.loads((_BUNDLE / "final-b5-replay-attestation.json").read_text(encoding="utf-8"))
    manifest = json.loads((_BUNDLE / "MANIFEST.json").read_text(encoding="utf-8"))
    b5 = manifest["executions"]["final_b5_replay"]
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        cwd=_REPO,
    ).stdout.strip()
    assert att["TEST_EXECUTION_SHA"] == b5["TEST_EXECUTION_SHA"]
    assert att["all_unique_primary_gates_pass"] is True
    assert att["l9_blocker_gates"]["all_mandatory_pass"] is True
    assert att["t2_gap"]["all_mandatory_pass"] is True
    assert att["ebh6_qualification"]["pytest_pass_count"] == 275
    assert b5["status"] == "COMPLETE"
    assert "PENDING" in str(att["independent_acceptance"])
    if head != b5["TEST_EXECUTION_SHA"]:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", b5["TEST_EXECUTION_SHA"], head],
            cwd=_REPO,
        )
        assert result.returncode == 0
