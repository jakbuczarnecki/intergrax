# © Artur Czarnecki. All rights reserved.

"""Protocol 03 manifest, digest and SHA ladder integrity (structured, fail-closed)."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[3]
_REPLAY = _REPO / "docs/project/maintainers/qualification/ebh_7_p03_independent_replay"
_MANIFEST = _REPLAY / "MANIFEST.json"
_DIGESTS = _REPLAY / "ARTIFACT_DIGESTS.json"
_LADDER = _REPO / "docs/project/maintainers/qualification/EBH_7_SHA_LADDER.md"
_L3_RECORD = _REPLAY / "L3_REPLAY_RECORD.md"
_CONSISTENCY = _REPLAY / "B6_EVIDENCE_CONSISTENCY_MATRIX.md"


def _load_manifest() -> dict[str, Any]:
    return json.loads(_MANIFEST.read_text(encoding="utf-8"))


def _git_is_ancestor(ancestor: str, head: str) -> bool:
    proc = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, head],
        cwd=_REPO,
        capture_output=True,
    )
    return proc.returncode == 0


def current_head_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO, text=True).strip()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def assert_ebh7_p03_manifest_role_integrity() -> None:
    """Protocol 03 SHA roles — TEST_EXECUTION runs on B6 corrective tree, not pre-B6 closure alone."""
    manifest = _load_manifest()
    head = current_head_sha()
    test_sha = manifest["TEST_EXECUTION_SHA"]
    closure_sha = manifest["EBH7_CLOSURE_IMPLEMENTATION_SHA"]
    b6_sha = manifest["EBH7_B6_CORRECTIVE_IMPLEMENTATION_SHA"]
    for label, value in (
        ("TEST_EXECUTION_SHA", test_sha),
        ("EBH7_CLOSURE_IMPLEMENTATION_SHA", closure_sha),
        ("EBH7_B6_CORRECTIVE_IMPLEMENTATION_SHA", b6_sha),
    ):
        assert isinstance(value, str) and len(value) == 40, label
    assert test_sha == b6_sha or _git_is_ancestor(b6_sha, test_sha), (
        "TEST_EXECUTION_SHA must equal or descend from EBH7_B6_CORRECTIVE_IMPLEMENTATION_SHA"
    )
    assert test_sha != closure_sha or b6_sha == closure_sha, (
        "When B6 corrective exists, TEST_EXECUTION_SHA must not silently alias pre-B6 closure only"
    )
    assert _git_is_ancestor(closure_sha, test_sha), (
        f"EBH7_CLOSURE_IMPLEMENTATION_SHA {closure_sha} must be ancestor of TEST_EXECUTION_SHA {test_sha}"
    )
    assert _git_is_ancestor(test_sha, head), f"TEST_EXECUTION_SHA {test_sha} must be ancestor of HEAD {head}"
    evidence_sha = manifest["EVIDENCE_PUBLICATION_SHA"]
    assert isinstance(evidence_sha, str) and len(evidence_sha) == 40
    assert _git_is_ancestor(evidence_sha, head), f"EVIDENCE_PUBLICATION_SHA {evidence_sha} must be ancestor of HEAD"
    assert _git_is_ancestor(test_sha, evidence_sha) or evidence_sha == test_sha, (
        "Evidence publication must be at or after TEST_EXECUTION_SHA"
    )


def assert_ebh7_p03_artifact_digests_integrity() -> None:
    if not _DIGESTS.is_file():
        raise AssertionError("ARTIFACT_DIGESTS.json missing — run render_ebh7_p03_artifact_digests.py")
    payload = json.loads(_DIGESTS.read_text(encoding="utf-8"))
    recorded: dict[str, str] = payload.get("sha256", {})
    for rel, expected in recorded.items():
        path = (_REPLAY / rel).resolve()
        assert path.is_file(), f"Digest artifact missing: {rel}"
        actual = _sha256_file(path)
        assert actual == expected, f"Digest mismatch for {rel}"


def assert_ebh7_p03_replay_record_matches_implementation() -> None:
    manifest = _load_manifest()
    test_sha = manifest["TEST_EXECUTION_SHA"]
    closure_sha = manifest["EBH7_CLOSURE_IMPLEMENTATION_SHA"]
    b6_sha = manifest["EBH7_B6_CORRECTIVE_IMPLEMENTATION_SHA"]
    text = _L3_RECORD.read_text(encoding="utf-8")
    ladder = _LADDER.read_text(encoding="utf-8")
    for sha in (test_sha, closure_sha, b6_sha):
        assert sha in text, f"L3_REPLAY_RECORD must cite {sha}"
        assert sha in ladder, f"SHA ladder must pin {sha}"


def assert_ebh7_p03_semantic_outcomes_match_ssot() -> None:
    from tests.qualification.ebh_7._rpl_h_normative_evidence import normative_rpl_b6_open_concern_ids
    from tests.qualification.ebh_7._rpl_h_semantic_evidence import RplSemanticVerdict, rpl_semantic_verdict_counts
    from tests.qualification.ebh_7._semantic_proof_binding_audit import semantic_exact_counts

    ten_exact, ten_open, xdm_exact, xdm_open = semantic_exact_counts()
    assert ten_exact == 26 and ten_open == 0, f"TEN-E2E SSOT {ten_exact}/{ten_open}"
    assert xdm_exact == 34 and xdm_open == 0, f"XDM SSOT {xdm_exact}/{xdm_open}"
    agent = rpl_semantic_verdict_counts()
    assert agent.get(RplSemanticVerdict.QUALIFIED, 0) == 56 and sum(agent.values()) == 56
    assert normative_rpl_b6_open_concern_ids() == ()


def assert_ebh7_p03_consistency_matrix_matches_ssot() -> None:
    if not _CONSISTENCY.is_file():
        raise AssertionError("B6_EVIDENCE_CONSISTENCY_MATRIX.md missing")
    manifest = _load_manifest()
    text = _CONSISTENCY.read_text(encoding="utf-8")
    test_sha = manifest["TEST_EXECUTION_SHA"]
    assert test_sha in text
    assert "26" in text and "34" in text and "56" in text
    assert "NOT ISSUED" in text, "Matrix must not claim independent B6 PASS"
    last = manifest["pytest_profile"]["last_replay"]
    passed = last["passed"]
    assert str(passed) in text, "Matrix must record L3 passed count from MANIFEST"
    assert re.search(r"TEN-E2E.*26.*EXACT.*0.*OPEN", text, re.I | re.S), "Matrix TEN row"
    assert re.search(r"XDM.*34.*EXACT.*0.*OPEN", text, re.I | re.S), "Matrix XDM row"


def assert_ebh7_p03_evidence_integrity_bundle() -> None:
    assert_ebh7_p03_manifest_role_integrity()
    assert_ebh7_p03_artifact_digests_integrity()
    assert_ebh7_p03_replay_record_matches_implementation()
    assert_ebh7_p03_semantic_outcomes_match_ssot()
    assert_ebh7_p03_consistency_matrix_matches_ssot()
