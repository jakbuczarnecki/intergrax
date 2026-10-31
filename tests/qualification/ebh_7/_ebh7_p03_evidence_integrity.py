# © Artur Czarnecki. All rights reserved.

"""Protocol 03 manifest, digest and SHA ladder integrity (structured, fail-closed)."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[3]
_REPLAY = _REPO / "docs/project/maintainers/qualification/ebh_7_p03_independent_replay"
_MANIFEST = _REPLAY / "MANIFEST.json"
_DIGESTS = _REPLAY / "ARTIFACT_DIGESTS.json"
_LADDER = _REPO / "docs/project/maintainers/qualification/EBH_7_SHA_LADDER.md"
_L3_RECORD = _REPLAY / "L3_REPLAY_RECORD.md"


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
    """TEST_EXECUTION_SHA must be an ancestor of HEAD; need not equal doc-only HEAD."""
    manifest = _load_manifest()
    head = current_head_sha()
    test_sha = manifest["TEST_EXECUTION_SHA"]
    impl_sha = manifest["EBH7_CLOSURE_IMPLEMENTATION_SHA"]
    assert isinstance(test_sha, str) and len(test_sha) == 40
    assert isinstance(impl_sha, str) and len(impl_sha) == 40
    assert test_sha == impl_sha, "TEST_EXECUTION_SHA must match immutable implementation baseline"
    assert _git_is_ancestor(test_sha, head), f"TEST_EXECUTION_SHA {test_sha} must be ancestor of current HEAD {head}"
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
    impl = manifest["EBH7_CLOSURE_IMPLEMENTATION_SHA"]
    text = _L3_RECORD.read_text(encoding="utf-8")
    assert impl in text, "L3_REPLAY_RECORD must cite EBH7_CLOSURE_IMPLEMENTATION_SHA / TEST_EXECUTION_SHA"
    ladder = _LADDER.read_text(encoding="utf-8")
    assert impl in ladder, "SHA ladder must pin EBH7_CLOSURE_IMPLEMENTATION_SHA"


def assert_ebh7_p03_evidence_integrity_bundle() -> None:
    assert_ebh7_p03_manifest_role_integrity()
    assert_ebh7_p03_artifact_digests_integrity()
    assert_ebh7_p03_replay_record_matches_implementation()
