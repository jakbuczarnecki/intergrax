# © Artur Czarnecki. All rights reserved.

"""B5-F01/F02 adversarial gates for P03 immutable evidence and executable identity."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.qualification.verify_ebh7_p03_immutable_evidence import (
    _DIGESTS_REL,
    _git_show,
    _path_allowed,
    _resolve_replay_path,
    _sha256_bytes,
    verify_executable_tree_identity,
    verify_immutable_digests_at_git_sha,
    verify_manifest_pins,
)
from tests.qualification.ebh_7._constants import EBH7_EVIDENCE_PUBLICATION_SHA, EBH7_TEST_EXECUTION_SHA

_REPO = Path(__file__).resolve().parents[3]
_TEST_SHA = EBH7_TEST_EXECUTION_SHA
_EVIDENCE_SHA = EBH7_EVIDENCE_PUBLICATION_SHA
_REPLAY_SHA = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO, text=True).strip()


def test_b5_path_allowlist_rejects_intergrax_change() -> None:
    assert _path_allowed("docs/project/maintainers/qualification/foo.md")
    assert not _path_allowed("intergrax/runtime/foo.py")


def test_b5_immutable_digests_git_objects_at_replay_head() -> None:
    verify_immutable_digests_at_git_sha(_REPLAY_SHA)


def test_b5_executable_identity_test_vs_replay_head() -> None:
    verify_executable_tree_identity(_TEST_SHA, _REPLAY_SHA)


def test_b5_manifest_pins_match_frozen_roles() -> None:
    verify_manifest_pins(
        test_execution_sha=_TEST_SHA,
        evidence_publication_sha=_EVIDENCE_SHA,
    )


def test_b5_immutable_digests_fail_on_wrong_git_tree_sha() -> None:
    recorded = json.loads(_git_show(_REPLAY_SHA, _DIGESTS_REL))["sha256"]
    mismatched = [
        rel
        for rel, expected in recorded.items()
        if _sha256_bytes(_git_show(_TEST_SHA, _resolve_replay_path(rel))) != expected
    ]
    assert mismatched, "replay-head digests must reject the TEST_EXECUTION_SHA tree bytes"


def test_b5_cli_all_pre_pytest_exit_zero() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/qualification/verify_ebh7_p03_immutable_evidence.py",
            "--mode",
            "all-pre-pytest",
            "--test-execution-sha",
            _TEST_SHA,
            "--evidence-publication-sha",
            _EVIDENCE_SHA,
            "--replay-head-sha",
            _REPLAY_SHA,
        ],
        cwd=_REPO,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_b5_cli_identity_fails_when_replay_equals_test_only_for_docs() -> None:
    """Replay must not silently alias TEST without allowlisted diff only."""
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/qualification/verify_ebh7_p03_immutable_evidence.py",
            "--mode",
            "identity",
            "--test-execution-sha",
            _TEST_SHA,
            "--evidence-publication-sha",
            _EVIDENCE_SHA,
            "--replay-head-sha",
            "0000000000000000000000000000000000000000",
        ],
        cwd=_REPO,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
