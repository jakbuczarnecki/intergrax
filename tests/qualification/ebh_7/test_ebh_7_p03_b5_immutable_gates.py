# © Artur Czarnecki. All rights reserved.

"""B5-F01/F02 adversarial gates for P03 immutable evidence and executable identity."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from scripts.qualification.verify_ebh7_p03_immutable_evidence import (
    _path_allowed,
    verify_executable_tree_identity,
    verify_immutable_digests_at_git_sha,
    verify_manifest_pins,
)

_REPO = Path(__file__).resolve().parents[3]
_TEST_SHA = "4eb7eb1a89f82215d5356aa7d000198e432777b0"
_EVIDENCE_SHA = "96069c09c2ca5a7a3a9586db336716c9430ca1d7"
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
    with pytest.raises(AssertionError, match="Immutable digest mismatch|git show"):
        verify_immutable_digests_at_git_sha(_TEST_SHA)


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
