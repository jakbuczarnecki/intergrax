# © Artur Czarnecki. All rights reserved.

"""NPSC-5F Final — frozen baseline Git provenance guards."""

from __future__ import annotations

from pathlib import Path

import pytest

from testing_support.frozen_baseline_provenance import (
    FrozenBaselineProvenanceError,
    assert_frozen_baseline_commit_exists,
    assert_frozen_baseline_is_ancestor_of_remote,
    assert_frozen_baseline_reachable,
)
from testing_support.npsc5f_final_evidence_plane_drift import NPSC5F_FINAL_EVIDENCE_PLANE_BASELINE_SHA

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ORPHAN_W5_H1_LOCAL_SHA = "8879dc8aa6b5be809b3081b61cd5d12b24b6183f"


def test_npsc5f_baseline_commit_exists() -> None:
    assert_frozen_baseline_commit_exists(
        repo_root=_REPO_ROOT,
        baseline_sha=NPSC5F_FINAL_EVIDENCE_PLANE_BASELINE_SHA,
    )


def test_npsc5f_baseline_is_ancestor_of_origin_development() -> None:
    assert_frozen_baseline_is_ancestor_of_remote(
        repo_root=_REPO_ROOT,
        baseline_sha=NPSC5F_FINAL_EVIDENCE_PLANE_BASELINE_SHA,
        remote_ref="origin/development",
    )


def test_npsc5f_baseline_reachable_without_reflog_dependency() -> None:
    assert_frozen_baseline_reachable(
        repo_root=_REPO_ROOT,
        baseline_sha=NPSC5F_FINAL_EVIDENCE_PLANE_BASELINE_SHA,
        remote_ref="origin/development",
    )


def test_npsc5f_invalid_orphan_baseline_not_ancestor_of_origin_development() -> None:
    assert_frozen_baseline_commit_exists(repo_root=_REPO_ROOT, baseline_sha=_ORPHAN_W5_H1_LOCAL_SHA)
    with pytest.raises(FrozenBaselineProvenanceError, match="not an ancestor"):
        assert_frozen_baseline_is_ancestor_of_remote(
            repo_root=_REPO_ROOT,
            baseline_sha=_ORPHAN_W5_H1_LOCAL_SHA,
            remote_ref="origin/development",
        )
