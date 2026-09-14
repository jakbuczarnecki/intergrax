# © Artur Czarnecki. All rights reserved.

"""EE-B2-FINAL — current HEAD ancestry to revalidation and frozen EE anchors."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.unit.runtime.architecture._ee_b2_final_facts import (
    ANCESTRY_ANCHORS,
    EE_FINAL_ARCH_COMMIT,
    NPSC5F_REQUALIFICATION_COMMIT,
    REVALIDATION_COMMIT,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[4]


def _is_ancestor(ancestor: str, descendant: str = "HEAD") -> bool:
    proc = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=_REPO,
        check=False,
    )
    return proc.returncode == 0


def test_ee_b2_final_current_head_platform_revalidation_is_ancestor() -> None:
    assert _is_ancestor(REVALIDATION_COMMIT)


def test_ee_b2_final_arch_certification_is_ancestor() -> None:
    assert _is_ancestor(EE_FINAL_ARCH_COMMIT)


def test_ee_b2_final_npsc5f_requalification_is_ancestor() -> None:
    assert _is_ancestor(NPSC5F_REQUALIFICATION_COMMIT)


def test_ee_b2_final_all_ancestry_anchors() -> None:
    missing = [label for label, sha in ANCESTRY_ANCHORS if not _is_ancestor(sha)]
    assert missing == []
