# © Artur Czarnecki. All rights reserved.

"""UE-11B — real root E2E proof surface and matrix proof-kind gates."""

from __future__ import annotations

from pathlib import Path

import pytest

from testing_support.unified_execution_validation import (
    validate_covered_root_e2e_proof_kind,
    validate_ue_11b_proof_surface,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]


def test_ue_11b_synthetic_e2e_gate_passes() -> None:
    violations = validate_ue_11b_proof_surface(repo_root_path=_REPO_ROOT)
    assert violations == [], "UE-11B synthetic E2E gate violations:\n" + "\n".join(violations)


def test_ue_11b_canonical_entry_gate_passes() -> None:
    violations = [
        message
        for message in validate_ue_11b_proof_surface(repo_root_path=_REPO_ROOT)
        if "missing required token" in message
    ]
    assert violations == []


def test_ue_11b_covered_root_e2e_proof_kind_gate_passes() -> None:
    violations = validate_covered_root_e2e_proof_kind()
    assert violations == [], "UE-11B proof kind gate violations:\n" + "\n".join(violations)
