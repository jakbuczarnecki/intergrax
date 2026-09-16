# © Artur Czarnecki. All rights reserved.

"""HARDENING-9 — ``tests/unit`` collection must stay error-free (repository quality gate)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_UNIT_TESTS = "tests/unit"


def test_unit_tests_collect_without_errors() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", _UNIT_TESTS, "--collect-only", "-q"],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    combined = f"{completed.stdout}\n{completed.stderr}"
    assert "errors during collection" not in combined.lower(), combined[-8000:]
    assert completed.returncode in (0, 5), combined[-8000:]


def test_mp4r1_decision_integration_namespace_importable_without_legacy_mp4b() -> None:
    from intergrax.contracts.decision import DecisionSystemIntegrationEngine
    from intergrax.contracts.decision_identity import (
        DecisionId,
        mint_decision_id,
        validate_decision_id,
    )

    decision_id = mint_decision_id()
    assert decision_id.startswith("decision_")
    assert validate_decision_id(decision_id) == decision_id
    assert DecisionId is not None
    assert DecisionSystemIntegrationEngine is not None
