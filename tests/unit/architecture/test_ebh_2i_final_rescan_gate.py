# © Artur Czarnecki. All rights reserved.

"""EBH-2I — final cross-subsystem public-contract rescan invariants."""

from __future__ import annotations

from pathlib import Path

import pytest

from testing_support.architecture.public_contract_boundary import (
    evaluate_public_contract_dependency_boundary,
    format_gate_failure,
)
from testing_support.architecture.public_contract_boundary.debt_registry import (
    PUBLIC_CONTRACT_DEPENDENCY_DEBT,
)
from testing_support.architecture.public_contract_boundary.models import RemovalStage

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]

_CLOSED_EBH_2_STAGES = frozenset(
    {
        RemovalStage.EBH_2E,
        RemovalStage.EBH_2F,
        RemovalStage.EBH_2G,
        RemovalStage.EBH_2H,
    }
)


def test_ebh_2i_no_expired_stage_debt_entries() -> None:
    expired = [
        entry
        for entry in PUBLIC_CONTRACT_DEPENDENCY_DEBT
        if entry.removal_stage in _CLOSED_EBH_2_STAGES
    ]
    assert not expired, "Expired EBH-2E/F/G/H debt entries must be reconciled: " + ", ".join(
        entry.finding_id for entry in expired
    )


def test_ebh_2i_public_contract_boundary_gate_passes() -> None:
    result = evaluate_public_contract_dependency_boundary(_REPO_ROOT)
    assert result.passed, format_gate_failure(result)
