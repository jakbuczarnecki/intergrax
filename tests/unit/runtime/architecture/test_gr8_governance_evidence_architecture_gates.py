# © Artur Czarnecki. All rights reserved.

"""GR-8 architecture gates — evidence boundary and authority separation."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


def test_gr8_governance_core_imports_persistence_contract_not_concrete_store() -> None:
    source = _read("intergrax/runtime/governance/root_execution_authority_admission.py")
    assert "GovernanceEvidenceRecorder" in source
    assert "sqlite" not in source.lower()
    assert "EvidencePersistencePort" not in source


def test_gr8_mse_boundary_uses_recorder_not_global_sink() -> None:
    source = _read("intergrax/runtime/policy/meaningful_side_effect_authorization.py")
    assert "GovernanceEvidenceRecorder" in source
    assert "GLOBAL" not in source
    assert "get_global" not in source


def test_gr8_public_contract_in_intergrax_contracts() -> None:
    path = REPO_ROOT / "intergrax/contracts/governed_execution_governance_evidence.py"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "GovernanceEvidencePersistencePort" in text
    assert "PolicyDecision" in text
    assert "def persist" in text


def test_gr8_persistence_outcome_not_policy_decision() -> None:
    text = _read("intergrax/contracts/governed_execution_governance_evidence.py")
    assert "class GovernanceEvidencePersistenceOutcome" in text
    assert "PolicyDecision" not in text.split("class GovernanceEvidencePersistenceOutcome")[1].split(
        "class ", 1
    )[0]
