# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.runtime.evidence.core_certification_spec import CoreCertificationLevel
from intergrax.runtime.evidence.scenario_contracts import (
    CORE_SCENARIO_CONTRACTS,
    core_scenario_contracts_for_level,
    get_core_scenario_contract,
    validate_core_scenario_catalog,
)

pytestmark = pytest.mark.unit

_EVIDENCE_ROOT = Path(__file__).resolve().parents[3] / "intergrax" / "runtime" / "evidence"


def test_validate_core_scenario_catalog_passes() -> None:
    validate_core_scenario_catalog()


def test_catalog_has_twelve_unique_scenarios() -> None:
    ids = [contract.scenario_id for contract in CORE_SCENARIO_CONTRACTS]
    assert len(ids) == 12
    assert len(set(ids)) == 12


def test_no_scenario_requires_network_or_real_llm() -> None:
    for contract in CORE_SCENARIO_CONTRACTS:
        assert contract.requires_network is False
        assert contract.requires_real_llm is False
        assert contract.deterministic_by_contract is True
        assert contract.required_evidence_kinds


def test_certification_report_emitted_in_core_l1() -> None:
    l1_contracts = core_scenario_contracts_for_level(CoreCertificationLevel.L1)
    ids = {contract.scenario_id for contract in l1_contracts}
    assert "certification_report_emitted" in ids
    report_contract = get_core_scenario_contract("certification_report_emitted")
    assert report_contract is not None
    assert report_contract.min_level is CoreCertificationLevel.L1


def test_core_scenario_contracts_for_level_counts() -> None:
    assert len(core_scenario_contracts_for_level("CORE-L1")) == 4
    assert len(core_scenario_contracts_for_level("l2")) == 8
    assert len(core_scenario_contracts_for_level("CORE-L3")) == 12


def test_get_core_scenario_contract_unknown_returns_none() -> None:
    assert get_core_scenario_contract("nonexistent_scenario") is None


def test_evidence_modules_have_no_applications_or_agents_imports() -> None:
    forbidden = ("applications.", "agents.", "from applications", "from agents")
    for path in _EVIDENCE_ROOT.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in source, f"{path.name} contains forbidden import token: {token}"
