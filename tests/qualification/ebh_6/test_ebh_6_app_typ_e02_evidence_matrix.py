# © Artur Czarnecki. All rights reserved.

"""E02 — per-debt APP-TYP evidence matrix (symbol, gates, recommendation — not operator ratification)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from tests.qualification.ebh_6._contracts_typing_qualification import (
    APPLICATION_SERVICES_TYPING_DEBT,
    ApplicationServicesTypingDebtEntry,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_REPO = Path(__file__).resolve().parents[3]

_POSITIVE_TYPING = (
    "tests/qualification/ebh_6/test_ebh_6_contracts_typing_qualification.py"
    "::test_ebh6_application_services_production_seam_typing_passes"
)
_NEG_UNREGISTERED = (
    "tests/qualification/ebh_6/test_ebh_6_contracts_typing_qualification.py"
    "::test_ebh6_application_services_typing_negative_unregistered_debt_fails"
)
_NEG_STALE = (
    "tests/qualification/ebh_6/test_ebh_6_contracts_typing_qualification.py"
    "::test_ebh6_application_services_typing_negative_stale_debt_detected"
)

_BEHAVIORAL_BY_FINDING: dict[str, tuple[str, ...]] = {
    "EBH6-DEBT-APP-TYP-04": (
        "tests/unit/applications/test_scenario_runtime_profiles.py::test_production_fail_closed_without_manifest",
        "tests/unit/applications/test_scenario_runtime_profiles.py::test_production_success_with_explicit_configuration",
    ),
    "EBH6-DEBT-APP-TYP-05": (
        "tests/unit/runtime/architecture/test_audit_ideal_depth_gate.py::test_audit_ideal_28_4_business_agent_deploy",
    ),
    "EBH6-DEBT-APP-TYP-11": (
        "tests/unit/applications/test_recovery_contract_wiring.py::test_apply_reliability_task_defaults_attaches_recovery_contract",
    ),
}


@dataclass(frozen=True, slots=True)
class AppTypRecommendation:
    finding_id: str
    recommendation: str  # A | B | C


_RECOMMENDATIONS: tuple[AppTypRecommendation, ...] = (
    AppTypRecommendation("EBH6-DEBT-APP-TYP-01", "A"),
    AppTypRecommendation("EBH6-DEBT-APP-TYP-02", "A"),
    AppTypRecommendation("EBH6-DEBT-APP-TYP-03", "A"),
    AppTypRecommendation("EBH6-DEBT-APP-TYP-04", "A"),
    AppTypRecommendation("EBH6-DEBT-APP-TYP-05", "A"),
    AppTypRecommendation("EBH6-DEBT-APP-TYP-06", "A"),
    AppTypRecommendation("EBH6-DEBT-APP-TYP-07", "A"),
    AppTypRecommendation("EBH6-DEBT-APP-TYP-08", "A"),
    AppTypRecommendation("EBH6-DEBT-APP-TYP-09", "A"),
    AppTypRecommendation("EBH6-DEBT-APP-TYP-10", "A"),
    AppTypRecommendation("EBH6-DEBT-APP-TYP-11", "A"),
)


def _resolve_test_node(node_id: str) -> None:
    rel, func = node_id.split("::", 1)
    path = _REPO / rel
    assert path.is_file(), node_id
    text = path.read_text(encoding="utf-8")
    base = func.split("[", 1)[0]
    assert f"def {base}" in text or f"async def {base}" in text, node_id


def _entry_has_symbol_on_disk(entry: ApplicationServicesTypingDebtEntry) -> None:
    path = _REPO / entry.source_path
    assert path.is_file(), entry.finding_id
    text = path.read_text(encoding="utf-8")
    assert f"def {entry.symbol}" in text or f"async def {entry.symbol}" in text, entry.finding_id


@pytest.mark.parametrize("entry", APPLICATION_SERVICES_TYPING_DEBT, ids=lambda e: e.finding_id)
def test_ebh6_app_typ_debt_symbol_registered_on_disk(entry: ApplicationServicesTypingDebtEntry) -> None:
    _entry_has_symbol_on_disk(entry)


def test_ebh6_app_typ_eleven_debt_rows_present() -> None:
    assert len(APPLICATION_SERVICES_TYPING_DEBT) == 11


def test_ebh6_app_typ_shared_positive_and_negative_gates_resolve() -> None:
    for gate in (_POSITIVE_TYPING, _NEG_UNREGISTERED, _NEG_STALE):
        _resolve_test_node(gate)


@pytest.mark.parametrize("entry", APPLICATION_SERVICES_TYPING_DEBT, ids=lambda e: e.finding_id)
def test_ebh6_app_typ_priority_behavioral_gates_when_declared(entry: ApplicationServicesTypingDebtEntry) -> None:
    extra = _BEHAVIORAL_BY_FINDING.get(entry.finding_id, ())
    for gate in extra:
        _resolve_test_node(gate)


def test_ebh6_app_typ_agent_recommendations_not_operator_acceptance() -> None:
    assert len(_RECOMMENDATIONS) == 11
    rec_by_id = {row.finding_id: row.recommendation for row in _RECOMMENDATIONS}
    for entry in APPLICATION_SERVICES_TYPING_DEBT:
        assert rec_by_id[entry.finding_id] in ("A", "B", "C")
        assert rec_by_id[entry.finding_id] != "OPERATOR_APPROVED"
