# © Artur Czarnecki. All rights reserved.

"""EBH-6 — FRZ-TYP behavioral gates for contracts_typing bundle (T2-AUD)."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.qualification.ebh_6._closed_world_discovery import discover_contracts_surface_paths
from tests.qualification.ebh_6._contracts_typing_qualification import (
    APPLICATION_SERVICES_TYPING_DEBT,
    ApplicationServicesTypingDebtEntry,
    discover_production_effective_application_shared_modules,
    evaluate_application_services_production_seam_typing,
    evaluate_application_services_typing_on_source,
    evaluate_ebh6_contracts_layer_surface_purity,
    format_application_services_typing_failure,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_REPO_ROOT = Path(__file__).resolve().parents[3]

_CONTRACTS_GATE = (
    "tests/qualification/ebh_6/test_ebh_6_contracts_typing_qualification.py"
    "::test_ebh6_contracts_layer_frz_typ_surface_purity_passes"
)
_APP_SERVICES_GATE = (
    "tests/qualification/ebh_6/test_ebh_6_contracts_typing_qualification.py"
    "::test_ebh6_application_services_production_seam_typing_passes"
)


def test_ebh6_contracts_layer_closed_world_surface_non_empty() -> None:
    assert len(discover_contracts_surface_paths()) >= 10


def test_ebh6_application_services_production_closure_non_empty() -> None:
    modules = discover_production_effective_application_shared_modules()
    assert len(modules) >= 50


def test_ebh6_contracts_layer_frz_typ_surface_purity_passes() -> None:
    passed, detail = evaluate_ebh6_contracts_layer_surface_purity(_REPO_ROOT)
    assert passed, detail


def test_ebh6_application_services_production_seam_typing_passes() -> None:
    result = evaluate_application_services_production_seam_typing(_REPO_ROOT)
    assert result.passed, format_application_services_typing_failure(result)


def test_ebh6_application_services_typing_debt_registry_covers_head() -> None:
    assert len(APPLICATION_SERVICES_TYPING_DEBT) == 11
    result = evaluate_application_services_production_seam_typing(_REPO_ROOT)
    assert not result.unregistered_violations
    assert not result.stale_debt_entries


def test_ebh6_contracts_typing_negative_untyped_public_wiring_function() -> None:
    source = """
def build_runtime(host: object) -> object:
    return host
"""
    violations = evaluate_application_services_typing_on_source(
        source=source,
        source_path="intergrax/applications/_shared/snippet_wiring.py",
        source_module="intergrax.applications._shared.snippet_wiring",
    )
    assert any(v.rule_id == "generic_object_public_seam" for v in violations)


def test_ebh6_contracts_typing_negative_missing_return_annotation() -> None:
    source = """
def wire_application_services():
    return None
"""
    violations = evaluate_application_services_typing_on_source(
        source=source,
        source_path="intergrax/applications/_shared/snippet_wiring.py",
        source_module="intergrax.applications._shared.snippet_wiring",
    )
    assert any(v.rule_id == "missing_public_return_annotation" for v in violations)


def test_ebh6_application_services_typing_negative_wrong_rule_id_not_covered_by_debt() -> None:
    entry = APPLICATION_SERVICES_TYPING_DEBT[0]
    wrong_rule_debt = (
        ApplicationServicesTypingDebtEntry(
            finding_id="EBH6-DEBT-APP-TYP-PROBE-RULE",
            source_path=entry.source_path,
            line=2,
            rule_id="semantic_any_public_seam",
            symbol=entry.symbol,
            rationale="wrong rule_id must not mask generic_object violation",
        ),
    )
    result = evaluate_application_services_production_seam_typing(
        _REPO_ROOT,
        debt_entries=wrong_rule_debt,
    )
    assert result.unregistered_violations
    assert any(v.rule_id == "generic_object_public_seam" for v in result.unregistered_violations)


def test_ebh6_application_services_typing_negative_duplicate_finding_id_rejected() -> None:
    dup = (
        APPLICATION_SERVICES_TYPING_DEBT[0],
        ApplicationServicesTypingDebtEntry(
            finding_id=APPLICATION_SERVICES_TYPING_DEBT[0].finding_id,
            source_path="intergrax/applications/_shared/other.py",
            line=1,
            rule_id="generic_object_public_seam",
            symbol="other",
            rationale="duplicate id probe",
        ),
    )
    result = evaluate_application_services_production_seam_typing(_REPO_ROOT, debt_entries=dup)
    assert result.registry_validation_errors


def test_ebh6_application_services_typing_negative_unregistered_debt_fails() -> None:
    result = evaluate_application_services_production_seam_typing(
        _REPO_ROOT,
        debt_entries=(),
    )
    assert not result.passed
    assert result.unregistered_violations


def test_ebh6_application_services_typing_negative_stale_debt_detected() -> None:
    stale = ApplicationServicesTypingDebtEntry(
        finding_id="EBH6-DEBT-APP-TYP-STALE",
        source_path="intergrax/applications/_shared/no_such.py",
        line=1,
        rule_id="generic_object_public_seam",
        symbol="never",
        rationale="stale probe",
    )
    result = evaluate_application_services_production_seam_typing(
        _REPO_ROOT,
        debt_entries=APPLICATION_SERVICES_TYPING_DEBT + (stale,),
    )
    assert not result.passed
    assert result.stale_debt_entries
