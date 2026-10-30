# © Artur Czarnecki. All rights reserved.

"""Phase A — semantic proof-binding audit (authoritative @ HEAD, fail-closed)."""

from __future__ import annotations

from typing import Final

from tests.qualification.ebh_7._cross_domain_scenarios import EBH7_CROSS_DOMAIN_SCENARIOS
from tests.qualification.ebh_7._semantic_disposition_registry import (
    all_verified_dispositions,
    assert_disposition_registry_complete,
    assert_disposition_sha_current,
    assert_skip_equivalents_documented,
    disposition_for,
)
from tests.qualification.ebh_7._semantic_proof_types import ProofBindingAuditRow, ProofBindingResult
from tests.qualification.ebh_7._tenant_e2e_scenarios import EBH7_TENANT_E2E_SCENARIOS

_OPEN_RESULTS: Final[frozenset[ProofBindingResult]] = frozenset(
    {
        ProofBindingResult.MISSING_PROOF,
        ProofBindingResult.BLOCKED,
        ProofBindingResult.PARTIAL_PROOF,
        ProofBindingResult.STRUCTURAL_ONLY,
        ProofBindingResult.MISMATCHED_PROOF,
    }
)


def _scenario_proof_label(module: str | None, function: str | None) -> str:
    if module and function:
        return f"{module}::{function}"
    return "MISSING"


def _validate_disposition_matches_registry() -> None:
    for s in EBH7_TENANT_E2E_SCENARIOS:
        d = disposition_for(s.scenario_id)
        assert d is not None, f"Missing disposition for {s.scenario_id}"
        registered = _scenario_proof_label(s.proof_module, s.proof_function)
        if registered != "MISSING":
            assert d.actual_test == registered, (
                f"{s.scenario_id}: disposition actual_test {d.actual_test!r} != registry {registered!r}"
            )
    for s in EBH7_CROSS_DOMAIN_SCENARIOS:
        d = disposition_for(s.scenario_id)
        assert d is not None, f"Missing disposition for {s.scenario_id}"
        registered = _scenario_proof_label(s.proof_module, s.proof_function)
        if registered != "MISSING":
            assert d.actual_test == registered, (
                f"{s.scenario_id}: disposition actual_test {d.actual_test!r} != registry {registered!r}"
            )


def _build_audit_rows() -> tuple[ProofBindingAuditRow, ...]:
    rows: list[ProofBindingAuditRow] = []
    for d in all_verified_dispositions():
        if d.scenario_id.startswith("EBH7-TEN-E2E-"):
            n = int(d.scenario_id.removeprefix("EBH7-TEN-E2E-"))
            scenario = EBH7_TENANT_E2E_SCENARIOS[n - 1]
            rows.append(
                ProofBindingAuditRow(
                    scenario_id=d.scenario_id,
                    claimed_boundary=scenario.production_path,
                    attack=scenario.attack_failure,
                    expected_behavior=scenario.expected_behavior,
                    existing_proof=d.actual_test,
                    actual_assertions=d.assertions_inspected,
                    semantic_match=d.semantic_match,
                    notes=d.test_execution_evidence,
                )
            )
        elif d.scenario_id.startswith("EBH7-XDM-"):
            n = int(d.scenario_id.removeprefix("EBH7-XDM-"))
            scenario = EBH7_CROSS_DOMAIN_SCENARIOS[n - 1]
            rows.append(
                ProofBindingAuditRow(
                    scenario_id=d.scenario_id,
                    claimed_boundary=scenario.combo,
                    attack=scenario.attack,
                    expected_behavior=scenario.expected,
                    existing_proof=d.actual_test,
                    actual_assertions=d.assertions_inspected,
                    semantic_match=d.semantic_match,
                    notes=d.test_execution_evidence,
                )
            )
    return tuple(rows)


EBH7_PROOF_BINDING_AUDIT: Final[tuple[ProofBindingAuditRow, ...]] = _build_audit_rows()


def audit_row(scenario_id: str) -> ProofBindingAuditRow:
    for row in EBH7_PROOF_BINDING_AUDIT:
        if row.scenario_id == scenario_id:
            return row
    raise KeyError(scenario_id)


def tenant_e2e_semantically_qualified_ids() -> frozenset[str]:
    return frozenset(
        row.scenario_id
        for row in EBH7_PROOF_BINDING_AUDIT
        if row.scenario_id.startswith("EBH7-TEN-E2E") and row.semantic_match is ProofBindingResult.EXACT_PROOF
    )


def cross_domain_semantically_qualified_ids() -> frozenset[str]:
    return frozenset(
        row.scenario_id
        for row in EBH7_PROOF_BINDING_AUDIT
        if row.scenario_id.startswith("EBH7-XDM") and row.semantic_match is ProofBindingResult.EXACT_PROOF
    )


def assert_ebh7_semantic_proof_binding_audit() -> None:
    assert_disposition_registry_complete()
    assert_disposition_sha_current()
    assert_skip_equivalents_documented()
    _validate_disposition_matches_registry()
    mismatched = [r for r in EBH7_PROOF_BINDING_AUDIT if r.semantic_match is ProofBindingResult.MISMATCHED_PROOF]
    for row in mismatched:
        assert disposition_for(row.scenario_id) is not None
    assert len([r for r in EBH7_PROOF_BINDING_AUDIT if r.scenario_id.startswith("EBH7-TEN-E2E")]) == 26
    assert len([r for r in EBH7_PROOF_BINDING_AUDIT if r.scenario_id.startswith("EBH7-XDM")]) == 34


def assert_ebh7_semantic_mandatory_exact_coverage() -> None:
    """B6 readiness — all mandatory scenarios must be EXACT_PROOF (not auto-promoted)."""
    ten_open = tenant_e2e_open_ids_from_audit()
    xdm_open = cross_domain_open_ids_from_audit()
    assert ten_open == () and xdm_open == (), f"EXACT blockers — TEN open {ten_open}; XDM open {xdm_open}"


def tenant_e2e_open_ids_from_audit() -> tuple[str, ...]:
    return tuple(
        sorted(
            row.scenario_id
            for row in EBH7_PROOF_BINDING_AUDIT
            if row.scenario_id.startswith("EBH7-TEN-E2E") and row.semantic_match in _OPEN_RESULTS
        )
    )


def cross_domain_open_ids_from_audit() -> tuple[str, ...]:
    return tuple(
        sorted(
            row.scenario_id
            for row in EBH7_PROOF_BINDING_AUDIT
            if row.scenario_id.startswith("EBH7-XDM") and row.semantic_match in _OPEN_RESULTS
        )
    )


def semantic_exact_counts() -> tuple[int, int, int, int]:
    ten_exact = len(tenant_e2e_semantically_qualified_ids())
    ten_open = len(tenant_e2e_open_ids_from_audit())
    xdm_exact = len(cross_domain_semantically_qualified_ids())
    xdm_open = len(cross_domain_open_ids_from_audit())
    return ten_exact, ten_open, xdm_exact, xdm_open
