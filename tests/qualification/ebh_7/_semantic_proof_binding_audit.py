# © Artur Czarnecki. All rights reserved.

"""Phase A — semantic proof-binding audit (authoritative @ HEAD)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from tests.qualification.ebh_7._cross_domain_scenarios import EBH7_CROSS_DOMAIN_SCENARIOS
from tests.qualification.ebh_7._tenant_e2e_scenarios import EBH7_TENANT_E2E_SCENARIOS


class ProofBindingResult(StrEnum):
    EXACT_PROOF = "EXACT_PROOF"
    PARTIAL_PROOF = "PARTIAL_PROOF"
    STRUCTURAL_ONLY = "STRUCTURAL_ONLY"
    MISMATCHED_PROOF = "MISMATCHED_PROOF"
    MISSING_PROOF = "MISSING_PROOF"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True, slots=True)
class ProofBindingAuditRow:
    scenario_id: str
    claimed_boundary: str
    attack: str
    expected_behavior: str
    existing_proof: str
    actual_assertions: str
    semantic_match: ProofBindingResult
    notes: str


_TEN_AUDIT_OVERRIDES: Final[dict[str, tuple[ProofBindingResult, str, str]]] = {}


_XDM_AUDIT_OVERRIDES: Final[dict[str, tuple[ProofBindingResult, str, str]]] = {}


def _proof_label(module: str | None, function: str | None) -> str:
    if module and function:
        return f"{module}::{function}"
    return "MISSING"


def _audit_tenant_rows() -> tuple[ProofBindingAuditRow, ...]:
    rows: list[ProofBindingAuditRow] = []
    for s in EBH7_TENANT_E2E_SCENARIOS:
        proof = _proof_label(s.proof_module, s.proof_function)
        override = _TEN_AUDIT_OVERRIDES.get(s.scenario_id)
        if override is not None:
            match, assertions, notes = override
        elif s.proof_module is None:
            match = ProofBindingResult.MISSING_PROOF
            assertions = "none"
            notes = "No executable proof node registered"
        else:
            match = ProofBindingResult.EXACT_PROOF
            assertions = s.assertions
            notes = s.reuse_program
        rows.append(
            ProofBindingAuditRow(
                scenario_id=s.scenario_id,
                claimed_boundary=s.production_path,
                attack=s.attack_failure,
                expected_behavior=s.expected_behavior,
                existing_proof=proof,
                actual_assertions=assertions,
                semantic_match=match,
                notes=notes,
            )
        )
    return tuple(rows)


def _audit_xdm_rows() -> tuple[ProofBindingAuditRow, ...]:
    rows: list[ProofBindingAuditRow] = []
    for s in EBH7_CROSS_DOMAIN_SCENARIOS:
        proof = _proof_label(s.proof_module, s.proof_function)
        override = _XDM_AUDIT_OVERRIDES.get(s.scenario_id)
        if override is not None:
            match, assertions, notes = override
        elif s.proof_module is None:
            match = ProofBindingResult.MISSING_PROOF
            assertions = "none"
            notes = "No executable proof node registered"
        else:
            match = ProofBindingResult.EXACT_PROOF
            assertions = s.expected
            notes = s.reuse_program
        rows.append(
            ProofBindingAuditRow(
                scenario_id=s.scenario_id,
                claimed_boundary=s.combo,
                attack=s.attack,
                expected_behavior=s.expected,
                existing_proof=proof,
                actual_assertions=assertions,
                semantic_match=match,
                notes=notes,
            )
        )
    return tuple(rows)


EBH7_PROOF_BINDING_AUDIT: Final[tuple[ProofBindingAuditRow, ...]] = _audit_tenant_rows() + _audit_xdm_rows()


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
    mismatched = [r for r in EBH7_PROOF_BINDING_AUDIT if r.semantic_match is ProofBindingResult.MISMATCHED_PROOF]
    assert mismatched == [], f"MISMATCHED_PROOF bindings: {[m.scenario_id for m in mismatched]}"
    assert len([r for r in EBH7_PROOF_BINDING_AUDIT if r.scenario_id.startswith("EBH7-TEN-E2E")]) == 26
    assert len([r for r in EBH7_PROOF_BINDING_AUDIT if r.scenario_id.startswith("EBH7-XDM")]) == 34


def tenant_e2e_open_ids_from_audit() -> tuple[str, ...]:
    open_results = {
        ProofBindingResult.MISSING_PROOF,
        ProofBindingResult.BLOCKED,
        ProofBindingResult.PARTIAL_PROOF,
        ProofBindingResult.STRUCTURAL_ONLY,
    }
    return tuple(
        sorted(
            row.scenario_id
            for row in EBH7_PROOF_BINDING_AUDIT
            if row.scenario_id.startswith("EBH7-TEN-E2E") and row.semantic_match in open_results
        )
    )


def cross_domain_open_ids_from_audit() -> tuple[str, ...]:
    open_results = {
        ProofBindingResult.MISSING_PROOF,
        ProofBindingResult.BLOCKED,
        ProofBindingResult.PARTIAL_PROOF,
        ProofBindingResult.STRUCTURAL_ONLY,
    }
    return tuple(
        sorted(
            row.scenario_id
            for row in EBH7_PROOF_BINDING_AUDIT
            if row.scenario_id.startswith("EBH7-XDM") and row.semantic_match in open_results
        )
    )
