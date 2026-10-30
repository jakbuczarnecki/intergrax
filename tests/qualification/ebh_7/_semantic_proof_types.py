# © Artur Czarnecki. All rights reserved.

"""EBH-7 semantic proof-binding types (fail-closed audit)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ProofBindingResult(StrEnum):
    EXACT_PROOF = "EXACT_PROOF"
    PARTIAL_PROOF = "PARTIAL_PROOF"
    STRUCTURAL_ONLY = "STRUCTURAL_ONLY"
    MISMATCHED_PROOF = "MISMATCHED_PROOF"
    MISSING_PROOF = "MISSING_PROOF"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True, slots=True)
class VerifiedSemanticDisposition:
    scenario_id: str
    required_invariant: str
    production_boundaries: str
    attack: str
    actual_test: str
    assertions_inspected: str
    semantic_match: ProofBindingResult
    test_execution_evidence: str
    executed_equivalent_node_id: str | None = None


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
