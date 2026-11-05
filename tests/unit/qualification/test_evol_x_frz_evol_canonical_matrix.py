# © Artur Czarnecki. All rights reserved.

"""Mechanical guard: FRZ-EVOL audit matrices use checklist-literal requirements."""

from __future__ import annotations

from scripts.qualification.evol_x_frz_evol_canonical_matrix import (
    FRZ_EVOL_CANONICAL_REQUIREMENTS,
    validate_frz_evol_id_set,
    validate_requirement_labels,
)

pytestmark = [__import__("pytest").mark.unit]


def test_frz_evol_canonical_matrix_covers_ten_ids() -> None:
    assert len(FRZ_EVOL_CANONICAL_REQUIREMENTS) == 10
    assert validate_frz_evol_id_set(set(FRZ_EVOL_CANONICAL_REQUIREMENTS)) == []


def test_r2_reconciliation_matrix_labels() -> None:
    """Labels published in R2 global reconciliation (must stay aligned with checklist)."""
    labels = dict(FRZ_EVOL_CANONICAL_REQUIREMENTS)
    assert validate_requirement_labels(labels) == []
