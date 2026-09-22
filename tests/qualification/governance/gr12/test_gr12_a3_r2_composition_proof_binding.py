# © Artur Czarnecki. All rights reserved.

"""GR-12-A3-R2 — composition qualification proof binding reconciliation gates."""

from __future__ import annotations

import pytest

from tests.qualification.governance.gr12.a3_path_qualifications import (
    GR12_A3_COMPOSITION_PROOF_SSOT,
    GR12_A3_CORE_PATH_PROOFS,
    Gr12CompositionDomain,
    Gr12QualificationPathKind,
)
from tests.qualification.governance.gr12.qualification_support import (
    assert_gr12_a3_composition_proof_structural_integrity,
    assert_gr12_a3_path_semantic_integrity,
    assert_proof_nodes_registered,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_gr12_a3_r2_composition_ssot_covers_both_paths() -> None:
    assert set(GR12_A3_COMPOSITION_PROOF_SSOT) == {
        "CP-HOST-BOUNDARY-OPTIONAL",
        "CP-ECP-BOUNDARY-OPTIONAL",
    }


def test_gr12_a3_r2_composition_proof_nodes_match_ssot_and_register_tests() -> None:
    for path_id, bindings in GR12_A3_COMPOSITION_PROOF_SSOT.items():
        bundle = next(row for row in GR12_A3_CORE_PATH_PROOFS if row.path_id == path_id)
        assert bundle.kind is Gr12QualificationPathKind.COMPOSITION_SURFACE
        assert_gr12_a3_composition_proof_structural_integrity(bundle)
        assert_proof_nodes_registered(tuple(test_id for test_id, _ in bindings.values()))


def test_gr12_a3_r2_composition_domains_do_not_cross_paths() -> None:
    for path_id, bindings in GR12_A3_COMPOSITION_PROOF_SSOT.items():
        expected = (
            Gr12CompositionDomain.HOST_TASK_CONTROL
            if path_id == "CP-HOST-BOUNDARY-OPTIONAL"
            else Gr12CompositionDomain.ECP
        )
        for _test_id, domain in bindings.values():
            assert domain is expected, path_id


def test_gr12_a3_r2_all_paths_remain_semantically_consistent() -> None:
    for bundle in GR12_A3_CORE_PATH_PROOFS:
        assert_gr12_a3_path_semantic_integrity(bundle)
