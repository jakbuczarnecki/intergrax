# © Artur Czarnecki. All rights reserved.

"""GR-12-A3-R1 — per-path qualification evidence integrity gates."""

from __future__ import annotations

import pytest

from tests.qualification.governance.gr12.a3_path_qualifications import (
    GR12_A3_CORE_PATH_PROOFS,
    GR12_A3_SHARED_MECHANISMS,
    GR12_A3_SHARED_MECHANISM_BY_ID,
    Gr12QualificationPathKind,
)
from tests.qualification.governance.gr12.qualification_support import (
    assert_gr12_a3_path_semantic_integrity,
    assert_proof_nodes_registered,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_gr12_a3_r1_shared_mechanisms_registered() -> None:
    assert len(GR12_A3_SHARED_MECHANISMS) == len(GR12_A3_SHARED_MECHANISM_BY_ID)
    for mechanism in GR12_A3_SHARED_MECHANISMS:
        assert mechanism.proof_tests
        assert_proof_nodes_registered(mechanism.proof_tests)


def test_gr12_a3_r1_composition_paths_use_composition_kind() -> None:
    for path_id in ("CP-HOST-BOUNDARY-OPTIONAL", "CP-ECP-BOUNDARY-OPTIONAL"):
        bundle = next(row for row in GR12_A3_CORE_PATH_PROOFS if row.path_id == path_id)
        assert bundle.kind is Gr12QualificationPathKind.COMPOSITION_SURFACE


def test_gr12_a3_r1_all_qualified_paths_semantically_consistent() -> None:
    for bundle in GR12_A3_CORE_PATH_PROOFS:
        assert_gr12_a3_path_semantic_integrity(bundle)
