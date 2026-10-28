# © Artur Czarnecki. All rights reserved.

"""EBH-5 proof integrity gates (EBH5-A01 negative + EBH5-A04 provenance)."""

from __future__ import annotations

import pytest
from tests.qualification.ebh_5._behavioral_proof_registry import (
    QUALIFIED_CONCERN_BEHAVIORAL_PROOF_NODE,
    QUALIFIED_CONCERN_IDS,
)
from tests.qualification.ebh_5._evidence_manifest import (
    AUDIT_BASELINE_SHA,
    assert_audit_baseline_resolvable,
    evidence_git_head_sha,
    proof_execution_sha,
)
from tests.qualification.ebh_5._proof_validation import (
    is_valid_git_sha40,
    parse_pytest_node_id,
    pytest_test_function_exists,
    qualified_rows_with_invalid_proofs,
    validate_behavioral_proof_node_for_concern,
)
from tests.qualification.ebh_5._replaceability_matrix import CROSS_LAYER_REPLACEABILITY_MATRIX
from tests.qualification.ebh_5._sqlite_touchpoint_registry import (
    SQLITE_TOUCHPOINT_COUNT,
    assert_closed_world_sqlite_accountability,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_ebh5_proof_inventory_matrix_qualified_parity() -> None:
    from tests.qualification.ebh_5._vendor_backend_inventory import VENDOR_BACKEND_DEPENDENCY_INVENTORY
    from tests.qualification.ebh_5._vendor_backend_types import ExtensibilityClassification

    matrix_q = {
        row.concern_id
        for row in CROSS_LAYER_REPLACEABILITY_MATRIX
        if row.rpl_h == ExtensibilityClassification.REPLACEABLE_QUALIFIED
    }
    inv_q = {
        row.concern_id
        for row in VENDOR_BACKEND_DEPENDENCY_INVENTORY
        if row.extensibility_classification == ExtensibilityClassification.REPLACEABLE_QUALIFIED
    }
    assert matrix_q == inv_q == QUALIFIED_CONCERN_IDS


def test_ebh5_proof_all_qualified_rows_resolve() -> None:
    assert qualified_rows_with_invalid_proofs(CROSS_LAYER_REPLACEABILITY_MATRIX) == ()


def test_ebh5_proof_registry_covers_five_qualified_concerns() -> None:
    assert len(QUALIFIED_CONCERN_BEHAVIORAL_PROOF_NODE) == 5


def test_ebh5_proof_audit_baseline_sha_resolves() -> None:
    assert_audit_baseline_resolvable()
    assert is_valid_git_sha40(AUDIT_BASELINE_SHA)


def test_ebh5_proof_execution_sha_is_valid_at_run_time() -> None:
    sha = proof_execution_sha()
    assert is_valid_git_sha40(sha)
    assert sha == evidence_git_head_sha()


def test_ebh5_proof_sqlite_closed_world_accountability() -> None:
    assert SQLITE_TOUCHPOINT_COUNT == 38
    assert_closed_world_sqlite_accountability()


@pytest.mark.parametrize(
    ("concern_id", "bad_ref", "fragment"),
    [
        (
            "persistence.relational_sqlite_lab",
            "INT-CONFIG-REAL-X @ a59744517b92847f55def1db22826d17d89ee155",
            "historical",
        ),
        (
            "int_config.configure_existing",
            "tests/qualification/ebh_5/test_ebh_5_behavioral_proofs.py::test_nonexistent",
            "pytest node",
        ),
        (
            "observability.otlp_transport",
            "tests/unit/foo.py::test_ebh5_behavioral_otlp_transport_port_injection",
            "ebh_5",
        ),
        (
            "tokenizer.selection",
            "PENDING",
            "pending",
        ),
        (
            "integration.key_value_cache",
            QUALIFIED_CONCERN_BEHAVIORAL_PROOF_NODE["persistence.relational_sqlite_lab"],
            "wrong concern",
        ),
    ],
)
def test_ebh5_proof_rejects_pozorny_qualified_refs(
    concern_id: str,
    bad_ref: str,
    fragment: str,
) -> None:
    err = validate_behavioral_proof_node_for_concern(concern_id, bad_ref)
    assert err is not None, fragment
    assert fragment in err or "wrong concern" in err or "pending" in err or "historical" in err


def test_ebh5_proof_sha_format_does_not_imply_git_object() -> None:
    fake = "0" * 40
    assert is_valid_git_sha40(fake)
    assert (
        parse_pytest_node_id(
            "tests/qualification/ebh_5/test_ebh_5_behavioral_proofs.py::test_ebh5_behavioral_otlp_transport_port_injection",
        )
        is not None
    )
    assert pytest_test_function_exists(
        "tests/qualification/ebh_5/test_ebh_5_behavioral_proofs.py::test_ebh5_behavioral_otlp_transport_port_injection",
    )
