# © Artur Czarnecki. All rights reserved.

"""GOV-X2 — mechanical GX2 invariant matrix and E2E class evidence collectability."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.qualification.governance.gov_x2.catalog import (
    GOV_X2_INVARIANT_CATALOG,
    GOV_X2_START_HEAD,
    GovX2InvariantResult,
    gov_x2_all_proof_pytest_node_ids,
)
from tests.qualification.governance.pytest_node_integrity import (
    pytest_nodes_missing_from_collection,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]


def test_gov_x2_catalog_covers_gx2_01_through_gx2_20() -> None:
    ids = {entry.invariant_id for entry in GOV_X2_INVARIANT_CATALOG}
    expected = {f"GX2-{i:02d}" for i in range(1, 21)}
    assert ids == expected


def test_gov_x2_invariants_declare_pass_with_proof_nodes() -> None:
    for entry in GOV_X2_INVARIANT_CATALOG:
        assert entry.result is GovX2InvariantResult.PASS, entry.invariant_id
        assert entry.primary_pytest_node_ids, entry.invariant_id


def test_gov_x2_all_proof_pytest_node_ids_are_collectable() -> None:
    combined = gov_x2_all_proof_pytest_node_ids()
    missing, proc = pytest_nodes_missing_from_collection(combined, _REPO_ROOT)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not missing, f"missing={len(missing)}: {sorted(missing)}"


def test_gov_x2_start_head_documented_in_catalog() -> None:
    assert len(GOV_X2_START_HEAD) == 40
