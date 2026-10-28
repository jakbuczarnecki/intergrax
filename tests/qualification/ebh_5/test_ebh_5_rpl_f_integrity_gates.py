# © Artur Czarnecki. All rights reserved.

"""RPL-F integrity gates (OD-EBH5-07)."""

from __future__ import annotations

import pytest
from tests.qualification.ebh_5._behavioral_proof_registry import QUALIFIED_CONCERN_IDS
from tests.qualification.ebh_5._replaceability_matrix import matrix_by_concern_id
from tests.qualification.ebh_5._rpl_f_evidence_registry import (
    rpl_f_disposition,
    rpl_f_validation_errors,
)
from tests.qualification.ebh_5._vendor_backend_types import ExtensibilityClassification, RplGateStatus

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_ebh5_rpl_f_registry_complete_for_qualified() -> None:
    assert not rpl_f_validation_errors()


def test_ebh5_rpl_f_matrix_matches_registry() -> None:
    by_id = matrix_by_concern_id()
    for concern_id in QUALIFIED_CONCERN_IDS:
        row = by_id[concern_id]
        assert row.rpl_h == ExtensibilityClassification.REPLACEABLE_QUALIFIED
        disp = rpl_f_disposition(concern_id)
        assert disp is not None
        assert row.rpl_f == disp.gate
        assert disp.gate == RplGateStatus.PASS
        assert len(disp.proof_pytest_nodes) >= 1
