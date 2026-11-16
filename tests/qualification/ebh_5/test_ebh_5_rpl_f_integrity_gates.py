# © Artur Czarnecki. All rights reserved.

"""RPL-F integrity gates (OD-EBH5-07)."""

from __future__ import annotations

import pytest
from tests.qualification.ebh_5._behavioral_proof_registry import QUALIFIED_CONCERN_IDS
from tests.qualification.ebh_5._replaceability_matrix import matrix_by_concern_id
from tests.qualification.ebh_5._rpl_f_evidence_registry import (
    _RPL_F_ACTIVE_PROOF_NODES,
    rpl_f_disposition,
    rpl_f_validation_errors,
)
from tests.qualification.ebh_5._vendor_backend_types import ExtensibilityClassification, RplGateStatus
from tests.qualification.ebh_7._ebh7_terminal_operator_authorization import (
    fixed_substrate_rpl_concern_ids,
    internal_rpl_concern_ids,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_ebh5_rpl_f_registry_complete_for_qualified() -> None:
    assert not rpl_f_validation_errors()


def test_ebh5_rpl_f_matrix_matches_registry() -> None:
    # PDC-X B-REG-T (T-IGX-036-I): aligned with EBH-7 terminal reclassification d4937889f
    by_id = matrix_by_concern_id()
    qualified_at_head = QUALIFIED_CONCERN_IDS - internal_rpl_concern_ids() - fixed_substrate_rpl_concern_ids()
    assert set(_RPL_F_ACTIVE_PROOF_NODES) <= qualified_at_head
    for concern_id in qualified_at_head:
        row = by_id[concern_id]
        assert row.rpl_h == ExtensibilityClassification.REPLACEABLE_QUALIFIED
        disp = rpl_f_disposition(concern_id)
        assert disp is not None
        assert row.rpl_f == disp.gate
        if concern_id in _RPL_F_ACTIVE_PROOF_NODES:
            assert disp.gate == RplGateStatus.PASS
            assert len(disp.proof_pytest_nodes) >= 1
        else:
            assert disp.gate == RplGateStatus.NA
            assert disp.normative_na_rationale.strip().startswith("N/A — WITH EVIDENCE")
