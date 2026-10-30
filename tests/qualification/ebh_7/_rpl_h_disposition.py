# © Artur Czarnecki. All rights reserved.

"""EBH-7 RPL-H disposition SSOT @ HEAD (mirrors ebh_5 matrix)."""

from __future__ import annotations

from tests.qualification.ebh_5._replaceability_matrix import CROSS_LAYER_REPLACEABILITY_MATRIX
from tests.qualification.ebh_5._vendor_backend_types import ExtensibilityClassification
from tests.qualification.ebh_7._constants import EBH7_RPL_MATRIX_ROWS


def assert_ebh7_rpl_h_disposition_integrity() -> None:
    assert len(CROSS_LAYER_REPLACEABILITY_MATRIX) == EBH7_RPL_MATRIX_ROWS
    not_yet = sum(
        1 for r in CROSS_LAYER_REPLACEABILITY_MATRIX if r.rpl_h == ExtensibilityClassification.REPLACEABLE_NOT_YET
    )
    qualified = sum(
        1 for r in CROSS_LAYER_REPLACEABILITY_MATRIX if r.rpl_h == ExtensibilityClassification.REPLACEABLE_QUALIFIED
    )
    assert not_yet == 51
    assert qualified == 5
