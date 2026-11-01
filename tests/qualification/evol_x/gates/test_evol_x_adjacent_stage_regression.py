# © Artur Czarnecki. All rights reserved.

import pytest

pytestmark = [pytest.mark.qualification, pytest.mark.unit]


def test_evol_x_adjacent_qual_x_catalog_importable() -> None:
    from tests.qualification.qual_x._p5_freeze_selection_catalog import QUAL_X_P5_FREEZE_SELECTION_MANIFEST

    assert QUAL_X_P5_FREEZE_SELECTION_MANIFEST
