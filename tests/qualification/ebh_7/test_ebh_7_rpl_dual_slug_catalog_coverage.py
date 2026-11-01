# © Artur Czarnecki. All rights reserved.

"""Package C — dual-slug catalog coverage @ Package C v2 (INT2E regression depth)."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.qualification.ebh_5._package_c_v2_dual_slug_catalog import INTEGRATION_DUAL_SLUG_CATALOG
from tests.qualification.ebh_5._package_c_v2_proof_helpers import run_package_c_v2_dual_slug_for_entry

pytestmark = [pytest.mark.qualification, pytest.mark.unit]


@pytest.mark.parametrize(
    "entry",
    INTEGRATION_DUAL_SLUG_CATALOG,
    ids=lambda e: e.concern_id,
)
def test_ebh7_rpl_dual_slug_catalog_entry_resolves(entry: object, tmp_path: Path) -> None:
    from tests.qualification.ebh_5._package_c_v2_dual_slug_catalog import IntegrationDualSlugCatalogEntry

    assert isinstance(entry, IntegrationDualSlugCatalogEntry)
    run_package_c_v2_dual_slug_for_entry(entry, tmp_path=tmp_path)
