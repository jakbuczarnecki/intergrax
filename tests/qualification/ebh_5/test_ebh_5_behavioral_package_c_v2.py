# © Artur Czarnecki. All rights reserved.

"""EBH-5 Package C v2 — dual-slug RPL-H batch C qualification tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.qualification.ebh_5._package_c_v2_dual_slug_catalog import INTEGRATION_DUAL_SLUG_CATALOG
from tests.qualification.ebh_5._package_c_v2_proof_helpers import (
    run_package_c_v2_dual_slug_for_entry,
    run_package_c_v2_relational_sqlite_duckdb_dual_slug,
    run_package_c_v2_relational_unregistered_fail_closed,
    run_package_c_v2_unregistered_for_entry,
)
from tests.qualification.ebh_5._package_c_v2_rpl_cde import run_package_c_v2_dual_slug_rpl_cde_equivalence

pytestmark = [pytest.mark.unit, pytest.mark.qualification]


def test_ebh5_pkgc_v2_relational_sqlite_duckdb_dual_slug(tmp_path: Path) -> None:
    run_package_c_v2_relational_sqlite_duckdb_dual_slug(tmp_path)


def test_ebh5_pkgc_v2_relational_unregistered_fail_closed() -> None:
    run_package_c_v2_relational_unregistered_fail_closed()


@pytest.mark.parametrize(
    "entry",
    INTEGRATION_DUAL_SLUG_CATALOG,
    ids=lambda e: e.concern_id,
)
def test_ebh5_pkgc_v2_dual_slug_integration(entry: object, tmp_path: Path) -> None:
    from tests.qualification.ebh_5._package_c_v2_dual_slug_catalog import IntegrationDualSlugCatalogEntry

    assert isinstance(entry, IntegrationDualSlugCatalogEntry)
    run_package_c_v2_dual_slug_for_entry(entry, tmp_path=tmp_path)


@pytest.mark.parametrize(
    "entry",
    INTEGRATION_DUAL_SLUG_CATALOG,
    ids=lambda e: e.concern_id,
)
def test_ebh5_pkgc_v2_dual_slug_unregistered(entry: object) -> None:
    from tests.qualification.ebh_5._package_c_v2_dual_slug_catalog import IntegrationDualSlugCatalogEntry

    assert isinstance(entry, IntegrationDualSlugCatalogEntry)
    run_package_c_v2_unregistered_for_entry(entry)


@pytest.mark.parametrize(
    "entry",
    INTEGRATION_DUAL_SLUG_CATALOG,
    ids=lambda e: e.concern_id,
)
def test_ebh5_pkgc_v2_dual_slug_rpl_cde_equivalence(entry: object, tmp_path: Path) -> None:
    from tests.qualification.ebh_5._package_c_v2_dual_slug_catalog import IntegrationDualSlugCatalogEntry

    assert isinstance(entry, IntegrationDualSlugCatalogEntry)
    run_package_c_v2_dual_slug_rpl_cde_equivalence(entry, tmp_path=tmp_path)
