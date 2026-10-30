# © Artur Czarnecki. All rights reserved.

"""EBH-5 Package C v2 — relational replaceability probes (semantic PARTIAL until RPL-A..H complete)."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.qualification.ebh_5._package_c_v2_proof_helpers import (
    run_package_c_v2_relational_sqlite_duckdb_dual_slug,
    run_package_c_v2_relational_unregistered_fail_closed,
)

pytestmark = [pytest.mark.unit, pytest.mark.qualification]


def test_ebh5_pkgc_v2_relational_sqlite_duckdb_dual_slug(tmp_path: Path) -> None:
    run_package_c_v2_relational_sqlite_duckdb_dual_slug(tmp_path)


def test_ebh5_pkgc_v2_relational_unregistered_fail_closed() -> None:
    run_package_c_v2_relational_unregistered_fail_closed()
