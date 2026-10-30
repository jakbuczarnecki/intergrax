# © Artur Czarnecki. All rights reserved.

"""EBH-5 Package C v2 behavioral proofs (dual resolve_from_profile)."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.qualification.ebh_5._package_c_v2_proof_helpers import run_package_c_v2_relational_dual_path

pytestmark = [pytest.mark.unit, pytest.mark.qualification]


def test_ebh5_pkgc_v2_relational_dual_path(tmp_path: Path) -> None:
    run_package_c_v2_relational_dual_path(tmp_path)
