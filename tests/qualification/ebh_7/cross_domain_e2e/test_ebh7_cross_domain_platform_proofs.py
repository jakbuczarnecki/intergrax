# © Artur Czarnecki. All rights reserved.

"""EBH-7 cross-domain platform proofs."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_ebh7_xdm_010_backend_persistence_sqlite_behavioral(tmp_path: Path) -> None:
    from tests.qualification.ebh_5.test_ebh_5_behavioral_proofs import (
        test_ebh5_behavioral_relational_store_negative_unregistered_provider_fail_closed,
        test_ebh5_behavioral_relational_store_sqlite_via_sanctioned_profile,
    )

    test_ebh5_behavioral_relational_store_sqlite_via_sanctioned_profile(tmp_path)
    test_ebh5_behavioral_relational_store_negative_unregistered_provider_fail_closed(tmp_path)
