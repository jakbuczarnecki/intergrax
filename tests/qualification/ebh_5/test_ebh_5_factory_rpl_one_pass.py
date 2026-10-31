# © Artur Czarnecki. All rights reserved.

"""EBH-7 one-pass factory RPL qualification tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.qualification.ebh_5._factory_rpl_one_pass_proofs import (
    FACTORY_RPL_ONE_PASS_CONCERNS,
    run_factory_rpl_negative,
    run_factory_rpl_positive,
)

pytestmark = [pytest.mark.unit, pytest.mark.qualification, pytest.mark.gate]


@pytest.mark.parametrize("concern_id", sorted(FACTORY_RPL_ONE_PASS_CONCERNS), ids=lambda c: c)
def test_ebh5_factory_rpl_positive(concern_id: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_factory_rpl_positive(concern_id, monkeypatch=monkeypatch, tmp_path=tmp_path)


@pytest.mark.parametrize("concern_id", sorted(FACTORY_RPL_ONE_PASS_CONCERNS), ids=lambda c: c)
def test_ebh5_factory_rpl_negative(concern_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    run_factory_rpl_negative(concern_id, monkeypatch=monkeypatch)
