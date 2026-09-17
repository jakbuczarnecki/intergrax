# © Artur Czarnecki. All rights reserved.

"""SESSION-01 — Q catalog integrity (collection gate, zero duplicate proofs)."""

from __future__ import annotations

import pytest

from tests.qualification.session_01.catalog import SESSION_01_Q_CATALOG

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.mark.parametrize(
    "q_id",
    [entry.q_id for entry in SESSION_01_Q_CATALOG],
    ids=[entry.q_id for entry in SESSION_01_Q_CATALOG],
)
def test_session_01_q_catalog_has_evidence(q_id: str) -> None:
    entry = next(item for item in SESSION_01_Q_CATALOG if item.q_id == q_id)
    assert entry.architecture_reason_na is None
    assert entry.pytest_node_ids
    for node_id in entry.pytest_node_ids:
        assert "::" in node_id
        assert node_id.startswith("tests/")


def test_session_01_q_catalog_covers_q1_through_q20() -> None:
    ids = {entry.q_id for entry in SESSION_01_Q_CATALOG}
    expected = {f"Q{i}" for i in range(1, 21)}
    assert ids == expected
