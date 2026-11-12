# © Artur Czarnecki. All rights reserved.

"""PDC-X P4 implementation WBS checks (T-IGX-036-G)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "scripts/qualification/pdc_x_wbs.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("pdc_x_wbs_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def wbs() -> ModuleType:
    return _load()


def test_committed_outputs_are_current(wbs: ModuleType) -> None:
    assert wbs.check() == []


def test_every_work_row_has_exactly_one_known_batch(wbs: ModuleType) -> None:
    ids = {b.bid for b in wbs.BATCHES}
    rows = wbs.work_rows(wbs.load_rows())
    assert rows
    for row in rows:
        got = wbs.assign(row)
        assert len(got) == 1 and got[0] in ids, (row["REQUIREMENT_ID"], got)


def test_product_batches_carry_the_mcp_merge_gate(wbs: ModuleType) -> None:
    for batch in wbs.BATCHES:
        if wbs._is_product(batch):
            assert batch.gate == wbs.GATE_PRODUCT, batch.bid


def test_validate_rejects_same_wave_write_conflict(wbs: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    batches = list(wbs.BATCHES)
    d1 = next(b for b in batches if b.bid == "B-D1")
    clash = d1._replace(bid="B-CLASH", writes=("intergrax/agent_distribution/admin_service.py",))
    monkeypatch.setattr(wbs, "BATCHES", (*batches, clash))
    assert any(e.startswith("WRITE_CONFLICT:W1:") for e in wbs.validate(wbs.load_rows()))


def test_validate_rejects_two_hotspot_writers_in_one_wave(wbs: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    batches = list(wbs.BATCHES)
    d3 = next(b for b in batches if b.bid == "B-D3")
    widened = d3._replace(writes=(*d3.writes, "intergrax/contracts/new_model_contract.py"))
    monkeypatch.setattr(wbs, "BATCHES", tuple(widened if b.bid == "B-D3" else b for b in batches))
    assert any(e.startswith("HOTSPOT:W1:intergrax/contracts/") for e in wbs.validate(wbs.load_rows()))


def test_validate_rejects_unassigned_row_and_cycle(wbs: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = wbs.load_rows()
    gap = next(r for r in rows if r["GAP_CLASS"] == "IMPLEMENTATION_COMPLETION")
    orphan = {**gap, "REQUIREMENT_ID": "PDC-D9-R999", "CORRECTIVE_TASK": "unplanned"}
    batches = list(wbs.BATCHES)
    d1 = next(b for b in batches if b.bid == "B-D1")
    cyclic = tuple(d1._replace(deps=("B-D2",)) if b.bid == "B-D1" else b for b in batches)
    monkeypatch.setattr(wbs, "BATCHES", cyclic)
    errors = wbs.validate([*rows, orphan])
    assert any(e.startswith("ASSIGNMENT:PDC-D9-R999") for e in errors)
    assert any(e.startswith("CYCLE:") for e in errors)
