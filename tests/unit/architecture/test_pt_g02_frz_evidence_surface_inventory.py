# © Artur Czarnecki. All rights reserved.

"""PT-G02 (T-IGX-046-G): FRZ evidence surface inventory and STATE-X citation registry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_INVENTORY = _REPO_ROOT / "governance/qualification/frz_evidence_surface_inventory.v1.json"
_EXPECTED_FAMILIES = frozenset({"BND", "TYP", "PLG", "OWN", "CTR", "DEBT"})


def _load_inventory() -> dict:
    raw = json.loads(_INVENTORY.read_text(encoding="utf-8"))
    assert raw.get("schema_version") == "frz_evidence_surface_inventory.v1"
    return raw


def test_pt_g02_inventory_documents_named_families() -> None:
    inv = _load_inventory()
    family_ids = {entry["family_id"] for entry in inv["families"]}
    assert family_ids == _EXPECTED_FAMILIES


@pytest.mark.parametrize("family_id", sorted(_EXPECTED_FAMILIES))
def test_pt_g02_family_gate_paths_exist(family_id: str) -> None:
    inv = _load_inventory()
    entry = next(item for item in inv["families"] if item["family_id"] == family_id)
    for key in ("gate_inventory", "companion"):
        rel = entry.get(key)
        if not rel:
            continue
        path = _REPO_ROOT / rel
        assert path.is_file(), rel


def test_pt_g02_state_x_citation_roots_exist() -> None:
    inv = _load_inventory()
    repair = inv["state_x_citation_repair"]
    assert repair.get("finding_id") == "A3-018"
    for rel in repair["qualification_roots"]:
        assert (_REPO_ROOT / rel).is_file(), rel


def test_pt_g02_frz_rows_non_empty() -> None:
    inv = _load_inventory()
    for entry in inv["families"]:
        rows = entry["frz_rows"]
        assert rows, entry["family_id"]
        for frz_id in rows:
            assert frz_id.startswith("FRZ-")
