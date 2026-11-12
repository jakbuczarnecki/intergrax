# © Artur Czarnecki. All rights reserved.

"""PDC-X P1 closed-world inventory scanner (T-IGX-036-B)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "scripts/qualification/pdc_x_inventory.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("pdc_x_inventory_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def inv() -> ModuleType:
    return _load()


def test_scan_is_deterministic(inv: ModuleType) -> None:
    assert inv.render(inv.build()) == inv.render(inv.build())


def test_committed_inventory_is_current_and_closed_world(inv: ModuleType) -> None:
    assert inv.check(inv.build()) == []


def test_vendored_runtime_context_copies_are_excluded(inv: ModuleType) -> None:
    assert all(inv.VENDORED_MARKER not in item["path"] for item in inv.build()["items"])


def test_every_item_has_exactly_one_known_category_and_status(inv: ModuleType) -> None:
    allowed = set(inv.DOMAINS) | {inv.FOUNDATION, inv.CONSUMER}
    for item in inv.build()["items"]:
        assert item["category"] in allowed, item
        assert item["status"] in {"INSPECTED", "INDEX_ONLY", "UNINSPECTED"}, item


def test_domain_items_are_never_index_only(inv: ModuleType) -> None:
    for item in inv.build()["items"]:
        if item["category"] in inv.DOMAINS:
            assert item["status"] in {"INSPECTED", "UNINSPECTED"}, item


def test_check_rejects_unmapped_surface(inv: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(inv, "classify_doc", lambda stem: inv.UNMAPPED)
    errors = inv.check(inv.build())
    assert any(err.startswith("UNMAPPED:") for err in errors)


def test_check_rejects_missing_domain_anchor(inv: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    links = {k: dict(v) for k, v in inv.DOMAIN_LINKS.items()}
    links["PDC-D8"]["semantic_owners"] = ["intergrax/does_not_exist_pdc.py"]
    monkeypatch.setattr(inv, "DOMAIN_LINKS", links)
    assert "PDC-D8:semantic_owners:intergrax/does_not_exist_pdc.py" in inv.missing_anchors()


def test_all_eight_domains_have_anchor_links(inv: ModuleType) -> None:
    assert set(inv.DOMAIN_LINKS) == set(inv.DOMAINS)
    for links in inv.DOMAIN_LINKS.values():
        assert links["semantic_owners"], links
