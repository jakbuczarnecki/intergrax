# © Artur Czarnecki. All rights reserved.

"""EBH-5 RPL-G regression gates — prohibited coupling reintroduction."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.qualification.config_x._config_x_discovery import discover_active_blocker_path_keys
from tests.qualification.ebh_5._replaceability_matrix import CROSS_LAYER_REPLACEABILITY_MATRIX
from tests.qualification.ebh_5._vendor_backend_inventory import VENDOR_BACKEND_DEPENDENCY_INVENTORY
from tests.qualification.ebh_5._vendor_backend_types import (
    ExtensibilityClassification,
    ExtensibilityPromised,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_REPO_ROOT = Path(__file__).resolve().parents[3]

_FORBIDDEN_PATTERNS: tuple[tuple[str, str], ...] = (
    ("intergrax/rag/retrieval_service.py", "openai"),
    ("intergrax/integrations/registry/factory.py", "shadow_registry"),
)


def test_ebh5_rpl_g_config_x_active_blockers_remain_empty() -> None:
    assert discover_active_blocker_path_keys() == frozenset()


def test_ebh5_rpl_g_inventory_matrix_concern_id_parity() -> None:
    matrix_ids = {row.concern_id for row in CROSS_LAYER_REPLACEABILITY_MATRIX}
    for inv in VENDOR_BACKEND_DEPENDENCY_INVENTORY:
        if inv.extensibility_promised in (
            ExtensibilityPromised.YES,
            ExtensibilityPromised.PARTIAL,
        ):
            assert inv.concern_id in matrix_ids


def test_ebh5_rpl_g_no_architecture_blocker_without_debt() -> None:
    for row in VENDOR_BACKEND_DEPENDENCY_INVENTORY:
        if row.extensibility_classification == ExtensibilityClassification.ARCHITECTURE_BLOCKER:
            pytest.fail(f"unresolved ARCHITECTURE BLOCKER: {row.concern_id}")


def test_ebh5_rpl_g_tier0_no_agents_import_applications() -> None:
    tier0 = _REPO_ROOT / "intergrax"
    skip_prefixes = (
        "intergrax/cli/",
        "intergrax/integrations/providers/",
    )
    for path in tier0.rglob("*.py"):
        rel = path.relative_to(_REPO_ROOT).as_posix()
        if rel.startswith(skip_prefixes):
            continue
        text = path.read_text(encoding="utf-8")
        if "from applications" in text:
            pytest.fail(f"tier boundary violation: {rel}")


def test_ebh5_rpl_g_lab_factory_documents_lab_only_adaptive_stores() -> None:
    factory = _REPO_ROOT / "applications/lab_application/host/factory.py"
    text = factory.read_text(encoding="utf-8")
    assert "lab_harness_preset" in text or "LabApplicationSettings" in text
    assert "include_adaptive_debug_routes" in text or "adaptive_signal_store" in text
