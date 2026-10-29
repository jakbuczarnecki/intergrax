# © Artur Czarnecki. All rights reserved.

"""DUP-X Wave B — class D integration selector + COMPAT-R3 production fence gates."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.qualification.dup_x._closed_world_inventory import (
    _LAB_COMPAT_PATHS,
    discover_integration_env_slug_reads_outside_factory,
)
from tests.qualification.ebh_6._closed_world_discovery import discover_sanctioned_production_composition_roots

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_REPO_ROOT = Path(__file__).resolve().parents[3]

_FORBIDDEN_LAB_IMPORT_MARKERS = (
    "debug_lab_nexus_loop",
    "runtime.transport.bootstrap",
    "build_lab_organization_nexus_loop",
)


def test_dup_x_class_d_factory_is_sanctioned_selector_owner() -> None:
    factory = _REPO_ROOT / "intergrax/integrations/registry/factory.py"
    text = factory.read_text(encoding="utf-8")
    assert "resolve_from_profile" in text
    assert "read_integration_slug_from_env" in text


def test_dup_x_class_d_no_shadow_category_slug_reads_in_intergrax() -> None:
    """Production intergrax modules must not call read_integration_slug outside factory/config."""
    shadow = discover_integration_env_slug_reads_outside_factory()
    assert not shadow, f"shadow env slug reads: {sorted(shadow)}"


def test_dup_x_compat_r3_lab_paths_not_imported_from_sanctioned_roots() -> None:
    sanctioned = discover_sanctioned_production_composition_roots()
    violations: list[str] = []
    for rel in sorted(sanctioned):
        path = _REPO_ROOT / rel.replace("\\", "/")
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for marker in _FORBIDDEN_LAB_IMPORT_MARKERS:
            if marker in text:
                violations.append(f"{rel}: {marker}")
    assert not violations, violations


def test_dup_x_compat_r3_lab_paths_exist_and_documented() -> None:
    for rel in _LAB_COMPAT_PATHS:
        assert (_REPO_ROOT / rel).is_file(), rel


def test_dup_x_negative_bypass_sanctioned_roots_compose_markers() -> None:
    """Sanctioned roots must retain composition markers (no silent empty shells)."""
    for rel in discover_sanctioned_production_composition_roots():
        text = (_REPO_ROOT / rel.replace("\\", "/")).read_text(encoding="utf-8").lower()
        assert any(token in text for token in ("compose", "wiring", "factory", "resolve_from_profile")), rel
