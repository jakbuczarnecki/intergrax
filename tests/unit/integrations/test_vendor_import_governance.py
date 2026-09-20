# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


def _load_vendor_boundary_checker(root: Path):
    script = root / "scripts" / "maintenance" / "check_integration_vendor_imports.py"
    spec = importlib.util.spec_from_file_location("check_integration_vendor_imports", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_vendor_imports_stay_in_integration_boundary() -> None:
    root = Path(__file__).resolve().parents[3]
    script = root / "scripts" / "check_integration_vendor_imports.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_qdrant_client_factory_is_canonical_integration_provider_boundary() -> None:
    root = Path(__file__).resolve().parents[3]
    checker = _load_vendor_boundary_checker(root)
    factory = root / "intergrax/integrations/providers/vector_store/qdrant/client_factory.py"
    assert checker._is_allowed(factory, scope="integrations")


def test_integration_provider_non_boundary_module_still_forbidden_for_vendor() -> None:
    root = Path(__file__).resolve().parents[3]
    checker = _load_vendor_boundary_checker(root)
    config = root / "intergrax/integrations/providers/vector_store/qdrant/config.py"
    assert not checker._is_allowed(config, scope="integrations")
    hits = checker._collect_imports(config)
    assert hits == []


def test_vendor_import_in_non_boundary_provider_module_is_detected(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[3]
    checker = _load_vendor_boundary_checker(root)
    hacks = root / "intergrax/integrations/providers/vector_store/qdrant/data_plane.py"
    assert not checker._is_allowed(hacks, scope="integrations")
    probe = tmp_path / "probe.py"
    probe.write_text("from qdrant_client import QdrantClient\n", encoding="utf-8")
    hits = checker._collect_imports(probe)
    assert any(root_name == "qdrant_client" for root_name, _ in hits)
