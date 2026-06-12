# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE_PKG = _REPO_ROOT / "tests" / "fixtures" / "plugin_packages" / "intergrax_catalog_fixture"


def _install_catalog_fixture_package() -> None:
    import importlib
    import shutil

    python = sys.executable
    uv = shutil.which("uv")
    if uv is not None:
        subprocess.check_call(
            [uv, "pip", "install", str(_FIXTURE_PKG), "--python", python],
            cwd=str(_REPO_ROOT),
        )
    else:
        subprocess.check_call(
            [python, "-m", "pip", "install", str(_FIXTURE_PKG), "-q"],
            cwd=str(_REPO_ROOT),
        )
    importlib.import_module("intergrax_catalog_fixture")


@pytest.fixture(scope="session")
def catalog_fixture_installed() -> None:
    """Install catalog entry-point fixture package for pytest (Phase P-Ext.0.5)."""
    _install_catalog_fixture_package()


@pytest.fixture(scope="session", autouse=True)
def _ensure_agent_fleet_inventory() -> None:
    """Generate fleet inventory when gate tests run without prior governance scripts."""
    inventory_path = _REPO_ROOT / "build" / "agent_fleet_inventory.json"
    if inventory_path.is_file():
        return
    subprocess.check_call(
        [sys.executable, str(_REPO_ROOT / "scripts" / "audit_agent_fleet_legacy.py")],
        cwd=str(_REPO_ROOT),
    )


@pytest.fixture
def session_manager_in_memory():
    from testing_support.builder import build_in_memory_session_manager

    return build_in_memory_session_manager()

