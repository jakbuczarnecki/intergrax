# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_REPO_ROOT = ROOT
_FIXTURE_PKG = _REPO_ROOT / "tests" / "fixtures" / "plugin_packages" / "intergrax_catalog_fixture"
_SECURITY_DEFENSE_FIXTURE_PKG = (
    _REPO_ROOT / "tests" / "fixtures" / "plugin_packages" / "intergrax_security_defense_fixture"
)
_ACME_REFERENCE_VK_PLUGIN_PKG = (
    _REPO_ROOT
    / "examples"
    / "platform_plugins"
    / "intergrax_reference_vendor_knowledge_plugin"
)
_REFERENCE_ENTERPRISE_PLUGIN_PKG = (
    _REPO_ROOT / "examples" / "platform_plugins" / "intergrax_reference_enterprise_plugin"
)


def _install_editable_package(package_dir: Path, module_name: str) -> None:
    import importlib
    import shutil

    python = sys.executable
    uv = shutil.which("uv")
    if uv is not None:
        subprocess.check_call(
            [uv, "pip", "install", str(package_dir), "--python", python],
            cwd=str(_REPO_ROOT),
        )
    else:
        subprocess.check_call(
            [python, "-m", "pip", "install", str(package_dir), "-q"],
            cwd=str(_REPO_ROOT),
        )
    importlib.import_module(module_name)


def _install_catalog_fixture_package() -> None:
    _install_editable_package(_FIXTURE_PKG, "intergrax_catalog_fixture")


def _install_security_defense_fixture_package() -> None:
    _install_editable_package(_SECURITY_DEFENSE_FIXTURE_PKG, "intergrax_security_defense_fixture")


def _install_acme_reference_vk_plugin_package() -> None:
    _install_editable_package(_ACME_REFERENCE_VK_PLUGIN_PKG, "acme_reference_vk_plugin")


def _install_reference_enterprise_plugin_package() -> None:
    _install_editable_package(_REFERENCE_ENTERPRISE_PLUGIN_PKG, "intergrax_reference_enterprise_plugin")


@pytest.fixture(scope="session")
def catalog_fixture_installed() -> None:
    """Install catalog entry-point fixture package for pytest (Phase P-Ext.0.5)."""
    _install_catalog_fixture_package()


@pytest.fixture(scope="session")
def security_defense_fixture_installed() -> None:
    """Install security defense EP fixture package for pytest (Phase SEC-EVOL-2)."""
    _install_security_defense_fixture_package()


@pytest.fixture(scope="session")
def acme_reference_vk_plugin_installed() -> None:
    """Install VK-EXT-3 reference external provider entry-point package."""
    _install_acme_reference_vk_plugin_package()


@pytest.fixture(scope="session")
def reference_enterprise_plugin_installed() -> None:
    """Install DOCS-6 multi-capability reference package."""
    _install_reference_enterprise_plugin_package()


@pytest.fixture(scope="session", autouse=True)
def _ensure_agent_fleet_inventory() -> None:
    """Regenerate fleet inventory so gate readiness checks see current migration roster."""
    subprocess.check_call(
        [sys.executable, str(_REPO_ROOT / "scripts" / "audit" / "audit_agent_fleet_legacy.py")],
        cwd=str(_REPO_ROOT),
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """When `-m gate` is used without `no_ci`, drop infra-heavy `no_ci` tests from selection."""
    markexpr = str(config.getoption("-m") or "")
    if "no_ci" in markexpr or "gate" not in markexpr:
        return
    deselected: list[pytest.Item] = []
    remaining: list[pytest.Item] = []
    for item in items:
        if item.get_closest_marker("no_ci"):
            deselected.append(item)
        else:
            remaining.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = remaining


@pytest.fixture
def session_manager_in_memory():
    from testing_support.builder import build_in_memory_session_manager

    return build_in_memory_session_manager()

