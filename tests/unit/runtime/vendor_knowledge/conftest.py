# © Artur Czarnecki. All rights reserved.

"""Install reference VK plugin before collection imports test modules."""

from __future__ import annotations

import importlib
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
_ACME_REFERENCE_VK_PLUGIN_PKG = (
    _REPO_ROOT
    / "examples"
    / "platform_plugins"
    / "intergrax_reference_vendor_knowledge_plugin"
)


def _install_acme_reference_vk_plugin_for_collection() -> None:
    if importlib.util.find_spec("acme_reference_vk_plugin") is not None:
        return
    python = sys.executable
    uv = shutil.which("uv")
    if uv is not None:
        subprocess.check_call(
            [uv, "pip", "install", str(_ACME_REFERENCE_VK_PLUGIN_PKG), "--python", python],
            cwd=str(_REPO_ROOT),
        )
    else:
        subprocess.check_call(
            [python, "-m", "pip", "install", str(_ACME_REFERENCE_VK_PLUGIN_PKG), "-q"],
            cwd=str(_REPO_ROOT),
        )
    importlib.import_module("acme_reference_vk_plugin")


def pytest_configure(config) -> None:  # noqa: ANN001, ARG001
    _install_acme_reference_vk_plugin_for_collection()
