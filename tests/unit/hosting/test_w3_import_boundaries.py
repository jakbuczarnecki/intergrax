# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import inspect
import pkgutil

import pytest

pytestmark = pytest.mark.unit

_FORBIDDEN = (
    "intergrax.runtime.task",
    "intergrax.runtime.nexus",
    "intergrax.agents",
    "fastapi",
    "uvicorn",
    "local_workspace_application",
    "nexus_loop",
)

_W3_MODULES = (
    "intergrax.hosting.instance",
    "intergrax.hosting.instance.file_guard",
    "intergrax.hosting.control",
    "intergrax.hosting.shutdown",
    "intergrax.hosting.signals",
    "intergrax.hosting.supervisor",
    "intergrax.hosting.supervisor.supervisor",
)


@pytest.mark.parametrize("module_name", _W3_MODULES)
def test_w3_import_boundaries(module_name: str) -> None:
    import importlib

    module = importlib.import_module(module_name)
    source = inspect.getsource(module)
    lowered = source.lower()
    for fragment in _FORBIDDEN:
        assert fragment not in lowered, f"{module_name} references forbidden fragment {fragment}"


def test_signals_only_module_may_import_signal() -> None:
    import importlib

    package = importlib.import_module("intergrax.hosting")
    for module_info in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        if module_info.name.endswith("signals"):
            continue
        if "instance._native_lock" in module_info.name:
            continue
        module = importlib.import_module(module_info.name)
        assert "import signal" not in inspect.getsource(module), module_info.name
