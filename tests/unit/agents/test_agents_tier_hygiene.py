# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import importlib
import sys

import pytest


@pytest.mark.gate
@pytest.mark.parametrize(
    "module_name",
    [
        "echo.echo_agent",
        "problem_radar.problem_radar_agent",
        "signoff_probe.signoff_probe_agent",
        "legal.legal_agent",
    ],
)
def test_agent_module_imports_without_applications_package(module_name: str) -> None:
    for key in list(sys.modules):
        if key == "applications" or key.startswith("applications."):
            sys.modules.pop(key, None)
    mod = importlib.import_module(module_name)
    source_path = mod.__file__ or ""
    assert "agents" in source_path.replace("\\", "/")
