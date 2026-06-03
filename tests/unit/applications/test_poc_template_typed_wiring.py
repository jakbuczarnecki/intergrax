# © Artur Czarnecki. All rights reserved.

"""poc_template uses typed manifest.integration_profile (Phase H-APP.0.3)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_WIRING = (
    Path(__file__).resolve().parents[3]
    / "applications"
    / "poc_template_application"
    / "host"
    / "wiring.py"
)


def test_poc_template_wiring_no_getattr_on_manifest() -> None:
    source = _WIRING.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "getattr":
                pytest.fail("poc_template host/wiring.py must not use getattr on manifest")
