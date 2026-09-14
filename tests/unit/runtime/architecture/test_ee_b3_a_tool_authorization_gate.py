# © Artur Czarnecki. All rights reserved.

"""EE-B3-A — tool invocation authorization seam gate."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[4]
_INVOKER = _REPO / "intergrax" / "runtime" / "nexus" / "tools" / "invoker.py"


def test_ee_b3_a_runtime_tool_invoker_enforces_side_effect_authorization() -> None:
    text = _INVOKER.read_text(encoding="utf-8")
    assert "require_meaningful_side_effect_authorization" in text
    assert "resolve_declarative_policy_enforcer" in text
    assert "class RuntimeToolInvoker" in text


def test_ee_b3_a_tool_registry_resolution_not_raw_import_path() -> None:
    text = _INVOKER.read_text(encoding="utf-8")
    assert "ToolRegistry" in text
    assert "importlib" not in text
