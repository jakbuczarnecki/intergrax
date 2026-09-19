# © Artur Czarnecki. All rights reserved.

"""EE-FINAL-ARCH — production tool side effects via RuntimeToolInvoker seam."""

from __future__ import annotations

import pytest

from tests.unit.runtime.architecture._ee_final_arch_facts import _REPO_ROOT

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_AGENTS_ROOT = _REPO_ROOT / "intergrax" / "agents"
_APPROVED_INVOKER_OWNERS = (
    _REPO_ROOT
    / "intergrax"
    / "applications"
    / "_shared"
    / "declarative_tool_wiring.py",
    _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "engine" / "runtime_context.py",
)
_INVOKER_MODULE = (
    _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "tools" / "invoker.py"
)


def test_ee_final_arch_runtime_tool_invoker_module_is_canonical_gateway() -> None:
    source = _INVOKER_MODULE.read_text(encoding="utf-8")
    assert "class RuntimeToolInvoker" in source
    assert "ToolExecutor" in source


def test_ee_final_arch_agents_do_not_construct_runtime_tool_invoker() -> None:
    violations: list[str] = []
    for path in _AGENTS_ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        if "RuntimeToolInvoker(" in path.read_text(encoding="utf-8"):
            violations.append(path.relative_to(_REPO_ROOT).as_posix())
    assert violations == []


def test_ee_final_arch_approved_runtime_tool_invoker_construction_sites_exist() -> None:
    for path in _APPROVED_INVOKER_OWNERS:
        assert path.is_file()
        source = path.read_text(encoding="utf-8")
        assert (
            "RuntimeToolInvoker(" in source
            or "build_production_runtime_tool_invoker(" in source
        )
