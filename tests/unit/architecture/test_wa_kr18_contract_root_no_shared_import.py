# © Artur Czarnecki. All rights reserved.

"""T-IGX-043-G (PT-WA, Part K): the environment-profile contract root imports no composition layer.

NT-WA-KR18-1 (structural) and NT-WA-KR18-2 (parity with the ``_shared`` helper) for KR-18.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from intergrax.applications._shared.strict_host_dependency_concurrency_admission import (
    build_reference_host_tool_dependency_concurrency_admission,
)
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    OrchestrationProfile,
)
from intergrax.contracts.dependency_concurrency_admission import (
    DependencyConcurrencyAdmissionConfiguration,
)
from intergrax.tools.contracts.tool_profile import ToolProfile

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[3]
_CONTRACTS_TREE = _REPO / "intergrax" / "applications" / "contracts"
_SHARED = "intergrax.applications._shared"


def _imports_shared(tree: ast.AST) -> list[int]:
    lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name == _SHARED or a.name.startswith(_SHARED + ".") for a in node.names):
                lines.append(node.lineno)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module == _SHARED or node.module.startswith(_SHARED + "."):
                lines.append(node.lineno)
            elif node.module == "intergrax.applications" and any(a.name == "_shared" for a in node.names):
                lines.append(node.lineno)
    return lines


def test_nt_wa_kr18_1_contract_tree_has_no_shared_import() -> None:
    violations: list[str] = []
    files = sorted(_CONTRACTS_TREE.rglob("*.py"))
    assert files
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        rel = path.relative_to(_REPO).as_posix()
        violations.extend(f"{rel}:{line}" for line in _imports_shared(tree))
    assert violations == []


_TOOL_PROFILES = (
    ToolProfile(enabled=["workspace.search", "workspace.write_file"]),
    ToolProfile(enabled=["workspace.search"]),
    ToolProfile(enabled=["workspace.search", "workspace.write_file", "workspace.read_file"]),
)


def _with_tools(tool_profile: ToolProfile):
    base = ApplicationEnvironmentProfile.lab_defaults(profile_id="wa.kr18.base")
    return base.capabilities.model_copy(update={"tools": tool_profile})


def test_nt_wa_kr18_2_inline_admission_equals_shared_helper() -> None:
    compared = 0
    for tool_profile in _TOOL_PROFILES:
        for max_inflight in (1, 8):
            base = ApplicationEnvironmentProfile.lab_defaults(profile_id="wa.kr18").model_copy(
                update={
                    "capabilities": _with_tools(tool_profile),
                    "orchestration_profile": OrchestrationProfile(max_inflight_nodes=max_inflight),
                },
            )
            assert base.reliability_profile.dependency_concurrency_admission is None
            expected = build_reference_host_tool_dependency_concurrency_admission(
                tool_profile,
                max_concurrent_calls=max_inflight,
            )
            assert expected is not None and expected.bindings
            applied = base.with_reference_host_platform_defaults()
            assert applied.reliability_profile.dependency_concurrency_admission == expected
            compared += 1
    assert compared == 6

    empty_tools = ApplicationEnvironmentProfile.lab_defaults(profile_id="wa.kr18.none").model_copy(
        update={
            "capabilities": _with_tools(ToolProfile()),
            "orchestration_profile": OrchestrationProfile(max_inflight_nodes=4),
        },
    )
    assert (
        empty_tools.with_reference_host_platform_defaults().reliability_profile.dependency_concurrency_admission is None
    )

    existing = DependencyConcurrencyAdmissionConfiguration(bindings=())
    preset = ApplicationEnvironmentProfile.product_defaults(profile_id="wa.kr18.keep")
    preset.reliability_profile = preset.reliability_profile.model_copy(
        update={"dependency_concurrency_admission": existing},
    )
    kept = preset.with_reference_host_platform_defaults()
    assert kept.reliability_profile.dependency_concurrency_admission is existing
