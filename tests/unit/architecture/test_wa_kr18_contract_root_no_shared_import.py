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


_FORBIDDEN_PREFIXES = ("intergrax.runtime.", "intergrax.applications._shared.")
_FORBIDDEN_EXACT = frozenset(
    {"intergrax.agents.agent_contract", "intergrax.rag.bootstrap", "intergrax.fastapi_core.config"}
)
_REGISTRY_EXEMPT_PREFIXES = (
    "intergrax.integrations.contracts.",
    "intergrax.llm_adapters.contracts.",
    "intergrax.tools.contracts.",
    "intergrax.skills.contracts.",
)


def _intergrax_imports(tree: ast.AST) -> list[tuple[int, str]]:
    """Every ``intergrax.*`` import of the module, function-level and ``TYPE_CHECKING`` included."""
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            hits.extend((node.lineno, a.name) for a in node.names if a.name.startswith("intergrax."))
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("intergrax."):
            hits.append((node.lineno, node.module))
    return hits


def _rule_violations(imported: str) -> list[str]:
    """The full rule of the KR-18 register node (test_ebh_2d_d_..._recertification)."""
    problems: list[str] = []
    if imported.startswith(_FORBIDDEN_PREFIXES):
        problems.append("forbidden prefix")
    if ".registry." in f".{imported}." and not imported.startswith(_REGISTRY_EXEMPT_PREFIXES):
        problems.append("registry edge")
    if imported in _FORBIDDEN_EXACT or any(imported.startswith(f"{name}.") for name in _FORBIDDEN_EXACT):
        problems.append("forbidden module")
    return problems


def test_nt_wa_kr18_1_contract_tree_has_no_shared_import() -> None:
    violations: list[str] = []
    files = sorted(_CONTRACTS_TREE.rglob("*.py"))
    assert files
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        rel = path.relative_to(_REPO).as_posix()
        for line, imported in _intergrax_imports(tree):
            violations.extend(f"{rel}:{line} -> {imported} ({why})" for why in _rule_violations(imported))
    assert violations == []


def test_nt_wa_kr18_3_tool_availability_catalog_import_is_lazy() -> None:
    path = _REPO / "intergrax" / "tools" / "contracts" / "tool_availability.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    module_level = [
        node.module if isinstance(node, ast.ImportFrom) else alias.name
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in (node.names if isinstance(node, ast.Import) else [None])
        if (node.module if isinstance(node, ast.ImportFrom) else alias.name)
    ]
    assert not [m for m in module_level if m.startswith("intergrax.tools.registry")], module_level
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "available_tool_ids_for_profile"
    )
    inner = {node.module for node in ast.walk(function) if isinstance(node, ast.ImportFrom) and node.module}
    assert {"intergrax.tools.registry.catalog", "intergrax.tools.registry.profile"} <= inner
    root = (_REPO / "intergrax/applications/contracts/environment_profile/root.py").read_text(encoding="utf-8")
    assert "from intergrax.tools.contracts.tool_availability import available_tool_ids_for_profile" in root


def test_nt_wa_kr18_4_single_definition_and_unchanged_result() -> None:
    from intergrax.skills.registry import tool_requirements
    from intergrax.tools.contracts import tool_availability
    from intergrax.tools.registry.catalog import list_catalog_tool_ids

    assert tool_requirements.available_tool_ids_for_profile is tool_availability.available_tool_ids_for_profile
    function = tool_availability.available_tool_ids_for_profile
    assert function(ToolProfile(register_all_catalog_bundles=True)) == tuple(sorted(list_catalog_tool_ids()))
    explicit = ToolProfile(enabled=[" workspace.search ", "workspace.write_file", ""])
    assert {"workspace.search", "workspace.write_file"} <= set(function(explicit))
    assert function(explicit) == tuple(sorted(function(explicit)))
    lab = ApplicationEnvironmentProfile.lab_defaults(profile_id="wa.kr18.lab").capabilities.tools
    assert function(lab) == tuple(sorted(set(function(lab))))


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
