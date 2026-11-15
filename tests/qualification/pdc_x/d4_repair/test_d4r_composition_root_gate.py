# © Artur Czarnecki. All rights reserved.

"""T-IGX-041-D / F-D4-05 (gate part): the single effective composition root is pinned.

Every product ``host/wiring.py`` composes through ``wire_application_environment``. No product
``host/wiring.py`` or ``host/factory.py`` builds a second tool wiring, opts out of conformance
or imports a product tool wiring function, except the declared exceptions below.

Declared exception (finding F-D4-05, not repaired in T-IGX-041-D):
    ``applications/lab_application/host/factory.py`` imports ``wire_lab_tools`` for the MCP path.
    It builds its own tool registry (hard-coded tenant ``lab``) beside the composition root.
    Reopen condition: the Operator decides to move the lab MCP tools under the composition root;
    then delete the exception and this test fails if it is left behind.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

_ROOT = Path(__file__).resolve().parents[4]
_EXPECTED_PRODUCT_HOSTS = frozenset(
    {
        "attestation_demo",
        "dispute_sim_application",
        "governed_contractor_application",
        "intergrax_assistant_application",
        "lab_application",
        "legal_application",
        "local_workspace_application",
        "poc_template_application",
        "research_application",
    }
)
DECLARED_EXCEPTIONS: frozenset[tuple[str, str]] = frozenset(
    {("applications/lab_application/host/factory.py", "wire_lab_tools")}
)


def _calls(tree: ast.AST) -> list[ast.Call]:
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)]


def _callee_name(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def composition_violations(
    sources: dict[str, str],
    exceptions: frozenset[tuple[str, str]] = DECLARED_EXCEPTIONS,
) -> list[str]:
    """Return violations of the composition-root rules over ``{relative path: source}``."""
    violations: list[str] = []
    for path, source in sorted(sources.items()):
        tree = ast.parse(source)
        if path.endswith("/host/wiring.py"):
            if "wire_application_environment" not in {_callee_name(call) for call in _calls(tree)}:
                violations.append(f"{path}: does not call wire_application_environment")
        for call in _calls(tree):
            if _callee_name(call) == "build_application_tool_wiring":
                violations.append(f"{path}: builds a second tool wiring (build_application_tool_wiring)")
            for keyword in call.keywords:
                if (
                    keyword.arg == "conformance_check"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is False
                ):
                    violations.append(f"{path}: opts out of conformance (conformance_check=False)")
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").endswith(".host.tool_wiring"):
                for alias in node.names:
                    if (path, alias.name) not in exceptions:
                        violations.append(f"{path}: undeclared product tool wiring import {alias.name}")
    return violations


def _read_host_sources() -> dict[str, str]:
    sources: dict[str, str] = {}
    for name in ("wiring.py", "factory.py"):
        for file in sorted((_ROOT / "applications").glob(f"*/host/{name}")):
            sources[file.relative_to(_ROOT).as_posix()] = file.read_text(encoding="utf-8")
    return sources


def test_the_product_hosts_hold_the_composition_root_rules() -> None:
    sources = _read_host_sources()
    found = {path.split("/")[1] for path in sources if path.endswith("/host/wiring.py")}
    assert _EXPECTED_PRODUCT_HOSTS <= found
    assert composition_violations(sources) == []


def test_every_declared_exception_is_still_in_use() -> None:
    sources = _read_host_sources()
    for path, name in DECLARED_EXCEPTIONS:
        assert path in sources, path
        imported = {
            alias.name
            for node in ast.walk(ast.parse(sources[path]))
            if isinstance(node, ast.ImportFrom) and (node.module or "").endswith(".host.tool_wiring")
            for alias in node.names
        }
        assert name in imported, f"stale exception, remove it from DECLARED_EXCEPTIONS: {path} {name}"


def test_the_gate_detects_a_missing_root_call() -> None:
    sources = {"applications/x_app/host/wiring.py": "def build():\n    return 1\n"}
    assert composition_violations(sources) == [
        "applications/x_app/host/wiring.py: does not call wire_application_environment"
    ]


def test_the_gate_detects_a_second_tool_wiring() -> None:
    source = "def f():\n    wire_application_environment()\n    return build_application_tool_wiring()\n"
    violations = composition_violations({"applications/x_app/host/wiring.py": source})
    assert any("second tool wiring" in item for item in violations)


def test_the_gate_detects_a_conformance_opt_out() -> None:
    source = "def f():\n    wire_application_environment(conformance_check=False)\n"
    violations = composition_violations({"applications/x_app/host/wiring.py": source})
    assert any("conformance_check=False" in item for item in violations)


def test_the_gate_detects_an_undeclared_tool_wiring_import() -> None:
    source = "from x_app.host.tool_wiring import wire_x_tools\n"
    violations = composition_violations({"applications/x_app/host/factory.py": source})
    assert violations == ["applications/x_app/host/factory.py: undeclared product tool wiring import wire_x_tools"]
    # The same import is accepted only where it is declared.
    declared = frozenset({("applications/x_app/host/factory.py", "wire_x_tools")})
    assert composition_violations({"applications/x_app/host/factory.py": source}, declared) == []
