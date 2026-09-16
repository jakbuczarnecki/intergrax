# © Artur Czarnecki. All rights reserved.

"""TR-01-RQ-C1C — runtime sandbox isolation authority contract import purity."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONTRACT = _REPO_ROOT / "intergrax/contracts/runtime_sandbox_isolation_authority.py"
_ADAPTER = _REPO_ROOT / "intergrax/tools/registry/sandbox_isolation_wiring.py"
_PROVIDER_ADAPTERS = _REPO_ROOT / "intergrax/runtime/sandbox/provider_adapters.py"

_FORBIDDEN_PREFIXES = (
    "intergrax.tools",
    "intergrax.runtime",
    "intergrax.agents",
    "intergrax.applications",
)


def _imported_modules(source_path: Path) -> set[str]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_c1c_a1_contract_does_not_import_tools() -> None:
    modules = _imported_modules(_CONTRACT)
    assert not any(m == "intergrax.tools" or m.startswith("intergrax.tools.") for m in modules)


def test_c1c_a2_contract_does_not_import_runtime_agents_applications() -> None:
    modules = _imported_modules(_CONTRACT)
    for prefix in _FORBIDDEN_PREFIXES[1:]:
        assert not any(m == prefix or m.startswith(f"{prefix}.") for m in modules)


def test_c1c_a3_wiring_adapter_imports_contract_not_reverse() -> None:
    adapter_modules = _imported_modules(_ADAPTER)
    assert "intergrax.contracts.runtime_sandbox_isolation_authority" in adapter_modules
    contract_modules = _imported_modules(_CONTRACT)
    assert not any(m.startswith("intergrax.tools") for m in contract_modules)


def test_c1c_d_provider_adapters_no_positive_fabrication_for_plain_backend() -> None:
    source = _PROVIDER_ADAPTERS.read_text(encoding="utf-8")
    assert "type(backend).__name__" not in source
    assert "capabilities_from_host_backend" in source
    assert "return None" in source or "return None\n" in source
