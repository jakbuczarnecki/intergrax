# © Artur Czarnecki. All rights reserved.

"""GR-4 — policy neutral core must not accumulate Nexus coupling."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
POLICY_ROOT = REPO_ROOT / "intergrax" / "runtime" / "policy"

# Documented adapter/wiring modules (GOV-GAP-011 residual — post GR-4-R1).
NEXUS_COUPLING_ALLOWLIST = frozenset(
    {
        "compliance_profiles.py",
        "execution_mode_defaults.py",
        "policy_trace_diagnostics.py",
        "declarative_enforcer.py",
    }
)


def _nexus_imports_in_tree(tree: ast.AST) -> list[str]:
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("intergrax.runtime.nexus"):
                found.append(node.module)
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("intergrax.runtime.nexus"):
                    found.append(alias.name)
    return found


def test_policy_neutral_core_has_no_undocumented_nexus_imports() -> None:
    violations: list[str] = []
    for path in sorted(POLICY_ROOT.rglob("*.py")):
        if path.name in NEXUS_COUPLING_ALLOWLIST:
            continue
        if path.name == "__init__.py":
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        imports = _nexus_imports_in_tree(tree)
        if imports:
            violations.append(f"{rel}: {sorted(set(imports))}")
    assert violations == [], "undocumented Nexus imports in policy core:\n" + "\n".join(violations)


def test_policy_bundle_and_tool_resolution_have_zero_nexus_imports() -> None:
    for name in ("policy_bundle.py", "tool_policy_resolution.py"):
        path = POLICY_ROOT / name
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        imports = _nexus_imports_in_tree(tree)
        assert imports == [], f"{name} must not import Nexus: {imports}"


def test_runtime_policy_bundle_evaluator_has_no_rule_id_suffix_dispatch() -> None:
    path = POLICY_ROOT / "runtime_policy_bundle_evaluator.py"
    source = path.read_text(encoding="utf-8-sig")
    assert "endswith(" not in source
    assert "rule_id." not in source
