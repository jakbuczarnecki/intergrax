# © Artur Czarnecki. All rights reserved.

"""EBH-2D-D-R3 — Tier-3 host settings runtime-DI elimination gates."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_APPLICATIONS = _REPO_ROOT / "applications"
_SCAFFOLD_PRODUCT = _REPO_ROOT / "intergrax/scaffold/new_application_product.py"

_FORBIDDEN_SETTINGS_RUNTIME_IMPORTS = frozenset(
    {
        "DecisionRequirementPolicy",
        "ExternalWorkIntegration",
        "HostAttestor",
        "ActiveExecutionTaskScopePort",
        "MeaningfulSideEffectAuthorizationBoundary",
        "WebSearchExecutor",
    }
)

_FORBIDDEN_SETTINGS_FIELD_NAMES = frozenset(
    {
        "orchestration_decision_requirement_policy",
        "websearch_executor",
        "external_work_integration",
        "meaningful_side_effect_authorization_boundary",
        "decision_requirement_policy",
        "collaborative_work_repositories",
        "active_execution_task_scope",
        "host_attestor",
    }
)

_FORBIDDEN_DI_ATTR_RE = re.compile(
    r"settings\.(" + "|".join(_FORBIDDEN_SETTINGS_FIELD_NAMES) + r")\b"
)

_ESCAPE_ANNOTATION_NAMES = frozenset({"object", "Any"})


def _discover_host_settings_modules() -> list[Path]:
    return sorted(_APPLICATIONS.glob("**/host/settings.py"))


def _settings_dataclass_names(tree: ast.Module) -> list[str]:
    names: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name) and dec.id == "dataclass":
                names.append(node.name)
                break
            if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name) and dec.func.id == "dataclass":
                names.append(node.name)
                break
    return names


def _field_annotation_escape(node: ast.AnnAssign) -> bool:
    if node.annotation is None:
        return False
    ann = ast.unparse(node.annotation)
    if "object" in ann.split() or ann.strip() in _ESCAPE_ANNOTATION_NAMES:
        return True
    if "Any" in ann:
        return True
    return False


@pytest.mark.parametrize("settings_path", _discover_host_settings_modules(), ids=lambda p: p.parents[1].name)
def test_host_settings_module_has_no_forbidden_runtime_imports(settings_path: Path) -> None:
    tree = ast.parse(settings_path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported.add(alias.name.split(".")[-1])
    assert imported.isdisjoint(_FORBIDDEN_SETTINGS_RUNTIME_IMPORTS)


@pytest.mark.parametrize("settings_path", _discover_host_settings_modules(), ids=lambda p: p.parents[1].name)
def test_host_settings_has_no_forbidden_runtime_fields(settings_path: Path) -> None:
    tree = ast.parse(settings_path.read_text(encoding="utf-8"))
    class_names = set(_settings_dataclass_names(tree))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name not in class_names:
            continue
        for child in node.body:
            if not isinstance(child, ast.AnnAssign) or not isinstance(child.target, ast.Name):
                continue
            assert child.target.id not in _FORBIDDEN_SETTINGS_FIELD_NAMES, (
                f"{settings_path}: forbidden runtime field {child.target.id}"
            )
            assert not _field_annotation_escape(child), (
                f"{settings_path}: object/Any escape on field {child.target.id}"
            )


def test_host_code_does_not_consume_removed_settings_runtime_fields() -> None:
    host_py = sorted(_APPLICATIONS.glob("**/host/*.py"))
    violations: list[str] = []
    for path in host_py:
        if path.name == "settings.py":
            continue
        source = path.read_text(encoding="utf-8")
        for match in _FORBIDDEN_DI_ATTR_RE.finditer(source):
            violations.append(f"{path.relative_to(_REPO_ROOT)}:{match.group(0)}")
    assert not violations, "\n".join(violations)


def test_scaffold_product_settings_template_has_no_policy_runtime_field() -> None:
    source = _SCAFFOLD_PRODUCT.read_text(encoding="utf-8")
    settings_tpl_start = source.index("def settings_py")
    settings_tpl_end = source.index("def integration_wiring_py")
    settings_tpl = source[settings_tpl_start:settings_tpl_end]
    assert "orchestration_decision_requirement_policy:" not in settings_tpl
    assert "host_runtime_composition_py" in source
    assert "resolved_host_runtime.orchestration_decision_requirement_policy" in source


def test_research_settings_has_no_websearch_executor_field() -> None:
    path = _APPLICATIONS / "research_application/host/settings.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            assert node.target.id != "websearch_executor"
