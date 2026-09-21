# © Artur Czarnecki. All rights reserved.

"""EBH-2D-D-R2 — settings vs host runtime composition separation gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SETTINGS = (
    _REPO_ROOT
    / "applications/governed_contractor_application/host/settings.py"
)
_AGENT_BUILDERS = (
    _REPO_ROOT
    / "applications/governed_contractor_application/host/agent_builders.py"
)
_PRODUCTION_COMPOSITION = (
    _REPO_ROOT
    / "applications/governed_contractor_application/host/production_external_work_composition.py"
)

_FORBIDDEN_SETTINGS_RUNTIME_TYPES = frozenset(
    {
        "ExternalWorkIntegration",
        "MeaningfulSideEffectAuthorizationBoundary",
        "ActiveExecutionTaskScopePort",
        "HostAttestor",
        "CollaborativeWorkMaterializedRepositories",
    }
)

_FORBIDDEN_SETTINGS_DI_ATTRS = frozenset(
    {
        "external_work_integration",
        "meaningful_side_effect_authorization_boundary",
        "decision_requirement_policy",
        "collaborative_work_repositories",
        "active_execution_task_scope",
        "host_attestor",
    }
)


def _settings_dataclass_field_names() -> set[str]:
    tree = ast.parse(_SETTINGS.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "GovernedContractorBackendSettings":
            names: set[str] = set()
            for child in node.body:
                if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                    names.add(child.target.id)
            return names
    raise AssertionError("GovernedContractorBackendSettings not found")


def test_settings_has_no_forbidden_runtime_port_fields() -> None:
    fields = _settings_dataclass_field_names()
    assert fields.isdisjoint(_FORBIDDEN_SETTINGS_DI_ATTRS)


def test_settings_module_does_not_import_forbidden_runtime_ports() -> None:
    tree = ast.parse(_SETTINGS.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported.add(alias.name)
    assert imported.isdisjoint(_FORBIDDEN_SETTINGS_RUNTIME_TYPES)


def test_agent_builders_do_not_read_runtime_services_from_settings() -> None:
    source = _AGENT_BUILDERS.read_text(encoding="utf-8")
    for attr in _FORBIDDEN_SETTINGS_DI_ATTRS:
        assert f"settings.{attr}" not in source


def test_production_composition_has_no_wire_settings_runtime_di() -> None:
    source = _PRODUCTION_COMPOSITION.read_text(encoding="utf-8")
    assert "wire_governed_contractor_production_external_work_settings" not in source
    assert "replace(" not in source


def test_host_runtime_composition_module_is_typed_frozen_dataclass() -> None:
    path = (
        _REPO_ROOT
        / "applications/governed_contractor_application/host/governed_contractor_host_runtime_composition.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = False
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if node.name != "GovernedContractorHostRuntimeComposition":
            continue
        found = True
        for dec in node.decorator_list:
            assert isinstance(dec, ast.Call)
            assert isinstance(dec.func, ast.Name) and dec.func.id == "dataclass"
    assert found
