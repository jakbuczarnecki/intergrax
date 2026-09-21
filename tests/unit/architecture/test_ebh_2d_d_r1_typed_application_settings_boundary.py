# © Artur Czarnecki. All rights reserved.

"""EBH-2D-D-R1 — typed ApplicationBuildContext.settings boundary proofs."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.manifest import ApplicationManifest
from legal_application.host.agent_factories import build_legal_agent_from_context
from legal_application.host.settings import LegalBackendSettings
from research_application.host.agent_builders import build_research_agent_builders
from research_application.host.settings import ResearchBackendSettings

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LEGAL_FACTORY = _REPO_ROOT / "applications/legal_application/host/agent_factories.py"
_RESEARCH_BUILDERS = _REPO_ROOT / "applications/research_application/host/agent_builders.py"
_EXTERNAL_FACTORY = (
    _REPO_ROOT
    / "platform_proofs/scenarios/ai_incident_investigation/integration/agent_factory.py"
)


def _first_param_annotation(func: object) -> str:
    sig = inspect.signature(func)
    first = next(iter(sig.parameters.values()))
    return str(first.annotation)


def test_for_manifest_preserves_concrete_settings_instance() -> None:
    manifest = ApplicationManifest.lab(app_id="t", name="T", agents=[])
    legal_settings = LegalBackendSettings.from_env()
    ctx = ApplicationBuildContext.for_manifest(manifest, settings=legal_settings)
    assert isinstance(ctx, ApplicationBuildContext)
    assert ctx.settings is legal_settings


def test_for_manifest_overloads_declared_in_build_context() -> None:
    path = _REPO_ROOT / "intergrax/applications/contracts/build_context.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for_manifest_defs = 0
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "ApplicationBuildContext":
            continue
        for child in node.body:
            if isinstance(child, ast.FunctionDef) and child.name == "for_manifest":
                for_manifest_defs += 1
    assert for_manifest_defs >= 3


def test_legal_factory_annotates_typed_build_context() -> None:
    ann = _first_param_annotation(build_legal_agent_from_context)
    assert "ApplicationBuildContext" in ann
    assert "LegalBackendSettings" in ann
    assert "isinstance" not in _LEGAL_FACTORY.read_text(encoding="utf-8")


def test_research_factory_annotates_typed_build_context() -> None:
    builders = build_research_agent_builders()
    factory = builders[__import__("research.research_agent", fromlist=["ResearchAgent"]).ResearchAgent]
    ann = _first_param_annotation(factory)
    assert "ResearchBackendSettings" in ann
    source = _RESEARCH_BUILDERS.read_text(encoding="utf-8")
    assert "isinstance(settings, ResearchBackendSettings)" not in source


def test_external_scenario_factory_annotates_typed_build_context() -> None:
    tree = ast.parse(_EXTERNAL_FACTORY.read_text(encoding="utf-8"))
    nested_factory: ast.FunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "build_incident_investigator_factory":
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == "_factory":
                    nested_factory = child
    assert nested_factory is not None
    ctx_param = nested_factory.args.args[0]
    assert ctx_param.annotation is not None
    ann = ast.unparse(ctx_param.annotation)
    assert "IncidentInvestigatorProductionSettings" in ann


def test_build_context_module_has_no_settings_any() -> None:
    path = _REPO_ROOT / "intergrax/applications/contracts/build_context.py"
    source = path.read_text(encoding="utf-8")
    assert "settings: Any" not in source
    assert "from typing import Any" not in source
