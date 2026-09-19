# © Artur Czarnecki. All rights reserved.

"""EBH-2D-A / EBH-2D-A-R1 — ApplicationBuildContext + factory composition boundary gate."""

from __future__ import annotations

import ast
import inspect
from dataclasses import fields, is_dataclass
from pathlib import Path

import pytest

from intergrax.agents.agent_contract import Agent
from intergrax.applications._shared.application_composition_context import (
    ApplicationCompositionContext,
    composition_for_factory_context,
    project_application_build_context,
)
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.factory import CanonicalAgentFactory
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from research.research_agent import ResearchAgent
from research_application.host.agent_builders import build_research_agent_builders

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_BUILD_CONTEXT_PATH = _REPO_ROOT / "intergrax/applications/contracts/build_context.py"
_COMPOSITION_CONTEXT_PATH = (
    _REPO_ROOT / "intergrax/applications/_shared/application_composition_context.py"
)

_FORBIDDEN_IMPORT_PREFIXES = (
    "intergrax.runtime.",
    "intergrax.skills.registry.runtime",
    "intergrax.tools.registry.runtime",
    "intergrax.tools.registry.wiring",
    "intergrax.prompts.registry.yaml_registry",
)

_FORBIDDEN_FIELD_MODULES = (
    "intergrax.runtime.",
    "intergrax.skills.registry.runtime",
    "intergrax.tools.registry.runtime",
    "intergrax.tools.registry.wiring",
    "intergrax.prompts.registry.yaml_registry",
)

_FACTORY_MODULES = (
    "applications/research_application/host/agent_builders.py",
    "applications/lab_application/host/agent_builders.py",
    "applications/intergrax_assistant_application/host/agent_builders.py",
    "applications/attestation_demo/host/agent_builders.py",
    "platform_proofs/scenarios/ai_incident_investigation/integration/agent_factory.py",
)

_AMBIENT_COMPOSITION_NAMES = frozenset(
    {
        "optional_factory_composition",
        "require_factory_composition",
        "factory_composition_scope",
        "_FACTORY_COMPOSITION",
        "ApplicationFactoryCompositionRequired",
    }
)

_COMPOSITION_MODULE = "intergrax.applications._shared.application_composition_context"


def _module_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            hits.append(node.module)
        if isinstance(node, ast.Import):
            for alias in node.names:
                hits.append(alias.name)
    return hits


def _imported_names_from_module(path: Path, module: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == module:
            for alias in node.names:
                names.add(alias.name)
    return names


def _call_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


def test_build_context_module_has_no_runtime_or_registry_imports() -> None:
    imports = _module_imports(_BUILD_CONTEXT_PATH)
    violations = [
        name
        for name in imports
        if any(
            name == prefix.rstrip(".") or name.startswith(prefix)
            for prefix in _FORBIDDEN_IMPORT_PREFIXES
        )
    ]
    assert not violations, f"forbidden imports in build_context: {violations}"


def test_public_build_context_is_frozen_dataclass_without_runtime_fields() -> None:
    assert is_dataclass(ApplicationBuildContext)
    assert ApplicationBuildContext.__dataclass_params__.frozen

    for field in fields(ApplicationBuildContext):
        annotation = field.type
        if isinstance(annotation, str):
            rendered = annotation
        else:
            rendered = str(annotation)
        assert not any(
            rendered.startswith(prefix) or f".{prefix}" in rendered
            for prefix in _FORBIDDEN_FIELD_MODULES
        ), f"field {field.name} references forbidden namespace: {rendered}"


def _minimal_lab_manifest(app_id: str, name: str) -> ApplicationManifest:
    from echo.echo_agent import EchoAgent

    return ApplicationManifest.lab(
        app_id=app_id,
        name=name,
        route_prefix=f"/v1/{app_id}",
        env_prefix=f"{app_id.upper()}_",
        agents=[AgentBinding.mount(EchoAgent, contract_id="echo", capabilities=["echo.basic"])],
    )


def test_composition_projection_does_not_leak_runtime_types_to_public_context() -> None:
    manifest = _minimal_lab_manifest("plugin_proof", "Plugin Proof")
    factory_context = ApplicationBuildContext.for_manifest(manifest)
    from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle
    from intergrax.tools.registry.runtime import ToolRegistry

    composition = composition_for_factory_context(
        factory_context,
        policy_bundle=RuntimePolicyBundle(),
        tool_registry=ToolRegistry(),
    )
    projected = project_application_build_context(composition)
    assert projected is factory_context
    assert not hasattr(projected, "tool_registry")
    assert not hasattr(projected, "policy_bundle")


def test_external_style_factory_uses_only_public_contract_imports() -> None:
    manifest = _minimal_lab_manifest("external_factory", "External")

    def external_factory(
        ctx: ApplicationBuildContext,
        binding: AgentBinding,
    ) -> Agent:
        from echo.echo_agent import EchoAgent

        _ = binding
        assert isinstance(ctx.manifest, ApplicationManifest)
        return EchoAgent()

    factory: CanonicalAgentFactory = external_factory
    ctx = ApplicationBuildContext.for_manifest(manifest)
    binding = manifest.enabled_agents()[0]
    agent = factory(ctx, binding)
    assert agent is not None


def test_composition_context_module_has_no_ambient_factory_locator() -> None:
    source = _COMPOSITION_CONTEXT_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    defined = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    assert not (_AMBIENT_COMPOSITION_NAMES & defined), (
        f"ambient factory composition locator still defined: "
        f"{sorted(_AMBIENT_COMPOSITION_NAMES & defined)}"
    )
    assert "ContextVar" not in source


@pytest.mark.parametrize("relative_path", _FACTORY_MODULES)
def test_production_factory_modules_do_not_import_ambient_composition(
    relative_path: str,
) -> None:
    path = _REPO_ROOT / relative_path
    imported = _imported_names_from_module(path, _COMPOSITION_MODULE)
    ambient_imports = imported & _AMBIENT_COMPOSITION_NAMES
    assert not ambient_imports, f"{relative_path} imports ambient accessors: {ambient_imports}"
    assert "ApplicationCompositionContext" not in imported, (
        f"{relative_path} must not import ApplicationCompositionContext"
    )
    calls = _call_names(path)
    ambient_calls = calls & _AMBIENT_COMPOSITION_NAMES
    assert not ambient_calls, f"{relative_path} calls ambient accessors: {ambient_calls}"


def test_configured_research_factory_works_without_composition_scope() -> None:
    class _ProbeProfile:
        def is_tool_enabled(self, tool_id: str) -> bool:
            return tool_id == "web.search"

    class _ProbeWiring:
        pass

    profile = _ProbeProfile()
    wiring = _ProbeWiring()
    builders = build_research_agent_builders(
        tool_profile=profile,
        tool_wiring_context=wiring,
    )
    factory = builders[ResearchAgent]
    manifest = ApplicationManifest.lab(
        app_id="research_factory_probe",
        name="Research Factory Probe",
        route_prefix="/v1/research_factory_probe",
        env_prefix="RESEARCH_FACTORY_PROBE_",
        agents=[
            AgentBinding.mount(
                ResearchAgent,
                contract_id="research",
                capabilities=["research.basic"],
            )
        ],
    )
    ctx = ApplicationBuildContext.for_manifest(manifest)
    binding = manifest.enabled_agents()[0]
    agent = factory(ctx, binding)
    assert isinstance(agent, ResearchAgent)
    assert agent._tool_profile is profile
    assert agent._tool_wiring_context is wiring


def test_canonical_agent_factory_signature_remains_two_arg() -> None:
    parameters = list(inspect.signature(CanonicalAgentFactory.__call__).parameters)
    assert parameters == ["self", "ctx", "binding"]
