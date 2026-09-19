# © Artur Czarnecki. All rights reserved.

"""EBH-2D-A — ApplicationBuildContext composition boundary gate."""

from __future__ import annotations

import ast
import importlib
import inspect
from dataclasses import fields, is_dataclass
from pathlib import Path

import pytest

from intergrax.agents.agent_contract import Agent
from intergrax.applications._shared.application_composition_context import (
    ApplicationCompositionContext,
    composition_for_factory_context,
    factory_composition_scope,
    project_application_build_context,
)
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.factory import CanonicalAgentFactory
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_BUILD_CONTEXT_PATH = _REPO_ROOT / "intergrax/applications/contracts/build_context.py"

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
    with factory_composition_scope(None):
        agent = factory(ctx, binding)
    assert agent is not None


def test_factory_composition_scope_enables_composition_helpers() -> None:
    manifest = _minimal_lab_manifest("scope_proof", "Scope")
    factory_context = ApplicationBuildContext.for_manifest(manifest)
    from intergrax.runtime.events.event_bus import RuntimeEventBus

    composition = composition_for_factory_context(
        factory_context,
        runtime_event_bus=RuntimeEventBus(),
    )
    with factory_composition_scope(composition):
        from intergrax.applications._shared.application_composition_context import (
            require_factory_composition,
        )

        active = require_factory_composition()
        assert active.runtime_event_bus is composition.runtime_event_bus
