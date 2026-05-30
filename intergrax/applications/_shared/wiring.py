# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Unified Tier-3 agent materialization (Phase N.2.1).

Manifest roster + :class:`ApplicationBuildContext` + typed factories/builders.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any, Callable, Union

from intergrax.agents.agent_contract import Agent
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.factory import AgentFactory
from intergrax.applications.contracts.errors import (
    AgentImportError,
    ApplicationManifestConformanceError,
)
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.runtime.registry.agent_registry import AgentRegistry

BuilderMap = Union[
    Mapping[type[Agent], AgentFactory],
    Mapping[str, AgentFactory],
]


def load_callable(import_path: str) -> Callable[..., Any]:
    """Resolve ``package.module.function`` to a callable (serialized manifests)."""
    module_path, _, attr_name = import_path.rpartition(".")
    if not module_path or not attr_name:
        raise AgentImportError(f"Invalid callable path: {import_path!r}")

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        raise AgentImportError(
            f"Cannot import module {module_path!r} for {import_path!r}"
        ) from exc

    try:
        target = getattr(module, attr_name)
    except AttributeError as exc:
        raise AgentImportError(
            f"Module {module_path!r} has no attribute {attr_name!r}"
        ) from exc

    if not callable(target):
        raise AgentImportError(f"{import_path!r} is not callable")
    return target


def resolve_builder(
    binding: AgentBinding,
    builders: BuilderMap | None,
) -> Callable[..., Any] | None:
    """Resolve factory: typed callable on binding, then type-keyed, then string key."""
    if binding.factory is not None:
        return binding.factory

    if builders is None:
        return None

    agent_type = binding.resolved_agent_type()
    if agent_type in builders:
        return builders[agent_type]  # type: ignore[index]

    if binding.builder_key is not None and binding.builder_key in builders:
        return builders[binding.builder_key]  # type: ignore[index]

    return None


def invoke_agent_factory(
    factory: Callable[..., Any],
    ctx: ApplicationBuildContext,
    binding: AgentBinding,
) -> Agent:
    """
    Call a Tier-3 factory with the canonical ``(ctx, binding)`` signature.

    Falls back to ``(settings,)``, ``(ctx,)``, or ``()`` for legacy factories.
    """
    attempts: list[tuple[tuple[Any, ...], dict[str, Any]]] = [
        ((ctx, binding), {}),
    ]
    if ctx.settings is not None:
        attempts.append(((ctx.settings,), {}))
    attempts.extend([((ctx,), {}), ((), {})])

    last_error: Exception | None = None
    for args, kwargs in attempts:
        try:
            result = factory(*args, **kwargs)
        except TypeError as exc:
            last_error = exc
            continue
        if not isinstance(result, Agent):
            raise AgentImportError(
                f"Factory {factory!r} must return Agent, got {type(result)!r}"
            )
        expected = binding.resolved_agent_type()
        if not isinstance(result, expected):
            raise AgentImportError(
                f"Factory for {binding.display_name()} returned {type(result)!r}, "
                f"expected instance of {expected.__name__}"
            )
        return result

    message = f"Cannot invoke factory for {binding.display_name()!r}"
    if last_error is not None:
        raise AgentImportError(message) from last_error
    raise AgentImportError(message)


def build_agent_from_binding(
    binding: AgentBinding,
    ctx: ApplicationBuildContext,
    *,
    builders: BuilderMap | None = None,
) -> Agent:
    """Materialize one agent: typed factory → builders map → serialized path → ctor."""
    factory = resolve_builder(binding, builders)
    if factory is not None:
        return invoke_agent_factory(factory, ctx, binding)

    if binding.factory_path is not None and binding.factory is None:
        loaded = load_callable(binding.factory_path)
        return invoke_agent_factory(loaded, ctx, binding)

    agent_cls = binding.resolved_agent_type()
    try:
        agent = agent_cls()
    except TypeError as exc:
        raise AgentImportError(
            f"{binding.display_name()} requires factory= on AgentBinding.mount() "
            f"or a builders entry (zero-arg construction failed)"
        ) from exc
    if not isinstance(agent, agent_cls):
        raise AgentImportError(f"{binding.display_name()}: constructor did not return {agent_cls.__name__}")
    return agent


def contract_for_binding(agent: Agent, binding: AgentBinding) -> AgentContract | None:
    if binding.contract_id is None:
        return None
    return agent.get_contract().model_copy(update={"id": binding.contract_id})


def validate_manifest_wiring(manifest: ApplicationManifest) -> list[str]:
    errors: list[str] = []
    if not manifest.enabled_agents():
        errors.append(f"{manifest.app_id}: no enabled agents in roster")

    seen_contract_ids: dict[str, str] = {}
    for binding in manifest.enabled_agents():
        contract_id = binding.contract_id
        if contract_id is None:
            continue
        prior = seen_contract_ids.get(contract_id)
        label = binding.display_name()
        if prior is not None:
            errors.append(
                f"{manifest.app_id}: duplicate contract_id {contract_id!r} "
                f"({prior} and {label})"
            )
        else:
            seen_contract_ids[contract_id] = label

    if len([b for b in manifest.enabled_agents() if b.default]) > 1:
        errors.append(f"{manifest.app_id}: multiple default agents")

    return errors


def build_application_registry(
    manifest: ApplicationManifest,
    ctx: ApplicationBuildContext,
    *,
    builders: BuilderMap | None = None,
    require_enabled: bool = True,
) -> AgentRegistry:
    """Canonical Tier-3 registry builder: manifest roster + context + optional builders."""
    structural = validate_manifest_wiring(manifest)
    if structural:
        raise ApplicationManifestConformanceError("; ".join(structural))

    if require_enabled:
        manifest.require_enabled_agents()

    registry = AgentRegistry()
    for binding in manifest.agents:
        if not binding.enabled:
            continue
        agent = build_agent_from_binding(binding, ctx, builders=builders)
        registry.register(agent, contract=contract_for_binding(agent, binding))

    return registry


def build_registry_from_manifest(
    manifest: ApplicationManifest,
    *,
    settings: Any = None,
    builders: BuilderMap | None = None,
    require_enabled: bool = True,
) -> AgentRegistry:
    """Backward-compatible entry: builds context from manifest + optional settings."""
    ctx = ApplicationBuildContext.for_manifest(manifest, settings=settings)
    return build_application_registry(
        manifest,
        ctx,
        builders=builders,
        require_enabled=require_enabled,
    )


def load_agent_from_binding(
    binding: AgentBinding,
    ctx: ApplicationBuildContext | None = None,
    *,
    builders: BuilderMap | None = None,
) -> Agent:
    """Build one agent; synthesizes minimal context when ``ctx`` is omitted."""
    if ctx is None:
        ctx = ApplicationBuildContext.for_manifest(
            ApplicationManifest.lab(
                app_id="inline",
                name="Inline",
                agents=[binding],
            )
        )
    return build_agent_from_binding(binding, ctx, builders=builders)


def load_agent_class(import_path: str) -> type[Agent]:
    """Resolve serialized class path (prefer :meth:`AgentBinding.mount` in application code)."""
    from intergrax.applications.contracts.agent_ref import resolve_agent_type

    return resolve_agent_type(agent_type=None, import_path=import_path)
