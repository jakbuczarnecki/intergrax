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
from intergrax.agent_distribution._immutable_json import distribution_json_to_plain
from intergrax.agent_distribution.roster import EffectiveRoster, EffectiveRosterEntry
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.factory import AgentFactory
from intergrax.applications.contracts.errors import (
    AgentImportError,
    ApplicationManifestConformanceError,
)
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.skills.registry.bootstrap import register_default_skills
from intergrax.skills.registry.factory import build_registry_from_profile
from intergrax.skills.registry.runtime import SkillRegistry

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

    namespace = vars(module)
    if attr_name not in namespace:
        raise AgentImportError(
            f"Module {module_path!r} has no attribute {attr_name!r}"
        )
    target = namespace[attr_name]

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

    if binding.builder_key is not None and binding.builder_key in builders:
        return builders[binding.builder_key]  # type: ignore[index]

    agent_type = binding.resolved_agent_type()
    if agent_type in builders:
        return builders[agent_type]  # type: ignore[index]

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
        if binding.agent_type is not None or binding.import_path is not None:
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


def _index_manifest_bindings(manifest: ApplicationManifest) -> dict[str, AgentBinding]:
    """Index manifest roster rows by contract id and manifest origin ref keys."""
    index: dict[str, AgentBinding] = {}
    for binding in manifest.agents:
        contract_id = binding.contract_id
        if contract_id is not None:
            index[contract_id] = binding
            index[f"manifest:agents/{contract_id}"] = binding
        import_path = binding.import_path
        if import_path is not None:
            class_name = import_path.rsplit(".", 1)[-1]
            if class_name.endswith("Agent") and len(class_name) > 5:
                stem = class_name[:-5].lower()
                index.setdefault(stem, binding)
                index.setdefault(f"manifest:agents/{stem}", binding)
    return index


def _resolve_manifest_binding_for_entry(
    entry: EffectiveRosterEntry,
    manifest_bindings: Mapping[str, AgentBinding],
) -> AgentBinding | None:
    if entry.manifest_origin_ref is not None:
        origin = entry.manifest_origin_ref
        if origin in manifest_bindings:
            return manifest_bindings[origin]
        suffix = origin.rsplit("/", 1)[-1]
        if suffix in manifest_bindings:
            return manifest_bindings[suffix]
        keyed = f"manifest:agents/{suffix}"
        if keyed in manifest_bindings:
            return manifest_bindings[keyed]
    if entry.logical_agent_id in manifest_bindings:
        return manifest_bindings[entry.logical_agent_id]
    keyed = f"manifest:agents/{entry.logical_agent_id}"
    if keyed in manifest_bindings:
        return manifest_bindings[keyed]
    return None


def binding_from_roster_entry(
    entry: EffectiveRosterEntry,
    manifest_bindings: Mapping[str, AgentBinding],
) -> AgentBinding:
    """Materialize one manifest binding row from a frozen effective roster entry."""
    base = _resolve_manifest_binding_for_entry(entry, manifest_bindings)
    roster_config = distribution_json_to_plain(dict(entry.merged_config))
    updates: dict[str, Any] = {
        "enabled": True,
        "default": entry.effective_default_agent,
        "config": roster_config,
    }
    if entry.factory_reference is not None:
        if entry.factory_reference.factory_path is not None:
            updates["factory_path"] = entry.factory_reference.factory_path
            updates["factory"] = None
        if entry.factory_reference.builder_key is not None:
            updates["builder_key"] = entry.factory_reference.builder_key
            updates["factory"] = None

    if base is not None:
        return base.model_copy(update=updates)

    if entry.factory_reference is None:
        raise ApplicationManifestConformanceError(
            f"roster entry {entry.logical_agent_id!r} lacks manifest binding and factory_reference"
        )
    return AgentBinding(
        contract_id=entry.logical_agent_id,
        **updates,
    )


def _register_binding(
    registry: AgentRegistry,
    binding: AgentBinding,
    ctx: ApplicationBuildContext,
    *,
    builders: BuilderMap | None,
    skill_registry: SkillRegistry | None,
) -> None:
    agent = build_agent_from_binding(binding, ctx, builders=builders)
    registry.register(
        agent,
        contract=contract_for_binding(agent, binding),
        skill_registry=skill_registry,
        tool_registry=ctx.tool_registry,
        event_bus=ctx.runtime_event_bus,
        requires_uaep=binding.requires_uaep,
    )


def build_application_registry(
    manifest: ApplicationManifest,
    ctx: ApplicationBuildContext,
    *,
    builders: BuilderMap | None = None,
    require_enabled: bool = True,
    effective_roster: EffectiveRoster | None = None,
) -> AgentRegistry:
    """Canonical Tier-3 registry builder: manifest roster + context + optional builders."""
    skill_registry = ctx.skill_registry
    if skill_registry is None and ctx.skill_profile is not None:
        register_default_skills()
        skill_registry = build_registry_from_profile(ctx.skill_profile)

    if effective_roster is None:
        structural = validate_manifest_wiring(manifest)
        if structural:
            raise ApplicationManifestConformanceError("; ".join(structural))

        if require_enabled:
            manifest.require_enabled_agents()

        registry = AgentRegistry()
        for binding in manifest.agents:
            if not binding.enabled:
                continue
            _register_binding(
                registry,
                binding,
                ctx,
                builders=builders,
                skill_registry=skill_registry,
            )
        return registry

    if effective_roster.application_id != manifest.app_id:
        raise ApplicationManifestConformanceError(
            f"effective roster application_id {effective_roster.application_id!r} "
            f"does not match manifest {manifest.app_id!r}"
        )

    enabled_entries = [
        entry for entry in effective_roster.entries if entry.effective_enablement
    ]
    if require_enabled and not enabled_entries:
        raise ApplicationManifestConformanceError(
            f"{manifest.app_id}: effective roster has no enabled agents"
        )
    default_entries = [entry for entry in enabled_entries if entry.effective_default_agent]
    if len(default_entries) > 1:
        raise ApplicationManifestConformanceError(
            f"{manifest.app_id}: multiple default agents in effective roster"
        )

    manifest_bindings = _index_manifest_bindings(manifest)
    registry = AgentRegistry()
    for entry in effective_roster.entries:
        if not entry.effective_enablement:
            continue
        binding = binding_from_roster_entry(entry, manifest_bindings)
        _register_binding(
            registry,
            binding,
            ctx,
            builders=builders,
            skill_registry=skill_registry,
        )
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
