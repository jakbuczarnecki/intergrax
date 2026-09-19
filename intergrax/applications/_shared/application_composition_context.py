# © Artur Czarnecki. All rights reserved.

"""Private Tier-3 composition state — not public application contract ABI."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator

from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.manifest import ApplicationManifest
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.prompts.registry.prompt_registry_protocol import PromptRegistryProtocol
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle
from intergrax.runtime.registry.agent_registry_read import AgentRegistryRead
from intergrax.skills.execution_binding import SkillExecutionPinningStore
from intergrax.skills.registry.profile import SkillProfile
from intergrax.skills.registry.runtime import SkillRegistry
from intergrax.tools.registry.profile import ToolProfile
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

from intergrax.runtime.attestation.buffer import BoundaryEventBuffer

_FACTORY_COMPOSITION: ContextVar[ApplicationCompositionContext | None] = ContextVar(
    "intergrax_application_factory_composition",
    default=None,
)


class ApplicationFactoryCompositionRequired(RuntimeError):
    """Factory helper invoked without an active composition scope."""


@dataclass(frozen=True, slots=True)
class ApplicationCompositionContext:
    """Runtime/composition objects plus projected public factory context."""

    factory_context: ApplicationBuildContext
    integration_profile: IntegrationProfile | None = None
    tool_profile: ToolProfile | None = None
    tool_wiring_context: ToolWiringContext | None = None
    skill_profile: SkillProfile | None = None
    skill_registry: SkillRegistry | None = None
    skill_pinning_store: SkillExecutionPinningStore | None = None
    tool_registry: ToolRegistry | None = None
    policy_bundle: RuntimePolicyBundle | None = None
    runtime_event_bus: RuntimeEventBus | None = None
    prompt_registry: PromptRegistryProtocol | None = None
    boundary_event_buffer: BoundaryEventBuffer | None = None
    agent_registry: AgentRegistryRead | None = None

    @property
    def manifest(self) -> ApplicationManifest:
        return self.factory_context.manifest


def composition_for_factory_context(
    factory_context: ApplicationBuildContext,
    *,
    integration_profile: IntegrationProfile | None = None,
    tool_profile: ToolProfile | None = None,
    tool_wiring_context: ToolWiringContext | None = None,
    skill_profile: SkillProfile | None = None,
    skill_registry: SkillRegistry | None = None,
    skill_pinning_store: SkillExecutionPinningStore | None = None,
    tool_registry: ToolRegistry | None = None,
    policy_bundle: RuntimePolicyBundle | None = None,
    runtime_event_bus: RuntimeEventBus | None = None,
    prompt_registry: PromptRegistryProtocol | None = None,
    boundary_event_buffer: BoundaryEventBuffer | None = None,
    agent_registry: AgentRegistryRead | None = None,
) -> ApplicationCompositionContext:
    """Test/composition helper — attach runtime fields to a public factory context."""
    return ApplicationCompositionContext(
        factory_context=factory_context,
        integration_profile=integration_profile,
        tool_profile=tool_profile,
        tool_wiring_context=tool_wiring_context,
        skill_profile=skill_profile,
        skill_registry=skill_registry,
        skill_pinning_store=skill_pinning_store,
        tool_registry=tool_registry,
        policy_bundle=policy_bundle,
        runtime_event_bus=runtime_event_bus,
        prompt_registry=prompt_registry,
        boundary_event_buffer=boundary_event_buffer,
        agent_registry=agent_registry,
    )


def project_application_build_context(
    composition: ApplicationCompositionContext,
) -> ApplicationBuildContext:
    """Deterministic public projection — no runtime implementation fields."""
    return composition.factory_context


def optional_factory_composition() -> ApplicationCompositionContext | None:
    return _FACTORY_COMPOSITION.get()


def require_factory_composition() -> ApplicationCompositionContext:
    composition = _FACTORY_COMPOSITION.get()
    if composition is None:
        raise ApplicationFactoryCompositionRequired(
            "application factory composition scope is not active"
        )
    return composition


@contextmanager
def factory_composition_scope(
    composition: ApplicationCompositionContext | None,
) -> Iterator[None]:
    token = _FACTORY_COMPOSITION.set(composition)
    try:
        yield
    finally:
        _FACTORY_COMPOSITION.reset(token)


__all__ = [
    "ApplicationCompositionContext",
    "ApplicationFactoryCompositionRequired",
    "factory_composition_scope",
    "optional_factory_composition",
    "composition_for_factory_context",
    "project_application_build_context",
    "require_factory_composition",
]
