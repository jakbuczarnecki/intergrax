# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Runtime context passed to Tier-3 agent factories (Phase N.2.1)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import ApplicationManifest
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle
from intergrax.skills.execution_binding import SkillExecutionPinningStore
from intergrax.skills.registry.profile import SkillProfile
from intergrax.skills.registry.runtime import SkillRegistry
from intergrax.tools.registry.profile import ToolProfile
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.prompts.registry.yaml_registry import YamlPromptRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

if TYPE_CHECKING:
    from intergrax.runtime.attestation.buffer import BoundaryEventBuffer


@dataclass(frozen=True)
class ApplicationBuildContext:
    """
    Inputs available when materializing agents for an application host.

    ``settings`` is application-specific (e.g. ``LabApplicationSettings``,
    ``LegalBackendSettings``). Factories read env-backed settings here — not
    from global process env directly.
    """

    manifest: Any
    settings: Any = None
    integration_profile: IntegrationProfile | None = None
    tool_profile: ToolProfile | None = None
    tool_wiring_context: ToolWiringContext | None = None
    skill_profile: SkillProfile | None = None
    skill_registry: SkillRegistry | None = None
    skill_pinning_store: SkillExecutionPinningStore | None = None
    tool_registry: ToolRegistry | None = None
    policy_bundle: RuntimePolicyBundle | None = None
    runtime_event_bus: RuntimeEventBus | None = None
    strict_harness: bool = False
    trace_db_path: Path | None = None
    environment: ApplicationEnvironmentProfile | None = None
    prompt_registry: YamlPromptRegistry | None = None
    boundary_event_buffer: BoundaryEventBuffer | None = None

    @classmethod
    def for_manifest(
        cls,
        manifest: ApplicationManifest | Any,
        *,
        settings: Any = None,
        integration_profile: IntegrationProfile | None = None,
        tool_profile: ToolProfile | None = None,
        tool_wiring_context: ToolWiringContext | None = None,
        skill_profile: SkillProfile | None = None,
        skill_registry: SkillRegistry | None = None,
        skill_pinning_store: SkillExecutionPinningStore | None = None,
        tool_registry: ToolRegistry | None = None,
        policy_bundle: RuntimePolicyBundle | None = None,
        runtime_event_bus: RuntimeEventBus | None = None,
        strict_harness: bool = False,
        trace_db_path: Path | None = None,
        environment: ApplicationEnvironmentProfile | None = None,
        prompt_registry: YamlPromptRegistry | None = None,
        boundary_event_buffer: BoundaryEventBuffer | None = None,
    ) -> ApplicationBuildContext:
        resolved_profile = integration_profile
        if resolved_profile is None and isinstance(manifest, ApplicationManifest):
            resolved_profile = manifest.integration_profile
        return cls(
            manifest=manifest,
            settings=settings,
            integration_profile=resolved_profile,
            tool_profile=tool_profile,
            tool_wiring_context=tool_wiring_context,
            skill_profile=skill_profile,
            skill_registry=skill_registry,
            skill_pinning_store=skill_pinning_store,
            tool_registry=tool_registry,
            policy_bundle=policy_bundle,
            runtime_event_bus=runtime_event_bus,
            strict_harness=strict_harness,
            trace_db_path=trace_db_path,
            environment=environment,
            prompt_registry=prompt_registry,
            boundary_event_buffer=boundary_event_buffer,
        )
