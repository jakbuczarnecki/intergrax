# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.applications._shared.wiring import build_application_registry
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.runtime.registry.agent_registry import AgentRegistry
from poc_template_application.host.agent_builders import POC_TEMPLATE_AGENT_BUILDERS
from poc_template_application.host.settings import PocTemplateApplicationSettings
from poc_template_application.host.tool_wiring import wire_poc_template_tools
from poc_template_application.manifest import build_poc_template_manifest


def build_poc_template_registry(
    *,
    settings: PocTemplateApplicationSettings | None = None,
) -> AgentRegistry:
    settings = settings or PocTemplateApplicationSettings.from_env()
    manifest = build_poc_template_manifest()
    tool_wiring = wire_poc_template_tools(integration_profile=getattr(manifest, "integration_profile", None))
    ctx = ApplicationBuildContext.for_manifest(
        manifest,
        settings=settings,
        tool_profile=tool_wiring.profile,
        tool_wiring_context=tool_wiring.wiring_context,
    )
    return build_application_registry(manifest, ctx, builders=POC_TEMPLATE_AGENT_BUILDERS)
