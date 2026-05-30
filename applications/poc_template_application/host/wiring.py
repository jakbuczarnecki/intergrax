# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.applications._shared.wiring import build_application_registry
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.runtime.registry.agent_registry import AgentRegistry
from poc_template_application.host.agent_builders import POC_TEMPLATE_AGENT_BUILDERS
from poc_template_application.host.settings import PocTemplateApplicationSettings
from poc_template_application.manifest import build_poc_template_manifest


def build_poc_template_registry(
    *,
    settings: PocTemplateApplicationSettings | None = None,
) -> AgentRegistry:
    settings = settings or PocTemplateApplicationSettings.from_env()
    manifest = build_poc_template_manifest()
    ctx = ApplicationBuildContext.for_manifest(manifest, settings=settings)
    return build_application_registry(manifest, ctx, builders=POC_TEMPLATE_AGENT_BUILDERS)
