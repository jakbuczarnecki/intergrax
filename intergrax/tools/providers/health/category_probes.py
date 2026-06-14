# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations
from intergrax.utils import attribute_access

from typing import Any

from intergrax.tools.providers.health.contracts import HealthCheckIntegrationOutput, HealthStatusOutput
from intergrax.tools.registry.wiring import ToolWiringContext

HEALTH_CHECK_OBJECT_STORAGE_TOOL_ID = "health.check_object_storage"
HEALTH_CHECK_KEY_VALUE_CACHE_TOOL_ID = "health.check_key_value_cache"
HEALTH_CHECK_MESSAGE_BUS_TOOL_ID = "health.check_message_bus"
HEALTH_CHECK_GRAPH_STORE_TOOL_ID = "health.check_graph_store"
HEALTH_CHECK_IDENTITY_PROVIDER_TOOL_ID = "health.check_identity_provider"
HEALTH_CHECK_RELATIONAL_STORE_TOOL_ID = "health.check_relational_store"
HEALTH_CHECK_WIKI_KNOWLEDGE_TOOL_ID = "health.check_wiki_knowledge"
HEALTH_CHECK_SEARCH_PROVIDER_TOOL_ID = "health.check_search_provider"
HEALTH_CHECK_NOTIFICATION_CHANNEL_TOOL_ID = "health.check_notification_channel"
HEALTH_CHECK_CODECRAFT_TOOL_ID = "health.check_codecraft"


def health_check_codecraft(ctx: ToolWiringContext, _params: object) -> HealthCheckIntegrationOutput:
    profile = ctx.extras.get("codecraft_profile")
    if profile is None:
        return HealthCheckIntegrationOutput(
            status=HealthStatusOutput(slug="codecraft", healthy=False, detail="profile_not_configured"),
        )
    if isinstance(profile, dict):
        mode = profile.get("mode")
    else:
        mode = attribute_access.optional(profile, "mode", None)
    if mode == "disabled":
        return HealthCheckIntegrationOutput(
            status=HealthStatusOutput(slug="codecraft", healthy=True, detail="mode_disabled"),
        )
    sandbox_ok = ctx.sandbox_session is not None or ctx.sandbox_host is not None
    detail = "ready" if sandbox_ok else "sandbox_not_configured"
    return HealthCheckIntegrationOutput(
        status=HealthStatusOutput(slug="codecraft", healthy=sandbox_ok, detail=detail),
    )


def _probe_backend(slug: str, backend: Any) -> HealthCheckIntegrationOutput:
    if backend is None:
        return HealthCheckIntegrationOutput(
            status=HealthStatusOutput(slug=slug, healthy=False, detail="not_configured"),
        )
    health_fn = attribute_access.optional(backend, "health", None)
    if health_fn is None:
        return HealthCheckIntegrationOutput(
            status=HealthStatusOutput(slug=slug, healthy=True, detail="no_health_method"),
        )
    try:
        result = health_fn()
        healthy = bool(attribute_access.optional(result, "healthy", True))
        detail = str(attribute_access.optional(result, "detail", "") or "ok")
        resolved_slug = str(attribute_access.optional(result, "slug", slug) or slug)
        return HealthCheckIntegrationOutput(
            status=HealthStatusOutput(slug=resolved_slug, healthy=healthy, detail=detail),
        )
    except Exception as exc:  # noqa: BLE001 — health probe must not raise
        return HealthCheckIntegrationOutput(
            status=HealthStatusOutput(slug=slug, healthy=False, detail=str(exc)),
        )


def health_check_object_storage(ctx: ToolWiringContext, _params: object) -> HealthCheckIntegrationOutput:
    return _probe_backend("object_storage", ctx.object_storage)


def health_check_key_value_cache(ctx: ToolWiringContext, _params: object) -> HealthCheckIntegrationOutput:
    return _probe_backend("key_value_cache", ctx.key_value_cache)


def health_check_message_bus(ctx: ToolWiringContext, _params: object) -> HealthCheckIntegrationOutput:
    return _probe_backend("message_bus", ctx.message_bus)


def health_check_graph_store(ctx: ToolWiringContext, _params: object) -> HealthCheckIntegrationOutput:
    return _probe_backend("graph_store", ctx.graph_store)


def health_check_identity_provider(ctx: ToolWiringContext, _params: object) -> HealthCheckIntegrationOutput:
    return _probe_backend("identity_provider", ctx.identity_provider)


def health_check_relational_store(ctx: ToolWiringContext, _params: object) -> HealthCheckIntegrationOutput:
    return _probe_backend("relational_store", ctx.relational_store)


def health_check_wiki_knowledge(ctx: ToolWiringContext, _params: object) -> HealthCheckIntegrationOutput:
    return _probe_backend("wiki_knowledge", ctx.wiki_knowledge)


def health_check_search_provider(ctx: ToolWiringContext, _params: object) -> HealthCheckIntegrationOutput:
    return _probe_backend("search_provider", ctx.search_provider)


def health_check_notification_channel(ctx: ToolWiringContext, _params: object) -> HealthCheckIntegrationOutput:
    return _probe_backend("notification_channel", ctx.notification_channel)
