# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Minimal integration catalog for lab / fast cold-start (Phase P-Ext.4)."""

from __future__ import annotations


def register_core_integrations(*, override: bool = False) -> None:
    """Register harness/lab-essential integration providers (~12 slugs)."""
    from intergrax.integrations.providers.key_value_cache.redis.register import register_redis_integration
    from intergrax.integrations.providers.notification_channel.log.register import register_log_integration
    from intergrax.integrations.providers.notification_channel.slack.register import register_slack_integration
    from intergrax.integrations.providers.notification_channel.webhook.register import (
        register_webhook_integration,
    )
    from intergrax.integrations.providers.observability_backend.otel.register import register_otel_integration
    from intergrax.integrations.providers.observability_backend.prometheus.register import (
        register_prometheus_integration,
    )
    from intergrax.integrations.providers.relational_store.sqlite.register import register_sqlite_integration
    from intergrax.integrations.providers.search_provider.bing.register import register_bing_integration
    from intergrax.integrations.providers.search_provider.google_cse.register import (
        register_google_cse_integration,
    )
    from intergrax.integrations.providers.vector_store.inmemory.register import register_inmemory_integration
    from intergrax.integrations.providers.vector_store.qdrant.register import register_qdrant_integration

    for register_fn in (
        register_redis_integration,
        register_sqlite_integration,
        register_google_cse_integration,
        register_bing_integration,
        register_log_integration,
        register_inmemory_integration,
        register_qdrant_integration,
        register_otel_integration,
        register_prometheus_integration,
        register_slack_integration,
        register_webhook_integration,
    ):
        register_fn(override=override)
