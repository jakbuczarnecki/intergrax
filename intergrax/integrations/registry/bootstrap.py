# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register default P0 integration providers (Phase M.4+)."""

from __future__ import annotations

_BOOTSTRAPPED = False


def register_default_integrations(*, override: bool = False) -> None:
    """
    Idempotent registration of shipped integration providers.

    Call from Tier-3 application factories before ``resolve()``.
    """
    from intergrax.integrations.providers.redis.register import register_redis_integration
    from intergrax.integrations.providers.sqlite.register import register_sqlite_integration
    from intergrax.integrations.providers.kafka.register import register_kafka_integration
    from intergrax.integrations.providers.celery.register import register_celery_integration
    from intergrax.integrations.providers.rabbitmq.register import register_rabbitmq_integration
    from intergrax.integrations.providers.google_cse.register import register_google_cse_integration
    from intergrax.integrations.providers.bing.register import register_bing_integration
    from intergrax.integrations.providers.slack.register import register_slack_integration
    from intergrax.integrations.providers.teams.register import register_teams_integration
    from intergrax.integrations.providers.webhook.register import register_webhook_integration
    from intergrax.integrations.providers.lab_json.register import register_lab_json_integration
    from intergrax.integrations.providers.log.register import register_log_integration
    from intergrax.integrations.providers.postgresql.register import register_postgresql_integration
    from intergrax.integrations.providers.mysql.register import register_mysql_integration
    from intergrax.integrations.providers.jira.register import register_jira_integration
    from intergrax.integrations.providers.confluence.register import register_confluence_integration
    from intergrax.integrations.providers.prometheus.register import register_prometheus_integration
    from intergrax.integrations.providers.ms365_graph.register import register_ms365_graph_integration
    from intergrax.integrations.providers.cassandra.register import register_cassandra_integration

    global _BOOTSTRAPPED
    if _BOOTSTRAPPED and not override:
        return
    register_redis_integration(override=override)
    register_sqlite_integration(override=override)
    register_kafka_integration(override=override)
    register_celery_integration(override=override)
    register_rabbitmq_integration(override=override)
    register_google_cse_integration(override=override)
    register_bing_integration(override=override)
    register_slack_integration(override=override)
    register_teams_integration(override=override)
    register_webhook_integration(override=override)
    register_lab_json_integration(override=override)
    register_log_integration(override=override)
    register_postgresql_integration(override=override)
    register_mysql_integration(override=override)
    register_jira_integration(override=override)
    register_confluence_integration(override=override)
    register_prometheus_integration(override=override)
    register_ms365_graph_integration(override=override)
    register_cassandra_integration(override=override)
    _BOOTSTRAPPED = True


def reset_default_integrations_state() -> None:
    """Test helper — allow re-bootstrap after ``clear_catalog()``."""
    global _BOOTSTRAPPED
    _BOOTSTRAPPED = False
